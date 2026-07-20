"""Remote backup destinations.

The backup archive itself (build + restore) is destination-agnostic and lives in
`remote_backup`. This module owns the *transport*: how that archive gets to and
from wherever the operator chose to keep it.

Two targets are supported, and an install uses exactly one at a time
(`backup_destination` setting):

- `SftpTarget` — an SFTP server the operator controls (the original behavior).
- `S3Target`   — S3-compatible object storage. Defaults to AWS, but an explicit
  endpoint URL points it at MinIO, Wasabi, Backblaze B2, or Cloudflare R2.

Every target exposes the same five operations, so `remote_backup` never branches
on destination. Errors are returned as `(False, message)` from `test()` and
raised as exceptions elsewhere — the caller records them in `backup_last_status`.
"""

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import asyncssh

from . import database as db
from .crypto import decrypt_password, is_encrypted

logger = logging.getLogger(__name__)

SSH_KEY_PATH = Path("/app/.ssh/backup_key")

# Archive naming — both targets filter listings to this so unrelated files
# sitting in the same bucket/directory are never listed or reaped by retention.
ARCHIVE_PREFIX = "sixtyops-backup-"
ARCHIVE_SUFFIX = ".tar.gz"
ARCHIVE_RE = re.compile(r"^sixtyops-backup-[\w.\-]+\.tar\.gz$")

# Upload/download ceiling, matching the original SFTP behavior.
TRANSFER_TIMEOUT = 300


def is_archive_name(name: str) -> bool:
    """Whether a remote entry is one of our backup archives."""
    return bool(ARCHIVE_RE.match(name)) and ".." not in name


class BackupTarget(ABC):
    """A place backup archives are stored."""

    @abstractmethod
    async def test(self) -> Tuple[bool, str]:
        """Verify credentials and read/write access. Returns (ok, message)."""

    @abstractmethod
    async def upload(self, name: str, data: bytes) -> None: ...

    @abstractmethod
    async def download(self, name: str) -> bytes: ...

    @abstractmethod
    async def list_archives(self) -> list[dict]:
        """Newest first. Each entry has name, and where known size + mtime."""

    @abstractmethod
    async def delete(self, name: str) -> None: ...

    @abstractmethod
    def describe(self) -> str:
        """Human-readable destination, shown in the UI status panel."""


# ---------------------------------------------------------------------------
# SFTP
# ---------------------------------------------------------------------------

class SftpTarget(BackupTarget):
    def __init__(self, host: str, port: int, path: str, username: str,
                 auth_method: str, password: str = ""):
        self.host = host
        self.port = port
        self.path = path or "/backups/sixtyops"
        self.username = username
        self.auth_method = auth_method
        self.password = password

    @classmethod
    def from_settings(cls, settings: dict) -> "SftpTarget":
        stored_pw = settings.get("backup_sftp_password", "")
        if stored_pw and is_encrypted(stored_pw):
            stored_pw = decrypt_password(stored_pw)
        return cls(
            host=settings.get("backup_sftp_host", ""),
            port=int(settings.get("backup_sftp_port") or "22"),
            path=settings.get("backup_sftp_path", ""),
            username=settings.get("backup_sftp_username", ""),
            auth_method=settings.get("backup_sftp_auth_method", "password"),
            password=stored_pw,
        )

    def describe(self) -> str:
        if not self.host:
            return ""
        return f"{self.username}@{self.host}:{self.port}{self.path}"

    async def _connect(self):
        connect_kwargs = {
            "host": self.host,
            "port": self.port,
            "username": self.username,
            "known_hosts": None,
            "login_timeout": 30,
        }
        if self.auth_method == "key" and SSH_KEY_PATH.exists():
            connect_kwargs["client_keys"] = [str(SSH_KEY_PATH)]
        elif self.auth_method == "password":
            connect_kwargs["password"] = self.password
        return await asyncssh.connect(**connect_kwargs)

    async def test(self) -> Tuple[bool, str]:
        try:
            async with await self._connect() as conn:
                async with conn.start_sftp_client() as sftp:
                    try:
                        await sftp.stat(self.path)
                    except asyncssh.SFTPNoSuchFile:
                        try:
                            await sftp.makedirs(self.path)
                        except Exception as e:
                            return False, f"Remote path does not exist and could not be created: {e}"
                    return True, "Connection successful"
        except asyncssh.PermissionDenied:
            return False, "Authentication failed - check username/password or key"
        except asyncssh.DisconnectError as e:
            return False, f"Connection dropped: {e}"
        except Exception as e:
            return False, f"Connection failed: {e}"

    async def upload(self, name: str, data: bytes) -> None:
        async with await self._connect() as conn:
            async with conn.start_sftp_client() as sftp:
                try:
                    await sftp.stat(self.path)
                except asyncssh.SFTPNoSuchFile:
                    await sftp.makedirs(self.path)
                try:
                    async with asyncio.timeout(TRANSFER_TIMEOUT):
                        async with sftp.open(f"{self.path}/{name}", "wb") as f:
                            await f.write(data)
                except TimeoutError:
                    raise TimeoutError("Upload timed out after 5 minutes")

    async def download(self, name: str) -> bytes:
        async with await self._connect() as conn:
            async with conn.start_sftp_client() as sftp:
                async with asyncio.timeout(TRANSFER_TIMEOUT):
                    async with sftp.open(f"{self.path}/{name}", "rb") as f:
                        return await f.read()

    async def list_archives(self) -> list[dict]:
        try:
            async with await self._connect() as conn:
                async with conn.start_sftp_client() as sftp:
                    try:
                        entries = await sftp.listdir(self.path)
                    except asyncssh.SFTPNoSuchFile:
                        return []
                    names = sorted(
                        (e for e in entries if is_archive_name(e)), reverse=True
                    )
                    results = []
                    for n in names:
                        try:
                            attrs = await sftp.stat(f"{self.path}/{n}")
                            mtime = (datetime.fromtimestamp(attrs.mtime).isoformat()
                                     if attrs.mtime else None)
                            results.append({
                                "name": n,
                                "size": attrs.size,
                                "mtime": mtime,
                            })
                        except Exception:
                            results.append({"name": n})
                    return results
        except Exception as e:
            logger.warning(f"Failed to list SFTP backups: {e}")
            return []

    async def delete(self, name: str) -> None:
        async with await self._connect() as conn:
            async with conn.start_sftp_client() as sftp:
                await sftp.remove(f"{self.path}/{name}")


# ---------------------------------------------------------------------------
# S3-compatible object storage
# ---------------------------------------------------------------------------

class S3Target(BackupTarget):
    """S3-compatible object storage.

    boto3 is synchronous, so every call is pushed to a worker thread. Clients are
    built per operation rather than cached: backups run at most once a day, so
    the construction cost is irrelevant next to always picking up credential or
    endpoint changes without a restart.

    Credentials are optional. When the access key and secret are both blank we
    build the client without explicit credentials, which lets botocore fall back
    to its default chain — an EC2/ECS instance role, `AWS_*` environment
    variables, or a mounted `~/.aws/credentials`. That way an AWS-hosted install
    never has to mint long-lived keys.
    """

    def __init__(self, bucket: str, prefix: str = "", region: str = "",
                 endpoint_url: str = "", access_key_id: str = "",
                 secret_access_key: str = ""):
        self.bucket = bucket
        # Normalize to a bare prefix: no leading slash, no trailing slash.
        self.prefix = (prefix or "").strip("/")
        self.region = region
        self.endpoint_url = endpoint_url
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key

    @classmethod
    def from_settings(cls, settings: dict) -> "S3Target":
        stored_secret = settings.get("backup_s3_secret_access_key", "")
        if stored_secret and is_encrypted(stored_secret):
            stored_secret = decrypt_password(stored_secret)
        return cls(
            bucket=settings.get("backup_s3_bucket", ""),
            prefix=settings.get("backup_s3_prefix", ""),
            region=settings.get("backup_s3_region", ""),
            endpoint_url=settings.get("backup_s3_endpoint_url", ""),
            access_key_id=settings.get("backup_s3_access_key_id", ""),
            secret_access_key=stored_secret,
        )

    def describe(self) -> str:
        if not self.bucket:
            return ""
        location = f"s3://{self.bucket}"
        if self.prefix:
            location += f"/{self.prefix}"
        if self.endpoint_url:
            location += f" @ {self.endpoint_url}"
        return location

    @property
    def uses_ambient_credentials(self) -> bool:
        return not (self.access_key_id and self.secret_access_key)

    def _key(self, name: str) -> str:
        return f"{self.prefix}/{name}" if self.prefix else name

    def _client(self):
        """Build a boto3 S3 client. Raises RuntimeError if boto3 is missing.

        Imported lazily so an install that never selects S3 doesn't pay the
        import cost, and so a missing dependency surfaces as a readable message
        in the UI rather than crashing app startup.
        """
        try:
            import boto3
            from botocore.config import Config
        except ImportError as e:
            raise RuntimeError(
                "boto3 is not installed - S3 backups unavailable. "
                "Rebuild the container image to pick up the dependency."
            ) from e

        kwargs = {
            "config": Config(
                connect_timeout=30,
                read_timeout=TRANSFER_TIMEOUT,
                retries={"max_attempts": 3, "mode": "standard"},
            )
        }
        if self.region:
            kwargs["region_name"] = self.region
        if self.endpoint_url:
            kwargs["endpoint_url"] = self.endpoint_url
        if not self.uses_ambient_credentials:
            kwargs["aws_access_key_id"] = self.access_key_id
            kwargs["aws_secret_access_key"] = self.secret_access_key
        return boto3.client("s3", **kwargs)

    async def test(self) -> Tuple[bool, str]:
        """Confirm the bucket is reachable *and* writable.

        A read-only probe (head_bucket) would report success on a bucket we
        can't actually back up to, so we round-trip a marker object. That is the
        SFTP behavior too, which creates the remote directory during its test.
        """
        if not self.bucket:
            return False, "Bucket name is required"

        def _probe():
            from botocore.exceptions import ClientError, NoCredentialsError
            client = self._client()
            probe_key = self._key(".sixtyops-write-test")
            try:
                client.head_bucket(Bucket=self.bucket)
            except NoCredentialsError:
                return False, (
                    "No credentials found. Provide an access key and secret, or "
                    "attach an IAM role to this host."
                )
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                if code in ("404", "NoSuchBucket"):
                    return False, f"Bucket '{self.bucket}' does not exist"
                if code in ("403", "AccessDenied"):
                    return False, "Access denied - check credentials and bucket policy"
                if code in ("301", "PermanentRedirect"):
                    return False, "Wrong region for this bucket - check the region setting"
                return False, f"Bucket check failed: {e}"
            except (BotoCoreError, Exception) as e:
                return False, f"Connection failed: {e}"

            try:
                client.put_object(Bucket=self.bucket, Key=probe_key, Body=b"ok")
                client.delete_object(Bucket=self.bucket, Key=probe_key)
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                if code in ("403", "AccessDenied"):
                    return False, (
                        "Bucket is reachable but not writable - the credentials "
                        "need s3:PutObject and s3:DeleteObject"
                    )
                return False, f"Write test failed: {e}"
            except Exception as e:
                return False, f"Write test failed: {e}"

            creds = "instance credentials" if self.uses_ambient_credentials else "access key"
            return True, f"Connection successful (using {creds})"

        try:
            return await asyncio.to_thread(_probe)
        except RuntimeError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Connection failed: {e}"

    async def upload(self, name: str, data: bytes) -> None:
        def _put():
            self._client().put_object(
                Bucket=self.bucket, Key=self._key(name), Body=data
            )
        await asyncio.to_thread(_put)

    async def download(self, name: str) -> bytes:
        def _get():
            resp = self._client().get_object(Bucket=self.bucket, Key=self._key(name))
            return resp["Body"].read()
        return await asyncio.to_thread(_get)

    async def list_archives(self) -> list[dict]:
        def _list():
            client = self._client()
            paginator = client.get_paginator("list_objects_v2")
            list_prefix = f"{self.prefix}/{ARCHIVE_PREFIX}" if self.prefix else ARCHIVE_PREFIX
            results = []
            for page in paginator.paginate(Bucket=self.bucket, Prefix=list_prefix):
                for obj in page.get("Contents", []):
                    name = obj["Key"].rsplit("/", 1)[-1]
                    if not is_archive_name(name):
                        continue
                    modified = obj.get("LastModified")
                    results.append({
                        "name": name,
                        "size": obj.get("Size"),
                        "mtime": modified.isoformat() if modified else None,
                    })
            return sorted(results, key=lambda r: r["name"], reverse=True)

        try:
            return await asyncio.to_thread(_list)
        except Exception as e:
            logger.warning(f"Failed to list S3 backups: {e}")
            return []

    async def delete(self, name: str) -> None:
        def _delete():
            self._client().delete_object(Bucket=self.bucket, Key=self._key(name))
        await asyncio.to_thread(_delete)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_destination(settings: Optional[dict] = None) -> str:
    """The configured destination id. Defaults to sftp for pre-S3 installs."""
    if settings is None:
        settings = db.get_all_settings()
    dest = settings.get("backup_destination") or "sftp"
    return dest if dest in ("sftp", "s3") else "sftp"


def get_target(settings: Optional[dict] = None) -> BackupTarget:
    """Build the target for the configured destination."""
    if settings is None:
        settings = db.get_all_settings()
    if get_destination(settings) == "s3":
        return S3Target.from_settings(settings)
    return SftpTarget.from_settings(settings)


def is_configured(settings: Optional[dict] = None) -> bool:
    """Whether the selected destination has the minimum fields to run.

    Used by the scheduler so a half-configured install is skipped instead of
    logging a failure every morning.
    """
    if settings is None:
        settings = db.get_all_settings()
    if get_destination(settings) == "s3":
        return bool(settings.get("backup_s3_bucket"))
    return bool(settings.get("backup_sftp_host"))
