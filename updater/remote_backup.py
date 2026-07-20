"""Scheduled backup of the database, settings, and device inventory.

This module owns *what* gets backed up and the restore path. *Where* it goes is
handled by `backup_targets`, which offers an SFTP server or S3-compatible object
storage — an install picks one via the `backup_destination` setting. Nothing
below branches on destination; it asks `backup_targets.get_target()` for a
transport and calls the same operations either way.
"""

import asyncio
import io
import json
import logging
import sqlite3
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from . import backup_targets
from . import crypto
from . import database as db
from .backup_targets import is_archive_name
from .crypto import encrypt_password

logger = logging.getLogger(__name__)

# Paths - match docker-compose volume mounts
STAGING_DIR = Path("/app/backups")
DATA_DIR = Path("/app/data")
DB_FILE = DATA_DIR / "sixtyops.db"
SSH_KEY_PATH = backup_targets.SSH_KEY_PATH

# Prevent concurrent backup runs
_backup_lock = asyncio.Lock()


def get_backup_status() -> dict:
    """Get current backup configuration and status."""
    settings = db.get_all_settings()
    destination = backup_targets.get_destination(settings)
    target = backup_targets.get_target(settings)

    return {
        "enabled": settings.get("backup_enabled") == "true",
        "destination": destination,
        "destination_display": target.describe(),
        "configured": backup_targets.is_configured(settings),
        # SFTP form state
        "sftp_host": settings.get("backup_sftp_host", ""),
        "sftp_port": settings.get("backup_sftp_port", "22"),
        "sftp_path": settings.get("backup_sftp_path", ""),
        "sftp_username": settings.get("backup_sftp_username", ""),
        "sftp_display": target.describe() if destination == "sftp" else "",
        "auth_method": settings.get("backup_sftp_auth_method", "password"),
        # S3 form state. The secret is never returned — the form shows a masked
        # placeholder and an empty submit keeps the stored value.
        "s3_bucket": settings.get("backup_s3_bucket", ""),
        "s3_prefix": settings.get("backup_s3_prefix", ""),
        "s3_region": settings.get("backup_s3_region", ""),
        "s3_endpoint_url": settings.get("backup_s3_endpoint_url", ""),
        "s3_access_key_id": settings.get("backup_s3_access_key_id", ""),
        "s3_secret_set": bool(settings.get("backup_s3_secret_access_key")),
        "retention_count": settings.get("backup_retention_count", "30"),
        "last_run": settings.get("backup_last_run", ""),
        "last_status": settings.get("backup_last_status", ""),
    }


async def _finalize_configuration(destination: str, retention_count: int) -> Tuple[bool, str]:
    """Persist shared settings, then verify the destination before enabling.

    Backup stays disabled if the connection test fails, so the UI can never show
    an enabled state for a destination we have not proven we can write to.
    """
    db.set_setting("backup_destination", destination)
    db.set_setting("backup_retention_count", str(retention_count))

    success, msg = await test_backup_connection()
    if not success:
        db.set_setting("backup_enabled", "false")
        return False, f"Configuration saved but connection test failed: {msg}"

    db.set_setting("backup_enabled", "true")
    label = "S3" if destination == "s3" else "SFTP"
    logger.info(f"Backup configured ({destination}): {backup_targets.get_target().describe()}")
    return True, f"{label} backup configured and connection verified"


async def configure_sftp_backup(
    host: str,
    port: int,
    path: str,
    username: str,
    auth_method: str,
    password: Optional[str] = None,
    ssh_key: Optional[str] = None,
    retention_count: int = 30,
) -> Tuple[bool, str]:
    """Save SFTP backup configuration and test the connection."""
    if not host or not username:
        return False, "Host and username are required"

    if auth_method == "password" and not password:
        # Allow keeping existing password on reconfigure
        existing = db.get_setting("backup_sftp_password")
        if not existing:
            return False, "Password is required for password authentication"
        password = None  # signal to skip overwriting
    if auth_method == "key" and not ssh_key and not SSH_KEY_PATH.exists():
        return False, "SSH key is required for key authentication"

    # Store SSH key if provided
    if auth_method == "key" and ssh_key:
        SSH_KEY_PATH.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        SSH_KEY_PATH.write_text(ssh_key.strip() + "\n")
        SSH_KEY_PATH.chmod(0o600)

    db.set_setting("backup_sftp_host", host)
    db.set_setting("backup_sftp_port", str(port))
    db.set_setting("backup_sftp_path", path)
    db.set_setting("backup_sftp_username", username)
    db.set_setting("backup_sftp_auth_method", auth_method)
    if auth_method == "password" and password:
        db.set_setting("backup_sftp_password", encrypt_password(password))

    return await _finalize_configuration("sftp", retention_count)


async def configure_s3_backup(
    bucket: str,
    prefix: str = "",
    region: str = "",
    endpoint_url: str = "",
    access_key_id: str = "",
    secret_access_key: Optional[str] = None,
    retention_count: int = 30,
) -> Tuple[bool, str]:
    """Save S3 backup configuration and test the connection.

    Leaving both the access key and the secret blank is valid and means "use
    ambient credentials" (IAM instance role / AWS_* env vars). A blank secret
    with a key already stored keeps the stored secret, so changing the bucket
    doesn't force the operator to re-enter it.
    """
    if not bucket:
        return False, "Bucket name is required"

    if access_key_id and not secret_access_key:
        existing = db.get_setting("backup_s3_secret_access_key")
        if not existing:
            return False, "Secret access key is required when an access key ID is set"
        secret_access_key = None  # signal to skip overwriting

    db.set_setting("backup_s3_bucket", bucket)
    db.set_setting("backup_s3_prefix", (prefix or "").strip("/"))
    db.set_setting("backup_s3_region", region)
    db.set_setting("backup_s3_endpoint_url", (endpoint_url or "").rstrip("/"))
    db.set_setting("backup_s3_access_key_id", access_key_id)
    if secret_access_key:
        db.set_setting("backup_s3_secret_access_key", encrypt_password(secret_access_key))
    elif not access_key_id:
        # Switching to ambient credentials — drop any stored secret so it can't
        # silently outlive the access key it belonged to.
        db.set_setting("backup_s3_secret_access_key", "")

    return await _finalize_configuration("s3", retention_count)


async def test_backup_connection() -> Tuple[bool, str]:
    """Test connectivity to the configured destination without uploading."""
    try:
        return await backup_targets.get_target().test()
    except Exception as e:
        return False, f"Connection failed: {e}"


async def list_backups() -> list[dict]:
    """List available backups at the configured destination."""
    return await backup_targets.get_target().list_archives()


async def restore_backup(archive_name: str) -> Tuple[bool, str]:
    """Download a backup from the configured destination and restore the DB.

    WARNING: This replaces the local sixtyops.db file. The application should
    ideally be restarted after this operation.
    """
    if not is_archive_name(archive_name):
        return False, "Invalid backup archive name"

    async with _backup_lock:
        try:
            STAGING_DIR.mkdir(parents=True, exist_ok=True)
            local_archive = STAGING_DIR / archive_name

            data = await backup_targets.get_target().download(archive_name)
            local_archive.write_bytes(data)

            return _restore_from_archive(local_archive)

        except Exception as e:
            logger.exception(f"Restore failed: {e}")
            return False, f"Restore failed: {e}"


def _safe_extract(tar: tarfile.TarFile, member: tarfile.TarInfo, dest: Path):
    """Extract one member, using tarfile's 'data' filter where the runtime
    supports it (Python 3.12+, and the 3.9.17+/3.10.12+/3.11.4+ backports).

    Older runtimes don't accept the keyword and raise TypeError; fall back
    without it. The production image runs 3.12 (filter applies); the CI test
    lane may run an older 3.11 (fallback). Either way the caller has already
    pinned member.name to an exact expected value, so there is no
    attacker-controlled path to traverse on the fallback path.
    """
    try:
        tar.extract(member, path=dest, filter="data")
    except TypeError:
        tar.extract(member, path=dest)


def _restore_from_archive(local_archive: Path) -> Tuple[bool, str]:
    """Restore the database (and its encryption key) from a downloaded archive.

    Split out from restore_backup so it can be tested without a live
    destination. Replaces the live DB and — when present — the credential
    encryption key, keeping the two consistent so a restore onto a fresh host
    can decrypt stored credentials.
    """
    import shutil

    # Extract sixtyops.db from the archive. The remote destination is in
    # principle untrusted (it could be MITM'd or compromised), so guard against
    # tar traversal / symlink overwrite: validate each member is a regular file
    # at the expected path, and use the tarfile data filter (where supported)
    # so unsafe attributes (absolute paths, ".." segments, device files, etc.)
    # are rejected on extract. See _safe_extract.
    extracted_key = None
    with tarfile.open(local_archive, "r:gz") as tar:
        try:
            member = tar.getmember("sixtyops.db")
        except KeyError:
            return False, "sixtyops.db not found in backup archive"
        if not member.isfile() or member.name != "sixtyops.db":
            return False, "Unsafe archive: unexpected sixtyops.db member type"
        _safe_extract(tar, member, STAGING_DIR)

        # Restore the credential encryption key alongside the DB so the restored
        # Fernet ciphertexts (device/RADIUS passwords) decrypt. Older archives
        # predate this and won't have it — restore the DB anyway (creds only
        # decrypt if the local key already matches).
        try:
            key_member = tar.getmember(".encryption_key")
            if key_member.isfile() and key_member.name == ".encryption_key":
                _safe_extract(tar, key_member, STAGING_DIR)
                extracted_key = STAGING_DIR / ".encryption_key"
        except KeyError:
            logger.warning(
                "Backup %s has no encryption key (older format); restored "
                "credentials will only decrypt if the local key matches",
                local_archive.name,
            )

    extracted_db = STAGING_DIR / "sixtyops.db"
    if not extracted_db.exists():
        return False, "sixtyops.db not found in backup archive"

    # Replace live database file.
    # Note: SQLite might have open connections; in a real deployment this might
    # need the app to stop or use .backup to a new file then swap. Here we do a
    # simple file swap which is risky but common for simple restores.
    shutil.copy2(extracted_db, DB_FILE)

    # Swap in the restored key (with the DB it matches) and drop the cached
    # cipher so the running process decrypts with it immediately.
    if extracted_key and extracted_key.exists():
        key_dest = crypto.key_path()
        key_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(extracted_key, key_dest)
        key_dest.chmod(0o600)
        crypto.reset_cache()

    # Cleanup
    local_archive.unlink(missing_ok=True)
    extracted_db.unlink(missing_ok=True)
    if extracted_key:
        extracted_key.unlink(missing_ok=True)

    logger.info(f"Database restored from backup: {local_archive.name}")
    return True, "Database restored successfully. System restart recommended."


async def run_backup() -> Tuple[bool, str]:
    """Run a full backup and upload it to the configured destination."""
    settings = db.get_all_settings()
    if settings.get("backup_enabled") != "true":
        return False, "Backup not configured"
    if not backup_targets.is_configured(settings):
        missing = ("S3 bucket" if backup_targets.get_destination(settings) == "s3"
                   else "SFTP host")
        return False, f"{missing} not configured"

    if _backup_lock.locked():
        return False, "Backup already in progress"

    async with _backup_lock:
        return await _run_backup_locked()


async def _run_backup_locked() -> Tuple[bool, str]:
    """Build the tar.gz archive and upload it to the configured destination."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    archive_name = f"sixtyops-backup-{timestamp}.tar.gz"
    logger.info(f"Starting backup: {archive_name}")

    try:
        # Build archive in memory.
        #
        # We include the DB *and* its credential encryption key
        # (_add_encryption_key): the DB stores device/RADIUS passwords as Fernet
        # ciphertext, so a restore onto a fresh host without the key leaves every
        # credential unrecoverable — a silent data-loss trap where the operator
        # believed they were backed up. Restorability wins (uptime), but it means
        # the archive effectively contains decryptable credentials, so the backup
        # destination MUST be access-controlled — a private bucket, or an SFTP
        # location you control (documented in the UI).
        #
        # We still do NOT include the per-device config snapshots
        # (_add_device_configs): those add WPA passphrases, SNMP write-
        # communities, and 802.1X secrets that aren't needed to restore the
        # manager's own ability to reach devices. The operator-driven CSV backup
        # (build_csv_export) still ships configs under a PBKDF2-derived passphrase.
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            _add_database(tar)
            _add_encryption_key(tar)
            _add_settings(tar)
            _add_device_inventory(tar, timestamp)

        archive_bytes = buf.getvalue()

        target = backup_targets.get_target()
        await target.upload(archive_name, archive_bytes)
        await _enforce_retention(target)

        db.set_setting("backup_last_run", datetime.now().isoformat())
        db.set_setting("backup_last_status", "success")
        logger.info(f"Backup uploaded: {archive_name} ({len(archive_bytes)} bytes)")
        return True, f"Backup completed: {archive_name}"

    except Exception as e:
        error_msg = str(e)[:200]
        db.set_setting("backup_last_run", datetime.now().isoformat())
        db.set_setting("backup_last_status", f"failed: {error_msg}")
        logger.exception(f"Backup failed: {e}")
        return False, f"Backup failed: {error_msg}"


def _add_database(tar: tarfile.TarFile):
    """Add a consistent SQLite snapshot to the archive."""
    if not DB_FILE.exists():
        logger.warning("Database file not found, skipping database backup")
        return

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    staging_db = STAGING_DIR / "sixtyops.db"
    try:
        src = sqlite3.connect(str(DB_FILE))
        dst = sqlite3.connect(str(staging_db))
        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()
        tar.add(str(staging_db), arcname="sixtyops.db")
    finally:
        staging_db.unlink(missing_ok=True)


def _add_encryption_key(tar: tarfile.TarFile):
    """Add the credential encryption key so a restore can decrypt the DB.

    Without this, the Fernet-encrypted device/RADIUS passwords in the database
    can't be decrypted after restoring onto a host with a different key.
    """
    key_file = crypto.key_path()
    if not key_file.exists():
        logger.warning("Encryption key not found, skipping (restored creds may not decrypt)")
        return
    data = key_file.read_bytes()
    info = tarfile.TarInfo(name=".encryption_key")
    info.size = len(data)
    info.mode = 0o600
    tar.addfile(info, io.BytesIO(data))


def _add_settings(tar: tarfile.TarFile):
    """Add sanitized settings JSON (no passwords, secrets, tokens, or keys)."""
    settings = db.get_all_settings()
    safe = {k: v for k, v in settings.items()
            if not any(s in k.lower() for s in ["password", "secret", "token", "key"])}
    data = json.dumps(safe, indent=2, sort_keys=True).encode()
    info = tarfile.TarInfo(name="settings.json")
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _add_device_inventory(tar: tarfile.TarFile, timestamp: str):
    """Add device inventory text file (no credentials)."""
    aps = db.get_access_points(enabled_only=False)
    switches = db.get_switches(enabled_only=False)

    lines = [f"# SixtyOps Device Inventory - {timestamp}\n"]
    lines.append(f"\n## Access Points ({len(aps)} total)\n")
    for ap in aps:
        name = ap.get("system_name") or "unnamed"
        model = ap.get("model") or "unknown"
        ver = ap.get("firmware_version") or "unknown"
        lines.append(f"{ap['ip']}\t{name}\t{model}\t{ver}\n")

    lines.append(f"\n## Switches ({len(switches)} total)\n")
    for sw in switches:
        name = sw.get("system_name") or "unnamed"
        model = sw.get("model") or "unknown"
        ver = sw.get("firmware_version") or "unknown"
        lines.append(f"{sw['ip']}\t{name}\t{model}\t{ver}\n")

    data = "".join(lines).encode()
    info = tarfile.TarInfo(name="devices.txt")
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _add_device_configs(tar: tarfile.TarFile):
    """Add individual device config JSON files under configs/ directory."""
    all_configs = db.get_all_latest_configs()
    for ip, config in all_configs.items():
        config_json = config.get("config_json", "{}")
        if isinstance(config_json, str):
            try:
                parsed = json.loads(config_json)
                pretty = json.dumps(parsed, indent=2)
            except json.JSONDecodeError:
                pretty = config_json
        else:
            pretty = json.dumps(config_json, indent=2)

        safe_ip = ip.replace(".", "-")
        model = config.get("model") or "unknown"
        filename = f"configs/{safe_ip}_{model}.json"
        data = pretty.encode()
        info = tarfile.TarInfo(name=filename)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))


async def _enforce_retention(target: backup_targets.BackupTarget):
    """Delete oldest backups beyond the retention count.

    Archive names are timestamped, so the newest-first ordering from
    list_archives is also the retention order — everything past the window is
    the tail of that list.
    """
    retention = int(db.get_setting("backup_retention_count") or "30")
    try:
        archives = await target.list_archives()
        for old in archives[retention:]:
            await target.delete(old["name"])
            logger.info(f"Retention cleanup: removed {old['name']}")
    except Exception as e:
        logger.warning(f"Retention cleanup failed: {e}")
