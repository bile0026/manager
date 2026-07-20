"""Backup destination behavior: S3 keying, listing hygiene, and retention.

The archive format and restore path are covered in test_remote_backup.py. These
tests pin the parts that differ per destination — object keys, which remote
entries we are willing to touch, and the fact that retention deletes the oldest
archives rather than the newest.

boto3 is never exercised for real: S3Target._client is replaced with a fake, so
these run without credentials, network, or the dependency installed.
"""

from datetime import datetime
from unittest.mock import patch

import pytest

from updater import backup_targets, remote_backup
from updater.backup_targets import S3Target, SftpTarget, is_archive_name


# ---------------------------------------------------------------------------
# Fake S3
# ---------------------------------------------------------------------------

class FakeS3Client:
    """Minimal in-memory stand-in for the boto3 S3 client surface we use."""

    def __init__(self, objects=None):
        self.objects = dict(objects or {})
        self.deleted = []
        self.put_keys = []

    def put_object(self, Bucket, Key, Body):
        self.objects[Key] = Body
        self.put_keys.append(Key)

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise KeyError(f"missing {Key}")

        class _Body:
            def __init__(self, data):
                self._data = data

            def read(self):
                return self._data

        return {"Body": _Body(self.objects[Key])}

    def delete_object(self, Bucket, Key):
        self.deleted.append(Key)
        self.objects.pop(Key, None)

    def head_bucket(self, Bucket):
        return {}

    def get_paginator(self, name):
        client = self

        class _Paginator:
            def paginate(self, Bucket, Prefix=""):
                contents = [
                    {
                        "Key": k,
                        "Size": len(v),
                        "LastModified": datetime(2026, 1, 1),
                    }
                    for k, v in sorted(client.objects.items())
                    if k.startswith(Prefix)
                ]
                return [{"Contents": contents}]

        return _Paginator()


def _target(bucket="bk", prefix="sixtyops", objects=None, **kw):
    t = S3Target(bucket=bucket, prefix=prefix, **kw)
    fake = FakeS3Client(objects)
    t._client = lambda: fake
    return t, fake


# ---------------------------------------------------------------------------
# Archive name validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "sixtyops-backup-2026-01-01_120000.tar.gz",
    "sixtyops-backup-1.4.1-dev5.tar.gz",
])
def test_valid_archive_names_accepted(name):
    assert is_archive_name(name) is True


@pytest.mark.parametrize("name", [
    "../../etc/passwd",
    "sixtyops-backup-../../evil.tar.gz",
    "not-ours.tar.gz",
    "sixtyops-backup-x.zip",
    "sixtyops-backup-.tar.gz.exe",
])
def test_hostile_or_foreign_names_rejected(name):
    """Retention deletes what listing returns, so listing must never surface an
    entry we would not have written ourselves."""
    assert is_archive_name(name) is False


# ---------------------------------------------------------------------------
# S3 keying
# ---------------------------------------------------------------------------

def test_prefix_is_normalized_and_applied():
    t = S3Target(bucket="bk", prefix="/nested/path/")
    assert t.prefix == "nested/path"
    assert t._key("a.tar.gz") == "nested/path/a.tar.gz"


def test_blank_prefix_writes_to_bucket_root():
    t = S3Target(bucket="bk", prefix="")
    assert t._key("a.tar.gz") == "a.tar.gz"


def test_describe_includes_endpoint_only_for_non_aws():
    aws = S3Target(bucket="bk", prefix="p")
    assert aws.describe() == "s3://bk/p"
    minio = S3Target(bucket="bk", prefix="p", endpoint_url="https://minio.local")
    assert "https://minio.local" in minio.describe()


def test_ambient_credentials_detected():
    """Blank key + secret means botocore's default chain (IAM role, env vars)."""
    assert S3Target(bucket="bk").uses_ambient_credentials is True
    assert S3Target(bucket="bk", access_key_id="AKIA").uses_ambient_credentials is True
    partial = S3Target(bucket="bk", access_key_id="AKIA", secret_access_key="s")
    assert partial.uses_ambient_credentials is False


# ---------------------------------------------------------------------------
# Upload / download / list
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upload_download_round_trip():
    t, fake = _target()
    await t.upload("sixtyops-backup-a.tar.gz", b"payload")
    assert fake.put_keys == ["sixtyops/sixtyops-backup-a.tar.gz"]
    assert await t.download("sixtyops-backup-a.tar.gz") == b"payload"


@pytest.mark.asyncio
async def test_list_returns_newest_first_and_ignores_foreign_objects():
    t, _ = _target(objects={
        "sixtyops/sixtyops-backup-2026-01-01_010000.tar.gz": b"a",
        "sixtyops/sixtyops-backup-2026-03-01_010000.tar.gz": b"b",
        "sixtyops/sixtyops-backup-2026-02-01_010000.tar.gz": b"c",
        "sixtyops/unrelated-company-data.tar.gz": b"nope",
        "sixtyops/notes.txt": b"nope",
    })
    names = [a["name"] for a in await t.list_archives()]
    assert names == [
        "sixtyops-backup-2026-03-01_010000.tar.gz",
        "sixtyops-backup-2026-02-01_010000.tar.gz",
        "sixtyops-backup-2026-01-01_010000.tar.gz",
    ]


@pytest.mark.asyncio
async def test_list_failure_is_not_fatal():
    """A listing error must not break the status panel."""
    t = S3Target(bucket="bk")

    def _boom():
        raise RuntimeError("network down")

    t._client = _boom
    assert await t.list_archives() == []


# ---------------------------------------------------------------------------
# Connection test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_test_probes_write_access_and_cleans_up():
    """head_bucket alone would pass on a read-only bucket, so we round-trip a
    marker object — and must not leave it behind."""
    t, fake = _target()
    ok, msg = await t.test()
    assert ok is True
    assert "sixtyops/.sixtyops-write-test" in fake.put_keys
    assert "sixtyops/.sixtyops-write-test" in fake.deleted
    assert fake.objects == {}


@pytest.mark.asyncio
async def test_test_requires_bucket():
    ok, msg = await S3Target(bucket="").test()
    assert ok is False
    assert "Bucket name is required" in msg


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retention_deletes_oldest_not_newest(mock_db):
    """The whole point of retention is keeping recent backups. Deleting from the
    wrong end of a newest-first list would silently destroy them."""
    from updater import database as db
    db.set_setting("backup_retention_count", "2")

    t, fake = _target(objects={
        "sixtyops/sixtyops-backup-2026-01-01_010000.tar.gz": b"oldest",
        "sixtyops/sixtyops-backup-2026-02-01_010000.tar.gz": b"middle",
        "sixtyops/sixtyops-backup-2026-03-01_010000.tar.gz": b"newest",
    })
    await remote_backup._enforce_retention(t)

    assert fake.deleted == ["sixtyops/sixtyops-backup-2026-01-01_010000.tar.gz"]
    remaining = sorted(fake.objects.values())
    assert remaining == [b"middle", b"newest"]


@pytest.mark.asyncio
async def test_retention_keeps_everything_under_the_limit(mock_db):
    from updater import database as db
    db.set_setting("backup_retention_count", "30")

    t, fake = _target(objects={
        "sixtyops/sixtyops-backup-2026-01-01_010000.tar.gz": b"a",
    })
    await remote_backup._enforce_retention(t)
    assert fake.deleted == []


# ---------------------------------------------------------------------------
# Destination selection
# ---------------------------------------------------------------------------

def test_destination_defaults_to_sftp_for_preexisting_installs(mock_db):
    """Installs predating S3 support have no backup_destination row and must
    keep using SFTP rather than silently switching."""
    from updater import database as db
    db.set_setting("backup_destination", "")
    assert backup_targets.get_destination() == "sftp"
    assert isinstance(backup_targets.get_target(), SftpTarget)


def test_unknown_destination_falls_back_to_sftp(mock_db):
    from updater import database as db
    db.set_setting("backup_destination", "dropbox")
    assert backup_targets.get_destination() == "sftp"


def test_s3_destination_selects_s3_target(mock_db):
    from updater import database as db
    db.set_setting("backup_destination", "s3")
    db.set_setting("backup_s3_bucket", "my-bucket")
    assert isinstance(backup_targets.get_target(), S3Target)


def test_is_configured_checks_the_selected_destination(mock_db):
    """The scheduler skips on this, so an S3 install must not be judged by
    whether an old SFTP host happens to still be filled in."""
    from updater import database as db
    db.set_setting("backup_destination", "s3")
    db.set_setting("backup_sftp_host", "leftover.example.com")
    db.set_setting("backup_s3_bucket", "")
    assert backup_targets.is_configured() is False

    db.set_setting("backup_s3_bucket", "my-bucket")
    assert backup_targets.is_configured() is True


@pytest.mark.asyncio
async def test_run_backup_reports_the_right_missing_field(mock_db):
    from updater import database as db
    db.set_setting("backup_enabled", "true")
    db.set_setting("backup_destination", "s3")
    db.set_setting("backup_s3_bucket", "")

    ok, msg = await remote_backup.run_backup()
    assert ok is False
    assert "S3 bucket" in msg


# ---------------------------------------------------------------------------
# Secret handling
# ---------------------------------------------------------------------------

def test_s3_secret_is_decrypted_from_settings():
    from updater.crypto import encrypt_password
    settings = {
        "backup_s3_bucket": "bk",
        "backup_s3_secret_access_key": encrypt_password("super-secret"),
    }
    t = S3Target.from_settings(settings)
    assert t.secret_access_key == "super-secret"


def test_s3_secret_never_appears_in_backup_status(mock_db):
    from updater import database as db
    db.set_setting("backup_destination", "s3")
    db.set_setting("backup_s3_bucket", "bk")
    db.set_setting("backup_s3_secret_access_key", "ciphertext-here")

    status = remote_backup.get_backup_status()
    assert status["s3_secret_set"] is True
    assert "ciphertext-here" not in repr(status)
    assert not any("secret_access_key" in k and k != "s3_secret_set"
                   for k in status)
