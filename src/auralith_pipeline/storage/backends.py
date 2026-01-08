"""Storage module for uploading and downloading shards."""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class UploadResult:
    """Result of an upload operation."""

    success: bool
    url: str | None = None
    error: str | None = None
    files_uploaded: int = 0
    bytes_uploaded: int = 0


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def upload(self, local_path: str, remote_path: str) -> UploadResult:
        """Upload a file or directory."""
        pass

    @abstractmethod
    def download(self, remote_path: str, local_path: str) -> bool:
        """Download a file or directory."""
        pass

    @abstractmethod
    def list_files(self, remote_path: str) -> list[str]:
        """List files at remote path."""
        pass


class HuggingFaceStorage(StorageBackend):
    """HuggingFace Hub storage backend."""

    def __init__(self, repo_id: str, token: str | None = None, repo_type: str = "dataset"):
        self.repo_id = repo_id
        self.token = token or os.environ.get("HF_TOKEN")
        self.repo_type = repo_type

    def upload(self, local_path: str, remote_path: str = "") -> UploadResult:
        """Upload files to HuggingFace Hub."""
        try:
            from huggingface_hub import HfApi, create_repo

            api = HfApi(token=self.token)

            # Create repo if it doesnt exist
            try:
                create_repo(
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    exist_ok=True,
                    token=self.token,
                )
            except Exception as e:
                logger.warning(f"Repo creation: {e}")

            local_path = Path(local_path)
            files_uploaded = 0
            bytes_uploaded = 0

            if local_path.is_file():
                api.upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=remote_path or local_path.name,
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    token=self.token,
                )
                files_uploaded = 1
                bytes_uploaded = local_path.stat().st_size
            else:
                # Upload folder
                api.upload_folder(
                    folder_path=str(local_path),
                    path_in_repo=remote_path,
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    token=self.token,
                )
                for f in local_path.rglob("*"):
                    if f.is_file():
                        files_uploaded += 1
                        bytes_uploaded += f.stat().st_size

            url = f"https://huggingface.co/datasets/{self.repo_id}"
            logger.info(f"Uploaded to {url}")

            return UploadResult(
                success=True,
                url=url,
                files_uploaded=files_uploaded,
                bytes_uploaded=bytes_uploaded,
            )

        except Exception as e:
            logger.error(f"HuggingFace upload failed: {e}")
            return UploadResult(success=False, error=str(e))

    def download(self, remote_path: str, local_path: str) -> bool:
        """Download files from HuggingFace Hub."""
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                local_dir=local_path,
                token=self.token,
            )
            logger.info(f"Downloaded to {local_path}")
            return True

        except Exception as e:
            logger.error(f"HuggingFace download failed: {e}")
            return False

    def list_files(self, remote_path: str = "") -> list[str]:
        """List files in HuggingFace repo."""
        try:
            from huggingface_hub import list_repo_files

            files = list_repo_files(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                token=self.token,
            )
            return list(files)

        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            return []


class S3Storage(StorageBackend):
    """AWS S3 storage backend."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        region: str | None = None,
    ):
        self.bucket = bucket
        self.prefix = prefix
        self.region = region
        self._client = None

    def _get_client(self):
        if self._client is None:
            import boto3

            self._client = boto3.client("s3", region_name=self.region)
        return self._client

    def upload(self, local_path: str, remote_path: str = "") -> UploadResult:
        """Upload to S3."""
        try:
            s3 = self._get_client()
            local_path = Path(local_path)

            files_uploaded = 0
            bytes_uploaded = 0

            if local_path.is_file():
                key = (
                    f"{self.prefix}/{remote_path}"
                    if remote_path
                    else f"{self.prefix}/{local_path.name}"
                )
                s3.upload_file(str(local_path), self.bucket, key.lstrip("/"))
                files_uploaded = 1
                bytes_uploaded = local_path.stat().st_size
            else:
                for f in local_path.rglob("*"):
                    if f.is_file():
                        rel_path = f.relative_to(local_path)
                        key = f"{self.prefix}/{rel_path}".lstrip("/")
                        s3.upload_file(str(f), self.bucket, key)
                        files_uploaded += 1
                        bytes_uploaded += f.stat().st_size

            url = f"s3://{self.bucket}/{self.prefix}"
            logger.info(f"Uploaded to {url}")

            return UploadResult(
                success=True,
                url=url,
                files_uploaded=files_uploaded,
                bytes_uploaded=bytes_uploaded,
            )

        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
            return UploadResult(success=False, error=str(e))

    def download(self, remote_path: str, local_path: str) -> bool:
        """Download from S3."""
        try:
            import boto3

            s3 = boto3.resource("s3")
            bucket = s3.Bucket(self.bucket)

            local_path = Path(local_path)
            local_path.mkdir(parents=True, exist_ok=True)

            prefix = f"{self.prefix}/{remote_path}".lstrip("/")
            for obj in bucket.objects.filter(Prefix=prefix):
                target = local_path / obj.key.replace(prefix, "").lstrip("/")
                target.parent.mkdir(parents=True, exist_ok=True)
                bucket.download_file(obj.key, str(target))

            logger.info(f"Downloaded to {local_path}")
            return True

        except Exception as e:
            logger.error(f"S3 download failed: {e}")
            return False

    def list_files(self, remote_path: str = "") -> list[str]:
        """List files in S3."""
        try:
            s3 = self._get_client()
            prefix = f"{self.prefix}/{remote_path}".lstrip("/")

            response = s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
            return [obj["Key"] for obj in response.get("Contents", [])]

        except Exception as e:
            logger.error(f"Failed to list S3 files: {e}")
            return []


class GCSStorage(StorageBackend):
    """Google Cloud Storage backend."""

    def __init__(self, bucket: str, prefix: str = ""):
        self.bucket_name = bucket
        self.prefix = prefix
        self._client = None

    def _get_bucket(self):
        if self._client is None:
            from google.cloud import storage

            client = storage.Client()
            self._client = client.bucket(self.bucket_name)
        return self._client

    def upload(self, local_path: str, remote_path: str = "") -> UploadResult:
        """Upload to GCS."""
        try:
            bucket = self._get_bucket()
            local_path = Path(local_path)

            files_uploaded = 0
            bytes_uploaded = 0

            if local_path.is_file():
                blob_name = (
                    f"{self.prefix}/{remote_path}"
                    if remote_path
                    else f"{self.prefix}/{local_path.name}"
                )
                blob = bucket.blob(blob_name.lstrip("/"))
                blob.upload_from_filename(str(local_path))
                files_uploaded = 1
                bytes_uploaded = local_path.stat().st_size
            else:
                for f in local_path.rglob("*"):
                    if f.is_file():
                        rel_path = f.relative_to(local_path)
                        blob_name = f"{self.prefix}/{rel_path}".lstrip("/")
                        blob = bucket.blob(blob_name)
                        blob.upload_from_filename(str(f))
                        files_uploaded += 1
                        bytes_uploaded += f.stat().st_size

            url = f"gs://{self.bucket_name}/{self.prefix}"
            logger.info(f"Uploaded to {url}")

            return UploadResult(
                success=True,
                url=url,
                files_uploaded=files_uploaded,
                bytes_uploaded=bytes_uploaded,
            )

        except Exception as e:
            logger.error(f"GCS upload failed: {e}")
            return UploadResult(success=False, error=str(e))

    def download(self, remote_path: str, local_path: str) -> bool:
        """Download from GCS."""
        try:
            bucket = self._get_bucket()
            local_path = Path(local_path)
            local_path.mkdir(parents=True, exist_ok=True)

            prefix = f"{self.prefix}/{remote_path}".lstrip("/")
            for blob in bucket.list_blobs(prefix=prefix):
                target = local_path / blob.name.replace(prefix, "").lstrip("/")
                target.parent.mkdir(parents=True, exist_ok=True)
                blob.download_to_filename(str(target))

            logger.info(f"Downloaded to {local_path}")
            return True

        except Exception as e:
            logger.error(f"GCS download failed: {e}")
            return False

    def list_files(self, remote_path: str = "") -> list[str]:
        """List files in GCS."""
        try:
            bucket = self._get_bucket()
            prefix = f"{self.prefix}/{remote_path}".lstrip("/")
            return [blob.name for blob in bucket.list_blobs(prefix=prefix)]
        except Exception as e:
            logger.error(f"Failed to list GCS files: {e}")
            return []


def create_storage_backend(
    backend_type: str,
    **kwargs,
) -> StorageBackend:
    """Factory function to create storage backends."""
    backends = {
        "huggingface": HuggingFaceStorage,
        "hf": HuggingFaceStorage,
        "s3": S3Storage,
        "gcs": GCSStorage,
    }

    if backend_type not in backends:
        raise ValueError(f"Unknown backend: {backend_type}. Available: {list(backends.keys())}")

    return backends[backend_type](**kwargs)
