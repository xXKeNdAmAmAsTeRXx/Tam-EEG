"""Configuration for saving and publishing model bundles."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

DEFAULT_VERSION = "0.1.0"
REPO_ID_ENV = "HOTB_MODEL_REPO_ID"
TOKEN_ENV = "HF_TOKEN"

_VERSION_RE = re.compile(r"^[0-9A-Za-z][0-9A-Za-z._-]*$")


def validate_version(version: str) -> str:
    """Return ``version`` when it is safe to use as a file name or git tag."""
    if not isinstance(version, str) or not _VERSION_RE.match(version):
        raise ValueError(
            "version must be a non-empty string of letters, digits, "
            f"'.', '_' or '-', got {version!r}"
        )
    return version


@dataclass(frozen=True)
class StoreConfig:
    """Settings shared by :func:`~hotb_2026.model_store.save_model` and
    :func:`~hotb_2026.model_store.publish`.

    Parameters
    ----------
    repo_id:
        Hugging Face model repository, ``"owner/name"``. When omitted, the
        ``HOTB_MODEL_REPO_ID`` environment variable is used.
    version:
        Model version written to the bundle metadata by
        :func:`~hotb_2026.model_store.save_model`.
        :func:`~hotb_2026.model_store.publish` reads it back from
        ``metadata.json`` when it creates the Hub tag.
    private:
        Create a private repository. The project publishes public models by
        default.
    token:
        Hugging Face token. When omitted, ``HF_TOKEN`` and then the token
        cached by ``hf auth login`` are used.
    commit_message:
        Commit message for the upload. Defaults to ``add model version <v>``.
    """

    repo_id: str | None = None
    version: str = DEFAULT_VERSION
    private: bool = False
    token: str | None = None
    commit_message: str | None = None

    def __post_init__(self) -> None:
        validate_version(self.version)
        if self.repo_id is not None:
            self._validate_repo_id(self.repo_id)
        if self.commit_message is not None and not self.commit_message.strip():
            raise ValueError("commit_message must not be blank")

    @staticmethod
    def _validate_repo_id(repo_id: str) -> None:
        owner, separator, name = repo_id.partition("/")
        if not owner or not separator or not name or "/" in name:
            raise ValueError(
                f"repo_id must look like 'owner/name', got {repo_id!r}"
            )

    def resolved_repo_id(self) -> str:
        """Return the configured repo id or read it from the environment."""
        repo_id = self.repo_id or os.environ.get(REPO_ID_ENV)
        if not repo_id:
            raise ValueError(
                "no Hugging Face repo id configured; pass repo_id or set "
                f"{REPO_ID_ENV}"
            )
        self._validate_repo_id(repo_id)
        return repo_id

    def resolved_token(self) -> str | None:
        """Return an explicit or environment token, falling back to the cache."""
        return self.token or os.environ.get(TOKEN_ENV) or None

    def resolved_commit_message(self, version: str | None = None) -> str:
        """Commit message for an upload, defaulting to the configured version."""
        target = version or self.version
        return self.commit_message or f"add model version {target}"


__all__ = [
    "DEFAULT_VERSION",
    "REPO_ID_ENV",
    "TOKEN_ENV",
    "StoreConfig",
    "validate_version",
]
