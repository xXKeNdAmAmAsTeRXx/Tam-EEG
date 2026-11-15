"""Publish a saved model bundle to the Hugging Face Hub."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .bundle import METADATA_FILE, MODEL_CARD_FILE, MODEL_FILE, read_metadata
from .config import StoreConfig

log = logging.getLogger(__name__)

REQUIRED_FILES = (MODEL_FILE, METADATA_FILE, MODEL_CARD_FILE)


def _open_api(config: StoreConfig, api: Any | None) -> Any:
    if api is not None:
        return api
    from huggingface_hub import HfApi

    return HfApi(token=config.resolved_token())


def _looks_like_access_error(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status in {401, 403, 404}:
        return True
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "401",
            "403",
            "404",
            "forbidden",
            "unauthorized",
            "permission",
            "not found",
        )
    )


def _username(api: Any) -> str:
    whoami = api.whoami()
    if isinstance(whoami, dict):
        name = whoami.get("name")
    else:
        name = getattr(whoami, "name", None)
    if not name:
        raise RuntimeError(
            "could not determine the Hugging Face username for the fallback "
            "repository"
        )
    return str(name)


def _create_repo(api: Any, repo_id: str, config: StoreConfig) -> None:
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        exist_ok=True,
        private=config.private,
    )


def _create_tag(api: Any, repo_id: str, tag: str) -> str | None:
    create_tag = getattr(api, "create_tag", None)
    if create_tag is None:  # very old huggingface_hub
        return None
    try:
        create_tag(repo_id=repo_id, tag=tag, repo_type="model")
    except Exception as exc:  # a tag already exists or races with the upload
        log.warning("Could not create tag %s on %s: %s", tag, repo_id, exc)
        return None
    return tag


def _upload(
    api: Any,
    repo_id: str,
    config: StoreConfig,
    directory: Path,
    version: str,
) -> Any:
    _create_repo(api, repo_id, config)
    return api.upload_folder(
        folder_path=str(directory),
        repo_id=repo_id,
        repo_type="model",
        commit_message=config.resolved_commit_message(version),
    )


def publish(
    bundle_dir: str | Path,
    config: StoreConfig | None = None,
    *,
    api: Any | None = None,
    fallback_to_user: bool = True,
    tag: bool = True,
) -> dict[str, Any]:
    """Upload a saved bundle as a Hugging Face model repository.

    The version recorded in ``metadata.json`` is authoritative: it is used as
    the git tag and in the commit message. ``config.version`` only matters
    when :func:`~hotb_2026.model_store.save_model` writes the bundle.

    Parameters
    ----------
    bundle_dir:
        Directory produced by :func:`~hotb_2026.model_store.save_model`.
    config:
        Repository, token and visibility settings.
    api:
        Pre-built ``huggingface_hub.HfApi`` (used by tests). Created from the
        configured token when omitted.
    fallback_to_user:
        When the configured owner is not writable, publish under the
        authenticated user instead of failing.
    tag:
        Create a git tag named after the bundle version.

    Returns
    -------
    A JSON-friendly summary with ``repo_id``, ``repo_url``, ``version``,
    ``tag`` and the upload ``commit``.

    Raises
    ------
    FileNotFoundError
        When the bundle is incomplete.
    ValueError
        When no repository is configured or the metadata is unsupported.
    """
    config = config or StoreConfig()
    directory = Path(bundle_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"bundle directory not found: {directory}")
    missing = [name for name in REQUIRED_FILES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"bundle at {directory} is incomplete; missing: {', '.join(missing)}"
        )

    metadata = read_metadata(directory)
    version = str(metadata["model_version"])

    # Resolve the target before importing huggingface_hub so a misconfigured
    # call fails without touching the network stack.
    repo_id = config.resolved_repo_id()
    client = _open_api(config, api)

    try:
        commit = _upload(client, repo_id, config, directory, version)
    except Exception as exc:
        owner, _, name = repo_id.partition("/")
        if not (
            fallback_to_user
            and name
            and _looks_like_access_error(exc)
        ):
            raise
        username = _username(client)
        if username == owner:
            raise
        fallback = f"{username}/{name}"
        log.warning(
            "Cannot publish to %s (%s); falling back to %s",
            repo_id,
            exc,
            fallback,
        )
        repo_id = fallback
        commit = _upload(client, repo_id, config, directory, version)

    tag_name = _create_tag(client, repo_id, version) if tag else None

    return {
        "repo_id": repo_id,
        "repo_url": f"https://huggingface.co/{repo_id}",
        "version": version,
        "tag": tag_name,
        "commit": getattr(commit, "oid", None),
    }


__all__ = ["REQUIRED_FILES", "publish"]
