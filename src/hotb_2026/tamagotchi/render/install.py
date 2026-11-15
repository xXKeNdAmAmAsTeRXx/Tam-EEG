"""Idempotent shell prompt hook: shows the pet before every prompt."""

from __future__ import annotations

from pathlib import Path

from hotb_2026.tamagotchi.core.constants import HOOK_BEGIN, HOOK_END, state_dir

_HOOK_BODY = f"""{HOOK_BEGIN}
# Print the neuro-tamagotchi status before each prompt (if installed).
if [ -f "{state_dir() / "status.txt"}" ]; then
    cat "{state_dir() / "status.txt"}"
fi
{HOOK_END}
"""


def _update_file(path: Path) -> bool:
    """Add/refresh the marked block in ``path``.  True when the file changed."""
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    if HOOK_BEGIN in text:
        start = text.index(HOOK_BEGIN)
        end = text.index(HOOK_END) + len(HOOK_END)
        if text[start : end + 1] == _HOOK_BODY:
            return False
        text = text[:start] + _HOOK_BODY.rstrip("\n") + text[end:]
    else:
        text = text.rstrip("\n") + "\n\n" + _HOOK_BODY.rstrip("\n") + "\n"
    path.write_text(text, encoding="utf-8")
    return True


def install_hook() -> list[Path]:
    """Install the prompt hook into existing rc files.

    Idempotent: the block is replaced in place thanks to the
    ``# >>> hotb tamagotchi >>>`` markers.  Returns the files touched.
    """
    home = Path.home()
    touched = []
    for name in (".bashrc", ".zshrc"):
        path = home / name
        if path.exists() and _update_file(path):
            touched.append(path)
    return touched


def remove_hook() -> list[Path]:
    """Remove the marked block from all rc files.  Returns files touched."""
    home = Path.home()
    touched = []
    for name in (".bashrc", ".zshrc"):
        path = home / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if HOOK_BEGIN not in text:
            continue
        start = text.index(HOOK_BEGIN)
        end = text.index(HOOK_END) + len(HOOK_END)
        text = text[:start].rstrip("\n") + "\n" + text[end:].lstrip("\n")
        path.write_text(text, encoding="utf-8")
        touched.append(path)
    return touched
