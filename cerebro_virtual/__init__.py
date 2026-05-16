from pathlib import Path

_pkg_root = Path(__file__).resolve().parent
_repo_root = _pkg_root.parent
__path__ = [str(_repo_root)]
