import os
import sys
from pathlib import Path


def configure_mover(workspace: Path | None = None) -> None:
    """Expose the vendored MOVER stack and its Windows runtime dependencies."""
    workspace = (workspace or Path(__file__).resolve().parents[1]).resolve()

    paths = [
        workspace / "vendor" / "meeteval",
        workspace / "research" / "third_party" / "mover",
        workspace / "research" / "third_party" / "spy-der-0.4.1" / "src",
        workspace
        / "research"
        / "third_party"
        / "spy-der-0.4.1"
        / "build"
        / "mingw",
    ]
    for path in reversed(paths):
        if not path.exists():
            raise FileNotFoundError(path)
        sys.path.insert(0, str(path))

    if sys.platform == "win32":
        packages = (
            Path(os.environ.get("LOCALAPPDATA", ""))
            / "Microsoft"
            / "WinGet"
            / "Packages"
        )
        candidates = list(packages.glob("*WinLibs*/mingw64/bin"))
        if not candidates:
            raise FileNotFoundError("WinLibs mingw64/bin was not found")
        os.add_dll_directory(str(candidates[0]))

