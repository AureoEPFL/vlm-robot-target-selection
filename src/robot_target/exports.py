"""Portable experiment exports, with optional links saved in notebook outputs."""
from __future__ import annotations

import base64
import hashlib
import html
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def pack_results(run: Path, destination: Path) -> Path:
    run, destination = Path(run), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
        for path in sorted(run.rglob("*")):
            relative = path.relative_to(run)
            if path.is_file() and not any(part in {"adapter", "private_gallery", "imported", "__pycache__"} for part in relative.parts) and path.suffix not in {".zip", ".safetensors", ".bin", ".pt", ".pyc"}:
                archive.write(path, relative)
    return destination


def pack_adapter(adapter: Path, destination: Path, config: dict | None = None) -> Path:
    adapter, destination = Path(adapter), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # The processor is reproducibly loaded from the pinned base revision, so
    # duplicating its tokenizer files is unnecessary for adapter inference.
    with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
        for name in ("adapter_model.safetensors", "adapter_config.json", "README.md"):
            path = adapter / name
            if name != "README.md" and not path.is_file():
                raise FileNotFoundError(path)
            if path.is_file():
                archive.write(path, "adapter/" + name)
        if config is not None:
            import json
            archive.writestr("experiment_config.json", json.dumps(config, indent=2))
    return destination


def describe(paths: list[Path]) -> list[dict]:
    return [{"filename": path.name, "bytes": path.stat().st_size,
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]


def embedded_links(paths: list[Path], maximum_total_bytes: int = 16_000_000) -> None:
    """Visible data links persist exports when the notebook's outputs are saved.

    Keep executed notebooks containing weights out of the source repository.
    This uses no new Drive permission, credential, or external upload service.
    """
    from IPython.display import HTML, display
    if sum(path.stat().st_size for path in paths) > maximum_total_bytes:
        raise ValueError("Exports are too large to embed. Use the normal download option.")
    blocks = ["<div><p>These archives are embedded in this saved notebook output. Keep this executed copy private; publish the source notebook with outputs cleared.</p>"]
    for path, info in zip(paths, describe(paths), strict=True):
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
        filename = html.escape(path.name, quote=True)
        blocks.append(f'<p><a download="{filename}" href="data:application/zip;base64,{payload}">Download {filename}</a><br>{info["bytes"]:,} bytes · SHA-256 <code>{info["sha256"]}</code></p>')
    blocks.append("</div>")
    display(HTML("".join(blocks)))
