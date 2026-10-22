"""Check notebook syntax, schema, bundle integrity and source synchronization."""
import ast
import base64
import hashlib
import io
from pathlib import Path
import zipfile
import nbformat

root = Path(__file__).resolve().parents[1]
notebook = nbformat.read(root / "notebooks/Robot_Target_Selection.ipynb", as_version=4)
nbformat.validate(notebook)
values = {}
for cell in notebook.cells:
    assert not any(ord(char) < 32 and char not in "\n\r\t" for char in cell.source), "Unexpected control character"
    if cell.cell_type != "code":
        continue
    tree = ast.parse(cell.source)
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
            name = statement.targets[0].id
            if name in {"SOURCE_SHA256", "SOURCE_B64"}:
                values[name] = ast.literal_eval(statement.value)
payload = base64.b64decode(values["SOURCE_B64"])
assert hashlib.sha256(payload).hexdigest() == values["SOURCE_SHA256"]
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    expected = {path.relative_to(root).as_posix() for path in root.rglob("*")
                if path.is_file() and (path.relative_to(root).parts[0] in {"src", "configs", "tests"}
                or path.name in {"pyproject.toml", "requirements-colab.txt"})
                and "__pycache__" not in path.parts}
    assert set(archive.namelist()) == expected, "Bundled file inventory differs; rebuild the notebook."
    for member in archive.namelist():
        assert archive.read(member) == (root / member).read_bytes(), f"Stale bundled source: {member}; rebuild the notebook."
print("Notebook schema, Python syntax, checksum and bundled sources verified.")
