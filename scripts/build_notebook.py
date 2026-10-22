"""Build one self-contained Colab notebook from the checked-in source files."""
from pathlib import Path
import base64
import hashlib
import io
import zipfile
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
buffer = io.BytesIO()
with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if path.is_file() and (relative.parts[0] in {"src", "configs", "tests"} or str(relative) in {"pyproject.toml", "requirements-colab.txt"}) and "__pycache__" not in relative.parts:
            info = zipfile.ZipInfo(relative.as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, path.read_bytes())
payload = buffer.getvalue()
encoded, digest = base64.b64encode(payload).decode(), hashlib.sha256(payload).hexdigest()
cells = []
md = lambda text: cells.append(nbf.v4.new_markdown_cell(text))
code = lambda text: cells.append(nbf.v4.new_code_cell(text))

md("""# Language-Guided Robot Target Selection

Fine-tune **SmolVLM-500M-Instruct** with **LoRA** to identify a referred object among provided candidate boxes in real cluttered scenes.

**Pipeline:** RGB image + numbered annotated boxes + instruction → frozen VLM with trainable LoRA attention updates → candidate ID → selected box.

This notebook downloads the author-provided RoboRefIt archive, creates audited splits, measures a pretrained baseline, trains an adapter, verifies loading in another process, evaluates held-out data, and exports reproducible artifacts. It does not demonstrate autonomous detection, grasping, or robot control.

Select **Runtime → Change runtime type → T4 GPU**, then run the cells in order. All model computation runs in Colab. The notebook includes its Python source bundle; no source-file uploads or repository setup are needed.
""")
code('''MODE = "smoke"  # "smoke", "experiment", or "inference"
RUN_TAG = "run-01"  # Change this to preserve a previous run.
ADAPTER_ZIP_URL = ""  # Optional public release URL in inference mode.
ADAPTER_ZIP_SHA256 = ""  # Required when downloading from a URL.
INTERACTIVE_DEMO = True
REPEAT_SEED = MODE == "experiment"  # A second seed checks training variability.
DOWNLOAD_EXPORTS = False  # Set True to trigger browser downloads at the end.
EMBED_EXPORTS = True  # Save portable download links inside this executed notebook.
assert MODE in {"smoke", "experiment", "inference"}
''')
md("## 1. Reconstruct the versioned source bundle\nThe SHA-256 check detects corruption. The files below are the same reusable modules shipped in the repository.")
code(f'''import base64, hashlib, io, json, os, pathlib, subprocess, sys, zipfile
SOURCE_SHA256 = "{digest}"
SOURCE_B64 = "{encoded}"
source = base64.b64decode(SOURCE_B64)
assert hashlib.sha256(source).hexdigest() == SOURCE_SHA256
PROJECT = pathlib.Path("/content/robot-target-selection")
PROJECT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(source)) as archive:
    assert all(not pathlib.PurePosixPath(n).is_absolute() and ".." not in pathlib.PurePosixPath(n).parts for n in archive.namelist())
    archive.extractall(PROJECT)
os.chdir(PROJECT)
os.environ.update(PYTHONPATH=str(PROJECT / "src"), TOKENIZERS_PARALLELISM="false", HF_HUB_DISABLE_PROGRESS_BARS="1")
import site
site.addsitedir(str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "src"))
print("Verified source bundle:", SOURCE_SHA256)
''')
cells[-1].metadata["cellView"] = "form"
md("## 2. Tested dependencies and GPU preflight\nThe optional torchao package is removed because the stock Colab image combined an incompatible torchao version with PEFT. This experiment does not use quantization. PyTorch is supplied by Colab; the tested version is recorded below.")
code('''subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", "."], check=True)
site.addsitedir(str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "src"))
import robot_target
print("Project package:", robot_target.__file__)
import shutil
if shutil.which("unrar") is None:
    subprocess.run(["apt-get", "update", "-qq"], check=True)
    subprocess.run(["apt-get", "install", "-y", "-qq", "unrar"], check=True)
import torch, importlib.metadata
assert torch.cuda.is_available(), "Select a GPU runtime before continuing."
print("Python", sys.version)
print("torch", torch.__version__, "GPU", torch.cuda.get_device_name(0))
for package in ["transformers", "peft", "accelerate", "Pillow", "gdown"]:
    print(package, importlib.metadata.version(package))
subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], check=True)
''')
md("## 3. Configuration and process logging\nSmoke mode checks the complete chain on a tiny subset. Experiment mode uses a larger fixed subset and validation-based checkpoint selection. Inference mode uses the small subset for fast exploration of an exported adapter. Each training, evaluation and reload command runs in its own Python process.")
code('''config = json.loads(pathlib.Path("configs/experiment.json" if MODE == "experiment" else "configs/smoke.json").read_text())
RUN = PROJECT / "artifacts" / RUN_TAG
RUN.mkdir(parents=True, exist_ok=True)
CONFIG = RUN / "config.json"
CONFIG.write_text(json.dumps(config, indent=2))
(RUN / "source_sha256.txt").write_text(SOURCE_SHA256)
MANIFESTS = RUN / "manifests"
DATA = pathlib.Path("/content")
def stage(arguments, name):
    log_path = RUN / f"{name}.log"
    command = [sys.executable, "-m", "robot_target.cli"] + list(map(str, arguments))
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        log.write(json.dumps({"command": command, "process_id": process.pid}) + chr(10))
        log.flush()
        for line in process.stdout:
            log.write(line); log.flush(); print(line, end="", flush=True)
        if process.wait():
            raise RuntimeError(f"Stage {name} failed; inspect {log_path}.")
common = ["--config", CONFIG, "--data", DATA, "--manifests", MANIFESTS]
print(json.dumps(config, indent=2))
''')
md("""## 4. Official data and audited splits

Download the author-linked archive (~6.1 GB). Only RGB images and JSON annotations are extracted. Preserve official testA/testB; hold out every `shelf` example from the official training split for validation. Candidate sets contain all unique boxes from an image, independent of its query. Sample one reference per image. Save every chosen ID, exclusion and coordinate correction.

The `scene` field denotes a support category, not a physical session ID. This category holdout is conservative for train/validation, but physical train/test independence is not established. Exact pixel duplicates are checked; near-duplicates are not guaranteed absent. Dataset images remain outside the source repository.
""")
code('''stage(["download", "--data", DATA], "download")
shutil.copy2(DATA / "provenance.json", RUN / "dataset_provenance.json")
stage(["prepare"] + common, "prepare")
audit = json.loads((MANIFESTS / "data_audit.json").read_text())
print(json.dumps(audit["protocol"], indent=2))
for split in ["train", "testA", "testB"]:
    print(split, {k: v for k, v in audit[split].items() if k not in ["excluded", "box_corrections"]})
''')
md(r"""## 5. Baseline and LoRA adaptation

For a frozen projection matrix $W$, LoRA learns $W' = W + (\alpha/r)BA$, where $r$ is small. Only the language attention LoRA parameters are updated; the vision encoder and original language weights remain frozen.

Cross-entropy supervises the answer and end-of-turn token. Prompt, image and padding tokens are excluded. Runtime assertions check the supervised text, frozen parameters and nonzero finite gradients. A response is one in-range integer, optionally followed by a full stop. A separate restricted-next-token diagnostic measures the ranking among candidate digit tokens.

Test results are not used for training decisions. Checkpoints are selected by validation accuracy, with the earliest checkpoint winning ties.
""")
code('''if MODE != "inference":
    stage(["evaluate"] + common + ["--output", RUN / "baseline", "--splits", "val"], "baseline_val")
    stage(["train"] + common + ["--output", RUN / "trained"], "train")
    ADAPTER = RUN / "trained/adapter"
else:
    import requests
    adapter_zip = RUN / "adapter.zip"
    if ADAPTER_ZIP_URL:
        assert len(ADAPTER_ZIP_SHA256) == 64, "Supply the published adapter SHA-256."
        with requests.get(ADAPTER_ZIP_URL, stream=True, timeout=60) as response:
            response.raise_for_status()
            with adapter_zip.open("wb") as stream:
                for chunk in response.iter_content(1024*1024): stream.write(chunk)
        assert hashlib.sha256(adapter_zip.read_bytes()).hexdigest() == ADAPTER_ZIP_SHA256
    else:
        from google.colab import files
        print("Upload the previously exported adapter ZIP (no source files needed).")
        uploaded = files.upload()
        assert len(uploaded) == 1
        adapter_zip.write_bytes(next(iter(uploaded.values())))
    with zipfile.ZipFile(adapter_zip) as archive:
        assert all(not pathlib.PurePosixPath(n).is_absolute() and ".." not in pathlib.PurePosixPath(n).parts for n in archive.namelist())
        archive.extractall(RUN / "imported")
    choices = list((RUN / "imported").rglob("adapter_config.json"))
    assert len(choices) == 1
    ADAPTER = choices[0].parent
print("Adapter:", ADAPTER)
''')
md("## 6. Independent reload verification\nThe adapter is loaded in a new process and must reproduce the exact saved validation outputs. No trained model object is reused.")
code('''if MODE != "inference":
    stage(["verify-reload"] + common + ["--output", RUN / "trained"], "verify_reload")
    print((RUN / "trained/reload_verification.json").read_text())
if MODE == "experiment" and REPEAT_SEED:
    repeat_config = dict(config, seed=43)
    REPEAT_CONFIG = RUN / "config-seed43.json"
    REPEAT_CONFIG.write_text(json.dumps(repeat_config, indent=2))
    repeat_common = ["--config", REPEAT_CONFIG, "--data", DATA, "--manifests", MANIFESTS]
    stage(["train"] + repeat_common + ["--output", RUN / "trained-seed43"], "train_seed43")
    stage(["verify-reload"] + repeat_common + ["--output", RUN / "trained-seed43"], "reload_seed43")
''')
md("## 7. Final evaluation and controlled robustness checks\nAll comparisons reuse identical manifests. Perturbations: change only the instruction wrapper; renumber the same boxes; reduce brightness to 80% and contrast to 90% before drawing labels. Wrapper changes are a limited paraphrase test, not broad language robustness. Probabilities among digit choices are not calibrated confidence.")
code('''if MODE != "inference":
    stage(["evaluate"] + common + ["--output", RUN / "baseline", "--splits", "testA", "testB"], "baseline_test")
stage(["evaluate"] + common + ["--output", RUN / "adapted", "--adapter", ADAPTER,
      "--splits", "val", "testA", "testB"], "adapted_original")
stage(["evaluate"] + common + ["--output", RUN / "adapted", "--adapter", ADAPTER,
      "--splits", "testA", "testB", "--variants", "paraphrase", "permutation", "visual"], "robustness")
if MODE == "experiment" and REPEAT_SEED:
    stage(["evaluate"] + repeat_common + ["--output", RUN / "adapted-seed43", "--adapter", RUN / "trained-seed43/adapter",
          "--splits", "testA", "testB"], "test_seed43")
''')
md("## 8. Measured results and example gallery\nMetrics are recomputed from prediction files and checked against stored summaries. Uncertainty uses image-level bootstrap intervals; it does not cover domain shift or training-seed variation. Gallery examples are selected deterministically from successes and failures.")
code('''from IPython.display import display, Markdown, Image as DisplayImage, HTML
from robot_target.report import build_report, gallery, seed_report
if MODE != "inference":
    build_report(RUN / "baseline", RUN / "adapted", RUN / "trained", RUN / "report")
    report_lines = (RUN / "report/RESULTS.md").read_text().splitlines()
    display(Markdown(chr(10).join(line for line in report_lines if not line.startswith("!["))))
    display(DisplayImage(filename=str(RUN / "report/accuracy.png")))
    display(DisplayImage(filename=str(RUN / "report/training.png")))
    if MODE == "experiment" and REPEAT_SEED:
        seed_report({42: RUN / "adapted", 43: RUN / "adapted-seed43"}, RUN / "report")
        display(Markdown((RUN / "report/SEEDS.md").read_text()))
gallery_rows = gallery(MANIFESTS / "testB.json", RUN / "adapted/testB/original/predictions.json", DATA, RUN / "private_gallery", config, limit=6,
                       baseline=RUN / "baseline/testB/original/predictions.json" if MODE != "inference" else None)
for row, path in zip(gallery_rows, sorted((RUN / "private_gallery").glob("example_*.png"))):
    print(row["id"], row["instruction"], "expected", row["answer"], "predicted", row["prediction"], "correct", row["correct"])
    display(DisplayImage(filename=str(path)))
''')
md("## 9. Interactive target selection\nChoose a scene and edit its instruction. The model selects among the supplied boxes. For a new instruction there is no automatic ground-truth label; inspect the selected box.")
code('''if INTERACTIVE_DEMO:
    from google.colab import output
    output.enable_custom_widget_manager()
    from robot_target.demo import launch
    demo = launch(config, MANIFESTS / "testB.json", DATA, ADAPTER)
    (RUN / "demo_verification.json").write_text(json.dumps(demo["state"], indent=2))
    print("Initial demo inference:", demo["state"])
''')
md("## 10. Export artifacts and adapter\nSave exports before the temporary runtime expires. The experiment ZIP contains metrics, manifests, logs, and reports; the separate adapter ZIP contains the LoRA weights and configuration. Its processor is loaded from the pinned base model. Raw dataset images and the private gallery are excluded. The notebook itself can be saved via File → Download → .ipynb.")
code('''from robot_target.exports import pack_results, pack_adapter, describe, embedded_links
export_dir = PROJECT / "exports"
export_dir.mkdir(exist_ok=True)
result_zip, adapter_zip = export_dir / f"{RUN_TAG}-results.zip", export_dir / f"{RUN_TAG}-adapter.zip"
pack_results(RUN, result_zip)
pack_adapter(ADAPTER, adapter_zip, config)
print(json.dumps(describe([result_zip, adapter_zip]), indent=2))
if EMBED_EXPORTS:
    embedded_links([result_zip, adapter_zip])
if DOWNLOAD_EXPORTS:
    from google.colab import files
    files.download(str(result_zip))
    files.download(str(adapter_zip))
''')
md("""## References and limits

- [RoboRefIt / VL-Grasp author repository](https://github.com/luyh20/VL-Grasp)
- [RoboRefIt project](https://luyh20.github.io/RoboRefIt.github.io/)
- [SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct)
- [PEFT LoRA](https://huggingface.co/docs/peft/conceptual_guides/lora)

The model revision is pinned in the configuration and the archive is checksum-verified. Base model: Apache-2.0. Original project code: MIT. No separate dataset-image license was found in the downloaded archive; consult the dataset authors for redistribution or other use terms. Gallery images are generated from the downloaded copy, not bundled into the code repository.

Scores apply to the sampled, filtered manifests. The task supplies candidate boxes and contains annotation noise. No improvement is assumed; retain negative results and inspect the complete predictions before making claims.
""")
notebook = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "colab": {"name": "Robot_Target_Selection.ipynb", "provenance": []}, "accelerator": "GPU"})
destination = ROOT / "notebooks/Robot_Target_Selection.ipynb"
destination.parent.mkdir(exist_ok=True)
nbf.validate(notebook)
nbf.write(notebook, destination)
print(destination, len(cells), "cells", "source", digest)
