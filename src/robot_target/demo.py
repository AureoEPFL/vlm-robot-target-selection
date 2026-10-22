"""Notebook-only demonstration using the same candidates and model as evaluation."""
from __future__ import annotations

import json
from pathlib import Path


def launch(config, manifests, data_root, adapter):
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    from PIL import Image, ImageDraw
    from .modeling import load_model, predict
    from .metrics import parse_answer
    from .render import make_input

    rows = json.loads(Path(manifests).read_text())
    model, processor = load_model(config, str(adapter))
    select = widgets.Dropdown(options=[(f'{r["id"]} · {r["scene"]}', i) for i, r in enumerate(rows)], description="Scene")
    instruction = widgets.Textarea(value=rows[0]["text"], description="Instruction", layout=widgets.Layout(width="95%"))
    button = widgets.Button(description="Select target", button_style="primary")
    output = widgets.Output()
    state = {}

    def change_scene(change):
        instruction.value = rows[change["new"]]["text"]

    def run(_):
        row = dict(rows[select.value], text=instruction.value)
        button.disabled = True
        try:
            image, prompt, _, order = make_input(row, data_root, config.get("eval_seed", 42), size=config["image_size"])
            text = predict(model, processor, image, prompt, config["max_new_tokens"])
            number = parse_answer(text, len(order))
            state.update(example_id=row["id"], instruction=instruction.value,
                         raw_output=text, prediction=number,
                         candidate_count=len(order))
            with output:
                clear_output(wait=True)
                display(image)
                print(f"Model output: {text!r}")
                if number is None:
                    print("No valid selection. The model output is left unchanged.")
                else:
                    with Image.open(Path(data_root) / row["image_path"]) as original:
                        selected = original.convert("RGB")
                    ImageDraw.Draw(selected).rectangle(row["boxes"][order[number-1]], outline="#00aa66", width=5)
                    display(selected)
                    print(f"Selected candidate {number}. This is a supplied box, not a predicted grasp.")
        finally:
            button.disabled = False

    select.observe(change_scene, names="value")
    button.on_click(run)
    display(widgets.VBox([select, instruction, button, output]))
    run(None)
    return {"model": model, "processor": processor,
            "widgets": (select, instruction, button, output), "state": state}
