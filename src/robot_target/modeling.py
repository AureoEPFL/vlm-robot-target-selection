"""Pinned VLM loading, language-only LoRA, and audited answer-only supervision."""
from __future__ import annotations

import random
import time
import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(config: dict, adapter: str | None = None):
    if not torch.cuda.is_available():
        raise RuntimeError("A remote CUDA GPU is required. Choose a Colab T4 runtime.")
    processor = AutoProcessor.from_pretrained(config["model_id"], revision=config["model_revision"])
    processor.image_processor.do_image_splitting = False
    processor.tokenizer.padding_side = "right"
    model = AutoModelForImageTextToText.from_pretrained(
        config["model_id"], revision=config["model_revision"],
        dtype=torch.float16, attn_implementation="sdpa",
    ).to("cuda")
    model.requires_grad_(False)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    return model, processor


def add_lora(model, config: dict):
    targets = [name for name, module in model.named_modules()
               if ".text_model." in name and name.rsplit(".", 1)[-1] in
               {"q_proj", "k_proj", "v_proj", "o_proj"} and isinstance(module, torch.nn.Linear)]
    if not targets:
        raise RuntimeError("No language attention modules found; architecture changed.")
    model = get_peft_model(model, LoraConfig(
        r=config["lora_rank"], lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"], target_modules=targets,
        bias="none", task_type="CAUSAL_LM",
    ))
    trainable = {name: p.numel() for name, p in model.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in name and ".text_model." in name for name in trainable)
    return model, {"trainable_parameters": sum(trainable.values()),
                   "total_parameters": sum(p.numel() for p in model.parameters()),
                   "target_modules": targets, "trainable_names": list(trainable)}


def messages(prompt: str, answer: int | None = None) -> list:
    result = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    if answer is not None:
        result.append({"role": "assistant", "content": [{"type": "text", "text": str(answer)}]})
    return result


def encode(processor, images: list, prompts: list[str], answers: list[int] | None = None):
    prefixes = [processor.apply_chat_template(messages(p), add_generation_prompt=True) for p in prompts]
    if answers is None:
        return processor(text=prefixes, images=[[im] for im in images], return_tensors="pt", padding=True)
    texts = [processor.apply_chat_template(messages(p, a), add_generation_prompt=False)
             for p, a in zip(prompts, answers, strict=True)]
    prefix_batch = processor(text=prefixes, images=[[im] for im in images], return_tensors="pt", padding=True)
    batch = processor(text=texts, images=[[im] for im in images], return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()
    for i, answer in enumerate(answers):
        prefix_ids = prefix_batch["input_ids"][i][prefix_batch["attention_mask"][i].bool()]
        full_ids = batch["input_ids"][i][batch["attention_mask"][i].bool()]
        n = len(prefix_ids)
        assert torch.equal(full_ids[:n], prefix_ids), "Chat template token boundary changed."
        labels[i, :n] = -100
        labels[i, ~batch["attention_mask"][i].bool()] = -100
        supervised = processor.tokenizer.decode(labels[i][labels[i] != -100], skip_special_tokens=True).strip()
        assert supervised == str(answer), f"Unexpected supervised text: {supervised!r}"
    image_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    assert not torch.any((labels != -100) & (batch["input_ids"] == image_id))
    assert torch.all((labels != -100).sum(dim=1) > 0)
    batch["labels"] = labels
    return batch


def to_device(batch):
    return {key: value.to("cuda", dtype=torch.float16 if value.is_floating_point() else value.dtype)
            for key, value in batch.items()}


@torch.inference_mode()
def predict(model, processor, image, prompt: str, max_new_tokens: int = 6, candidates: int | None = None):
    model.eval()
    batch = to_device(encode(processor, [image], [prompt]))
    output = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False,
                            use_cache=True, pad_token_id=processor.tokenizer.pad_token_id,
                            return_dict_in_generate=True)
    new_ids = output.sequences[0, batch["input_ids"].shape[1]:]
    text = processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    if candidates is None:
        return text
    torch.cuda.synchronize()
    diagnostic_started = time.perf_counter()
    ids = [processor.tokenizer.encode(str(i), add_special_tokens=False) for i in range(1, candidates+1)]
    assert all(len(tokens) == 1 for tokens in ids), "Restricted next-token diagnostic requires single-token IDs."
    # A fixed dummy answer locates the canonical digit position after the chat
    # template's leading whitespace. Causal logits before that position cannot
    # see the dummy digit or any subsequent tokens; the real target is not used.
    diagnostic = to_device(encode(processor, [image], [prompt], [1]))
    labels = diagnostic.pop("labels")
    locations = ((labels[0] != -100) & (diagnostic["input_ids"][0] == ids[0][0])).nonzero()
    assert locations.shape[0] == 1
    position = int(locations[0, 0])
    logits = model(**diagnostic, use_cache=False).logits[0, position-1]
    distribution = logits[[tokens[0] for tokens in ids]].float().softmax(dim=-1)
    probabilities = distribution.tolist()
    torch.cuda.synchronize()
    return text, {"restricted_prediction": int(distribution.argmax()) + 1,
                  "restricted_probabilities": probabilities,
                  "diagnostic_latency_s": time.perf_counter() - diagnostic_started}
