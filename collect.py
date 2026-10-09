"""Frozen GSM8K generation followed by token-aligned, budgeted Qwen2 replay."""

import argparse
import codecs
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
import fnmatch
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import time

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_ID = "openai/gsm8k"
SYSTEM_PROMPT = "You are a helpful assistant."
INSTRUCTION = (
    "Solve this math word problem in English. Show your reasoning step by step. "
    "Finish with a separate last line in the exact format: #### <number>. "
    "Use a numeric answer without units on that line.\n\n"
)
ERROR_TYPES = ["CA", "CO", "CV", "HA", "UC", "OP", "FC", "MS", "CS"]
GIB = 1024 ** 3
MAX_NEW_TOKENS = 1024
MAX_SEQUENCE = 2048
CHUNK_SIZE = 128
MAX_DETAILED = 20
FILE_ALLOWANCE = 16 * 1024
REPLAY_ATOL = 0.10
NUMBER = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
FINAL_LINE = re.compile(r"\s*(" + NUMBER + r")\s*\Z")


def answer_number(text):
    if "####" not in text:
        return None, "missing_marker"
    match = FINAL_LINE.fullmatch(text.rsplit("####", 1)[1])
    if not match:
        return None, "invalid_number_or_trailing_text"
    value = Decimal(match[1].replace(",", ""))
    return format(value, "f"), "ok"


def grade(response, reference, truncated=False):
    expected, status = answer_number(reference)
    if status != "ok":
        raise ValueError("GSM8K reference has no valid numeric answer")
    predicted, status = answer_number(response)
    return {
        "reference_number": expected,
        "predicted_number": predicted,
        "answer_status": status,
        "numeric_match": predicted is not None and Decimal(predicted) == Decimal(expected),
        "correct": not truncated and predicted is not None and Decimal(predicted) == Decimal(expected),
    }


def digest_file(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def tree_bytes(root):
    # Count hard links once; Hugging Face snapshot symlinks point into its blob cache.
    seen = set()
    total = 0
    for parent, _, files in os.walk(root):
        for name in files:
            path = Path(parent) / name
            if path.is_symlink():
                continue
            info = path.stat()
            key = (info.st_dev, info.st_ino)
            if key not in seen:
                seen.add(key)
                total += max(info.st_size, getattr(info, "st_blocks", 0) * 512)
    return total


def allocated_bytes(path):
    if not Path(path).exists():
        return 0
    info = Path(path).stat()
    return max(info.st_size, getattr(info, "st_blocks", 0) * 512)


class DiskBudget:
    def __init__(self, workspace):
        self.workspace = Path(workspace).resolve()
        self.refresh()

    def refresh(self):
        self.used = tree_bytes(self.workspace)

    def available(self):
        return min(45 * GIB - self.used, shutil.disk_usage(self.workspace).free - 5 * GIB)

    def check(self, additional):
        if additional > self.available():
            raise RuntimeError(
                f"Disk budget exceeded: need {additional / GIB:.3f} GiB, "
                f"available {self.available() / GIB:.3f} GiB; files preserved."
            )

    def write(self, path, writer, estimate):
        path = Path(path)
        path.resolve().relative_to(self.workspace)
        self.check(estimate)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + ".tmp")
        old = sum(allocated_bytes(p) for p in (path, temp))
        try:
            writer(temp)
            size = temp.stat().st_size
            if size > estimate:
                raise RuntimeError(f"Serialized size exceeded allowance: {temp}")
            os.replace(temp, path)
            self.used += allocated_bytes(path) - old
        except BaseException:
            self.refresh()
            raise

    def json(self, path, value):
        data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
        self.write(path, lambda p: p.write_bytes(data), len(data) + FILE_ALLOWANCE)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@contextmanager
def run_lock(output):
    path = output / ".lock"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RuntimeError(f"Run is locked: {path}. If its process has stopped, remove only this lock and resume.")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps({"pid": os.getpid(), "created": now()}))
        yield
    finally:
        path.unlink()


def now():
    return datetime.now(timezone.utc).isoformat()


def selected_indices(length, limit, seed):
    indices = list(range(length))
    random.Random(seed).shuffle(indices)
    return indices if limit == 0 else indices[:limit]


def tensor_specs(config, detailed):
    specs = {"embedding": {"embedding": config.hidden_size}}
    for layer in range(config.num_hidden_layers):
        fields = {"residual": config.hidden_size}
        if detailed:
            fields.update(attention_output=config.hidden_size, mlp_output=config.hidden_size,
                          mlp_intermediate=config.intermediate_size)
        specs[f"layer_{layer:02d}"] = fields
    specs["final_norm"] = {"final_norm": config.hidden_size}
    return specs


def replay_ranges(prompt_length, total, chunk_size=CHUNK_SIZE):
    yield 0, prompt_length
    for start in range(prompt_length, total, chunk_size):
        yield start, min(start + chunk_size, total)


def estimated_capture_bytes(record, config, detailed):
    total = len(record["token_ids"])
    specs = tensor_specs(config, detailed)
    data = 2 * total * sum(sum(fields.values()) for fields in specs.values())
    chunks = len(list(replay_ranges(record["prompt_length"], total)))
    return data + chunks * len(specs) * FILE_ALLOWANCE + 4 * 1024 ** 2


def make_capture_plan(records, config, available, seed):
    baseline = sum(estimated_capture_bytes(r, config, False) for r in records)
    if baseline > available:
        raise RuntimeError(
            f"Basic activations need {baseline / GIB:.3f} GiB, available {available / GIB:.3f} GiB. "
            "Responses are saved; no activations were started."
        )
    order = list(records)
    random.Random(seed).shuffle(order)
    detailed = []
    planned = baseline
    for record in order[:MAX_DETAILED]:
        extra = estimated_capture_bytes(record, config, True) - estimated_capture_bytes(record, config, False)
        if planned + extra > available:
            break
        detailed.append(record["problem_id"])
        planned += extra
    return {"basic_estimated_bytes": baseline, "total_estimated_bytes": planned,
            "available_bytes_at_planning": available, "detailed_ids": detailed,
            "selection": "seeded order, prefix up to 20; independent of correctness"}


def byte_unicode_map():
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    mapping = {byte: chr(byte) for byte in visible}
    mapping.update({byte: chr(256 + index) for index, byte in
                    enumerate(byte for byte in range(256) if byte not in mapping)})
    return mapping


def token_offsets(tokenizer, ids):

    decoder = {char: byte for byte, char in byte_unicode_map().items()}
    special = set(tokenizer.all_special_ids)
    pieces = []
    for token_id, token in zip(ids, tokenizer.convert_ids_to_tokens(ids)):
        pieces.append(token.encode("utf-8") if token_id in special else bytes(decoder[c] for c in token))
    raw = b"".join(pieces)
    text = raw.decode("utf-8", errors="replace")
    if text != tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        raise ValueError("Tokenizer byte decoder changed; refusing incorrect text offsets")
    incremental = codecs.getincrementaldecoder("utf-8")(errors="replace")
    counts, ceilings = [0], [0]
    count = 0
    for byte in raw:
        count += len(incremental.decode(bytes([byte])))
        counts.append(count)
        ceilings.append(count + bool(incremental.getstate()[0]))
    count += len(incremental.decode(b"", final=True))
    counts[-1] = ceilings[-1] = count
    offsets = []
    start = 0
    for piece in pieces:
        end = start + len(piece)
        offsets.append({"byte_offset": [start, end], "text_offset": [counts[start], ceilings[end]]})
        start = end
    return text, offsets


def distribution_stats(logits, target_ids, positions):
    import torch

    values = logits.float()
    log_probs = values.log_softmax(dim=-1)
    selected = log_probs.gather(-1, target_ids[:, None]).squeeze(-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    top_probs, top_ids = log_probs.topk(min(5, values.shape[-1]), dim=-1)
    if not torch.isfinite(selected).all() or not torch.isfinite(entropy).all():
        raise ValueError("Non-finite token probability")
    return [{"position": position, "token_id": token_id,
             "log_probability": lp, "entropy": ent,
             "top5_ids": candidates, "top5_log_probabilities": candidate_probs}
            for position, token_id, lp, ent, candidates, candidate_probs in zip(
                positions, target_ids.tolist(), selected.tolist(), entropy.tolist(), top_ids.tolist(), top_probs.tolist())]


def eos_ids(model):
    value = model.generation_config.eos_token_id
    if value is None:
        value = model.config.eos_token_id
    if value is None:
        raise ValueError("Model has no EOS token")
    return value if isinstance(value, list) else [value]


def generate_response(model, tokenizer, problem, run_id):
    import torch

    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": INSTRUCTION + problem["question"]}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if len(prompt_ids) + MAX_NEW_TOKENS > MAX_SEQUENCE:
        return {"run_id": run_id, **problem, "generation_complete": False,
                "generation_status": "input_too_long", "prompt": prompt, "prompt_length": len(prompt_ids),
                "correct": False, "error": "Input was not truncated; 2048-token sequence budget exceeded."}
    device = next(model.parameters()).device
    current = torch.tensor([prompt_ids], device=device)
    stop_ids = eos_ids(model)
    ids, stats = list(prompt_ids), []
    cache = None
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    began = time.perf_counter()
    with torch.inference_mode():
        for _ in range(MAX_NEW_TOKENS):
            result = model(input_ids=current, past_key_values=cache, use_cache=True, logits_to_keep=1)
            logits = result.logits[0, -1:]
            target = logits.argmax(dim=-1)
            stats.extend(distribution_stats(logits, target, [len(ids)]))
            token_id = target.item()
            ids.append(token_id)
            cache = result.past_key_values
            if token_id in stop_ids:
                break
            current = target.view(1, 1)
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    seconds = time.perf_counter() - began
    response_ids = ids[len(prompt_ids):]
    truncated = response_ids[-1] not in stop_ids
    response = tokenizer.decode(response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    full_text, offsets = token_offsets(tokenizer, ids)
    return {"run_id": run_id, **problem, "generation_complete": True,
            "generation_status": "complete", "prompt": prompt, "prompt_length": len(prompt_ids),
            "token_ids": ids, "response_token_ids": response_ids, "decoded_full_text": full_text,
            "token_offsets": offsets, "response": response, "generation_token_stats": stats,
            "termination_reason": "max_new_tokens" if truncated else "eos",
            "generated_tokens": len(response_ids), "generation_seconds": seconds,
            "truncated": truncated, **grade(response, problem["reference_answer"], truncated),
            "reasoning_annotation": {"status": "unannotated", "has_error": None, "error_types": None,
                                     "first_error_step": None, "evidence": None, "unclassified_reason": None}}


def replay_comparison(generated, replayed, atol=REPLAY_ATOL):
    if len(generated) != len(replayed):
        raise ValueError("Replay did not score every generated token")
    delta = 0.0
    mismatches = []
    for expected, actual in zip(generated, replayed):
        if (expected["position"], expected["token_id"]) != (actual["position"], actual["token_id"]):
            raise ValueError("Replay probability is aligned to the wrong token")
        delta = max(delta, abs(expected["log_probability"] - actual["log_probability"]))
        if actual["top5_ids"][0] != expected["top5_ids"][0]:
            mismatches.append(actual["position"])
    return {"passed": not mismatches and delta <= atol, "max_log_probability_delta": delta,
            "absolute_tolerance": atol, "top1_mismatch_positions": mismatches}


def capture_activations(model, record, detailed, directory, budget):
    import torch

    device = next(model.parameters()).device
    ids = record["token_ids"]
    prompt_length = record["prompt_length"]
    specs = tensor_specs(model.config, detailed)
    shards, stats, handles, pending, buffered = [], [], [], {}, {}
    start = end = chunk = 0

    def cpu(tensor):
        if not torch.isfinite(tensor).all():
            raise ValueError("Non-finite activation")
        return tensor.detach().to(device="cpu", dtype=torch.bfloat16)[0].contiguous()

    def save(group, tensors):
        expected = specs[group]
        if set(tensors) != set(expected):
            raise ValueError(f"Missing activation in {group}")
        for name, width in expected.items():
            if list(tensors[name].shape) != [end - start, width]:
                raise ValueError(f"Unexpected tensor shape in {group}/{name}")
        path = directory / group / f"chunk_{chunk:04d}.pt"
        payload = {"token_start": start, "token_end": end, "tensors": tensors}
        size = sum(t.numel() * t.element_size() for t in tensors.values()) + FILE_ALLOWANCE
        budget.write(path, lambda p: torch.save(payload, p), size)
        shards.append({"file": path.relative_to(directory).as_posix(), "group": group,
                       "token_start": start, "token_end": end, "dtype": "bfloat16",
                       "shapes": {k: list(t.shape) for k, t in tensors.items()},
                       "bytes": path.stat().st_size, "sha256": digest_file(path)})

    def emit(group, tensors):
        if start == 0:
            save(group, tensors)
        else:
            for name, tensor in tensors.items():
                buffered.setdefault(group, {}).setdefault(name, []).append(tensor)

    def simple_hook(group):
        def hook(_module, _args, output):
            emit(group, {group: cpu(output)})
        return hook

    def output_hook(group, field):
        def hook(_module, _args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            pending.setdefault(group, {})[field] = cpu(tensor)
        return hook

    def intermediate_hook(group):
        def hook(_module, args):
            pending.setdefault(group, {})["mlp_intermediate"] = cpu(args[0])
        return hook

    def residual_hook(group):
        def hook(_module, _args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            tensors = pending.pop(group, {})
            tensors["residual"] = cpu(tensor)
            emit(group, tensors)
        return hook

    handles.append(model.model.embed_tokens.register_forward_hook(simple_hook("embedding")))
    handles.append(model.model.norm.register_forward_hook(simple_hook("final_norm")))
    for index, layer in enumerate(model.model.layers):
        group = f"layer_{index:02d}"
        if detailed:
            handles.append(layer.self_attn.register_forward_hook(output_hook(group, "attention_output")))
            handles.append(layer.mlp.register_forward_hook(output_hook(group, "mlp_output")))
            handles.append(layer.mlp.down_proj.register_forward_pre_hook(intermediate_hook(group)))
        handles.append(layer.register_forward_hook(residual_hook(group)))
    began = time.perf_counter()
    cache = None
    try:
        with torch.inference_mode():
            for chunk, (start, end) in enumerate(replay_ranges(prompt_length, len(ids))):
                steps = [(0, prompt_length)] if start == 0 else [(p, p + 1) for p in range(start, end)]
                for step_start, step_end in steps:
                    current = torch.tensor([ids[step_start:step_end]], device=device)
                    result = model(input_ids=current, past_key_values=cache, use_cache=True, logits_to_keep=1)
                    cache = result.past_key_values
                    if step_end < len(ids):
                        target = torch.tensor([ids[step_end]], device=device)
                        stats.extend(distribution_stats(result.logits[0, -1:], target, [step_end]))
                if start != 0:
                    for group, fields in buffered.items():
                        save(group, {name: torch.cat(values, dim=0) for name, values in fields.items()})
                    buffered.clear()
        comparison = replay_comparison(record["generation_token_stats"], stats)
        metadata = {"run_id": record["run_id"], "problem_id": record["problem_id"],
                    "response_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                    "capture_method": "causal teacher-forced replay: original prefill, then one token per forward",
                    "detailed": detailed, "complete": comparison["passed"],
                    "token_count": len(ids), "prompt_length": prompt_length,
                    "chunk_size": CHUNK_SIZE, "capture_seconds": time.perf_counter() - began,
                    "replay_comparison": comparison, "response_token_stats": stats, "shards": shards}
        budget.json(directory / "activation.json", metadata)
        if not comparison["passed"]:
            raise RuntimeError(f"Replay differs from generation: {directory / 'activation.json'}. "
                               "Saved shards are not marked complete; review before using them.")
        return metadata
    finally:
        for handle in handles:
            handle.remove()


def verify_capture(directory, record, config, detailed):
    metadata = read_json(directory / "activation.json")
    token_hash = hashlib.sha256(json.dumps(record["token_ids"]).encode()).hexdigest()
    if (not metadata["complete"] or metadata["run_id"] != record["run_id"] or
            metadata["problem_id"] != record["problem_id"] or metadata["response_sha256"] != token_hash or
            metadata["detailed"] != detailed or not metadata["replay_comparison"]["passed"]):
        raise ValueError(f"Incomplete or incompatible activation: {directory}")
    expected = {(group, start, end): {k: [end - start, width] for k, width in fields.items()}
                for start, end in replay_ranges(record["prompt_length"], len(record["token_ids"]))
                for group, fields in tensor_specs(config, detailed).items()}
    actual = {}
    for shard in metadata["shards"]:
        path = (directory / shard["file"]).resolve()
        path.relative_to(directory.resolve())
        key = (shard["group"], shard["token_start"], shard["token_end"])
        if key in actual or shard["dtype"] != "bfloat16":
            raise ValueError(f"Duplicate or invalid shard: {path}")
        actual[key] = shard["shapes"]
        if path.stat().st_size != shard["bytes"] or digest_file(path) != shard["sha256"]:
            raise ValueError(f"Activation file is missing or corrupted: {path}")
    if actual != expected:
        raise ValueError(f"Activation does not cover every layer and token: {directory}")
    return metadata


def check_manifest(manifest, settings):
    if manifest["settings"] != settings:
        raise ValueError("Resume settings/code/environment differ from manifest; use the original environment or a new output directory")


def validate_response(record, problem, run_id, tokenizer):
    if record["run_id"] != run_id or any(record[k] != v for k, v in problem.items()):
        raise ValueError("Saved response belongs to a different problem or run")
    if not record["generation_complete"]:
        if record["generation_status"] != "input_too_long":
            raise ValueError("Unknown incomplete generation state")
        return
    p = record["prompt_length"]
    ids = record["token_ids"]
    prompt_ids = tokenizer(record["prompt"], add_special_tokens=False)["input_ids"]
    if (not 0 < p < len(ids) <= MAX_SEQUENCE or ids[:p] != prompt_ids or
            ids[p:] != record["response_token_ids"] or len(ids) - p != record["generated_tokens"]):
        raise ValueError("Saved response tokens or boundary are invalid")
    if tokenizer.decode(ids[p:], skip_special_tokens=True, clean_up_tokenization_spaces=False) != record["response"]:
        raise ValueError("Saved response text differs from tokens")
    stats = record["generation_token_stats"]
    if len(stats) != len(ids) - p or any(s["position"] != p + i or s["token_id"] != ids[p + i]
                                       or not math.isfinite(s["log_probability"]) for i, s in enumerate(stats)):
        raise ValueError("Saved generation probabilities are misaligned")
    if any(record[k] != v for k, v in grade(record["response"], problem["reference_answer"], record["truncated"]).items()):
        raise ValueError("Saved answer grading is inconsistent")


def package_versions():
    return {name: importlib.metadata.version(name) for name in
            ("torch", "transformers", "datasets", "huggingface_hub", "safetensors", "tokenizers", "numpy")}


def run(args):
    output = Path(args.output_dir).resolve()
    workspace = output.parent
    if output == workspace or output.is_symlink():
        raise ValueError("Output must be a dedicated directory")
    existing = output.exists() and any(output.iterdir())
    if existing and not args.resume:
        raise ValueError("Output directory is not empty; use --resume or a new directory")
    if args.resume and not (output / "manifest.json").exists():
        raise ValueError("--resume requires an existing manifest.json")
    workspace.mkdir(parents=True, exist_ok=True)
    cache = workspace / "cache"
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_HUB_CACHE"] = str(cache / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(cache / "datasets")
    os.environ["HF_XET_CACHE"] = str(cache / "xet")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi, snapshot_download, try_to_load_from_cache
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA GPU with BF16 support is required; no precision/device fallback is used")
    random.seed(args.seed)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    budget = DiskBudget(workspace)
    budget.check(64 * 1024 ** 2)
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output):
        manifest_path = output / "manifest.json"
        manifest = read_json(manifest_path) if args.resume else None
        if manifest:
            model_revision = manifest["settings"]["model_revision"]
            dataset_revision = manifest["settings"]["dataset_revision"]
        else:
            model_revision = HfApi().model_info(MODEL_ID).sha
            dataset_revision = HfApi().dataset_info(DATASET_ID).sha
        allow = ["*.json", "*.safetensors", "*.model", "*.txt"]
        info = HfApi().model_info(MODEL_ID, revision=model_revision, files_metadata=True)
        download_size = sum(f.size or 0 for f in info.siblings
                            if any(fnmatch.fnmatch(f.rfilename, p) for p in allow)
                            and not isinstance(try_to_load_from_cache(MODEL_ID, f.rfilename,
                                                                    revision=model_revision), str))
        budget.check(download_size + (512 * 1024 ** 2 if not manifest else FILE_ALLOWANCE))
        model_path = snapshot_download(MODEL_ID, revision=model_revision, allow_patterns=allow)
        budget.refresh()
        budget.check(64 * 1024 ** 2)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
        dataset = load_dataset(DATASET_ID, "main", split=args.split, revision=dataset_revision)
        budget.refresh()
        budget.check(64 * 1024 ** 2)
        if args.limit > len(dataset):
            raise ValueError(f"--limit exceeds split length {len(dataset)}")
        indices = selected_indices(len(dataset), args.limit, args.seed)
        problems = [{"problem_id": f"gsm8k/{args.split}/{i}", "dataset_split": args.split,
                     "dataset_index": i, "question": dataset[i]["question"],
                     "reference_answer": dataset[i]["answer"]} for i in indices]
        settings = {"model_id": MODEL_ID, "model_revision": model_revision, "dataset_id": DATASET_ID,
                    "dataset_revision": dataset_revision, "dataset_config": "main", "split": args.split,
                    "limit": args.limit, "seed": args.seed, "system_prompt": SYSTEM_PROMPT,
                    "instruction": INSTRUCTION, "chat_template": tokenizer.chat_template,
                    "generation": {"method": "raw-logit greedy loop", "batch_size": 1,
                                   "max_new_tokens": MAX_NEW_TOKENS, "max_sequence": MAX_SEQUENCE,
                                   "use_cache": True}, "dtype": "bfloat16", "quantization": None,
                    "attention_implementation": "eager", "cpu_threads": 1, "chunk_size": CHUNK_SIZE,
                    "replay_schedule": "original prefill, then one token per forward; 128-token storage shards",
                    "replay_atol": REPLAY_ATOL, "max_detailed": MAX_DETAILED,
                    "packages": package_versions(), "python": sys.version.split()[0],
                    "cuda_version": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                    "code_sha256": digest_file(Path(__file__)),
                    "problem_ids": [p["problem_id"] for p in problems],
                    "problem_content_sha256": hashlib.sha256(json.dumps(problems, sort_keys=True).encode()).hexdigest()}
        if manifest:
            check_manifest(manifest, settings)
        else:
            manifest = {"schema_version": 1, "created": now(), "settings": settings,
                        "run_id": hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest(),
                        "workspace": str(workspace), "budget_gib": 50, "reserve_gib": 5,
                        "python_environment": sys.prefix, "error_types": ERROR_TYPES,
                        "capture_plan": None, "status": "generating"}
            budget.json(manifest_path, manifest)
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16,
                                                    attn_implementation="eager", local_files_only=True).to("cuda").eval()
        if model.config.model_type != "qwen2":
            raise ValueError("Capture hooks support Qwen2 architecture only")
        budget.refresh()
        records = []
        for index, problem in enumerate(problems):
            item = output / "items" / f"{args.split}_{problem['dataset_index']:05d}"
            path = item / "response.json"
            if path.exists():
                record = read_json(path)
                validate_response(record, problem, manifest["run_id"], tokenizer)
            else:
                budget.check(4 * 1024 ** 2)
                record = generate_response(model, tokenizer, problem, manifest["run_id"])
                budget.json(path, record)
            print(f"[{index + 1}/{len(problems)}] {problem['problem_id']} "
                  f"correct={record['correct']} status={record['generation_status']}", flush=True)
            if record["generation_complete"]:
                records.append(record)
        if manifest["capture_plan"] is None:
            budget.refresh()
            manifest["capture_plan"] = make_capture_plan(records, model.config, budget.available() - 16 * 1024 ** 2, args.seed)
            manifest["status"] = "capturing"
            manifest["model_config"] = model.config.to_dict()
            manifest["tensor_specs_basic"] = tensor_specs(model.config, False)
            manifest["tensor_specs_detailed"] = tensor_specs(model.config, True)
            budget.json(manifest_path, manifest)
        detailed_ids = set(manifest["capture_plan"]["detailed_ids"])
        print(f"Capture plan: {len(records)} basic, {len(detailed_ids)} detailed; "
              f"estimated {manifest['capture_plan']['total_estimated_bytes'] / GIB:.2f} GiB", flush=True)
        for index, record in enumerate(records):
            item = output / "items" / f"{args.split}_{record['dataset_index']:05d}"
            directory = item / "activations"
            path = directory / "activation.json"
            detailed = record["problem_id"] in detailed_ids
            if path.exists() and read_json(path)["complete"]:
                verify_capture(directory, record, model.config, detailed)
            else:
                capture_activations(model, record, detailed, directory, budget)
                verify_capture(directory, record, model.config, detailed)
            print(f"[{index + 1}/{len(records)}] activation saved {record['problem_id']} detailed={detailed}", flush=True)
        budget.refresh()
        summary = {"selected_problems": len(problems), "generated_responses": len(records),
                   "generation_errors": len(problems) - len(records),
                   "correct": sum(r["correct"] for r in records),
                   "accuracy_including_failures": sum(r["correct"] for r in records) / len(problems),
                   "truncated": sum(r["truncated"] for r in records),
                   "answer_parse_failures": sum(r["answer_status"] != "ok" for r in records),
                   "basic_activation_complete": len(records), "detailed_activation_complete": len(detailed_ids),
                   "workspace_bytes": budget.used, "generation_seconds": sum(r["generation_seconds"] for r in records),
                   "generated_tokens": sum(r["generated_tokens"] for r in records)}
        budget.json(output / "summary.json", summary)
        manifest["status"] = "complete_with_generation_errors" if summary["generation_errors"] else "complete"
        manifest["completed"] = now()
        budget.json(manifest_path, manifest)
        print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--limit", type=int, default=200, help="0 selects the entire split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "pilot"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.limit < 0 or args.seed < 0:
        parser.error("--limit and --seed must be non-negative")
    try:
        run(args)
    except (RuntimeError, ValueError, OSError) as error:
        parser.exit(1, f"Stopped: {error}\n")


if __name__ == "__main__":
    main()
