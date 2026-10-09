"""Read one completed item; assemble only the requested layer, not the whole run."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from collect import digest_file, read_json


def load_layer(item_dir, group="layer_00", field="residual"):
    item_dir = Path(item_dir)
    record = read_json(item_dir / "response.json")
    metadata = read_json(item_dir / "activations" / "activation.json")
    if not metadata["complete"] or not metadata["replay_comparison"]["passed"]:
        raise ValueError("Activation capture is not complete/validated")
    token_hash = hashlib.sha256(json.dumps(record["token_ids"]).encode()).hexdigest()
    if metadata["run_id"] != record["run_id"] or metadata["response_sha256"] != token_hash:
        raise ValueError("Activation belongs to a different response/run")
    shards = sorted((s for s in metadata["shards"] if s["group"] == group), key=lambda s: s["token_start"])
    if not shards or field not in shards[0]["shapes"]:
        raise ValueError(f"{group}/{field} was not collected for this item")
    tensors = []
    end = 0
    for shard in shards:
        if shard["token_start"] != end:
            raise ValueError("Missing or overlapping token range")
        path = (item_dir / "activations" / shard["file"]).resolve()
        path.relative_to((item_dir / "activations").resolve())
        if path.stat().st_size != shard["bytes"] or digest_file(path) != shard["sha256"]:
            raise ValueError("Stored tensor is missing or corrupted")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (payload["token_start"], payload["token_end"]) != (shard["token_start"], shard["token_end"]):
            raise ValueError("Shard token range differs from metadata")
        tensor = payload["tensors"][field]
        if list(tensor.shape) != shard["shapes"][field] or not torch.isfinite(tensor).all():
            raise ValueError("Invalid stored tensor")
        tensors.append(tensor)
        end = shard["token_end"]
    if end != len(record["token_ids"]):
        raise ValueError("Final response token is missing")
    return record, torch.cat(tensors, dim=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("item_dir")
    parser.add_argument("--layer", type=int, default=0)
    args = parser.parse_args()
    record, tensor = load_layer(args.item_dir, f"layer_{args.layer:02d}")
    prompt_end = record["prompt_length"]
    print("Question:", record["question"])
    print("Response:", record["response"])
    print("Predicted:", record["predicted_number"], "Reference:", record["reference_number"], "Correct:", record["correct"])
    print("Residual shape:", tuple(tensor.shape), "dtype:", tensor.dtype)
    print("Prompt-final feature:", tuple(tensor[prompt_end - 1].shape))
    print("Response-token features:", tuple(tensor[prompt_end:].shape))


if __name__ == "__main__":
    main()
