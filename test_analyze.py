"""Offline analysis tests using small tensors in the real collection file format."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

import analyze
from collect import digest_file


def write_item(root, index, correct, complete=True):
    directory = root / "items" / f"train_{index:05d}"
    (directory / "activations/layer_00").mkdir(parents=True)
    ids = [1, 2, 3, 4, 5]  # Two prompt tokens, two text tokens, one EOS.
    token_hash = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    record = {"run_id": "synthetic-run", "problem_id": f"gsm8k/train/{index}",
              "question": f"Question {index}", "generation_complete": True,
              "generation_status": "complete", "correct": correct, "prompt_length": 2,
              "token_ids": ids, "response_token_ids": ids[2:], "answer_status": "ok",
              "truncated": False, "reasoning_annotation": {"status": "unannotated", "has_error": None},
              "generation_token_stats": [{"position": i, "token_id": ids[i],
                    "entropy": (0.2 if correct else 0.9) + index / 100,
                    "log_probability": -0.2 if correct else -0.7,
                    "top5_log_probabilities": [-0.2, -2.0] if correct else [-0.7, -0.8]}
                                         for i in range(2, 5)]}
    analyze.write_json(directory / "response.json", record)
    shards = []
    for layer in range(2):
        group = f"layer_{layer:02d}"
        (directory / "activations" / group).mkdir(exist_ok=True)
        residual = torch.tensor([[index / 100, float(not correct), layer, token / 10]
                                 for token in range(5)], dtype=torch.bfloat16)
        path = directory / "activations" / group / "chunk_0000.pt"
        torch.save({"token_start": 0, "token_end": 5, "tensors": {"residual": residual}}, path)
        shards.append({"group": group, "file": f"{group}/chunk_0000.pt",
                       "token_start": 0, "token_end": 5, "shapes": {"residual": [5, 4]},
                       "bytes": path.stat().st_size, "sha256": digest_file(path)})
    analyze.write_json(directory / "activations/activation.json",
                       {"complete": complete, "run_id": "synthetic-run", "problem_id": record["problem_id"],
                        "response_sha256": token_hash, "replay_comparison": {"passed": complete}, "shards": shards})
    return directory


def make_collection(root, n=20):
    root.mkdir()
    analyze.write_json(root / "manifest.json", {"run_id": "synthetic-run", "status": "capturing",
                                              "model_config": {"num_hidden_layers": 2}})
    return [write_item(root, i, i % 2 == 0) for i in range(n)]


class FeatureTests(unittest.TestCase):
    def test_probability_gap_uses_exp_and_includes_eos(self):
        record = {"prompt_length": 1, "token_ids": [1, 2, 3], "response_token_ids": [2, 3],
                  "generation_token_stats": [
                      {"position": 1, "token_id": 2, "entropy": 1, "log_probability": -1,
                       "top5_log_probabilities": [-1, -2]},
                      {"position": 2, "token_id": 3, "entropy": 5, "log_probability": -3,
                       "top5_log_probabilities": [-3, -4]}]}
        gap = [np.exp(-1)-np.exp(-2), np.exp(-3)-np.exp(-4)]
        np.testing.assert_allclose(analyze.probability_features(record), [3, 5, -2, -3, np.mean(gap), min(gap)])
        record["generation_token_stats"][-1]["position"] = 1
        with self.assertRaisesRegex(ValueError, "misaligned"):
            analyze.probability_features(record)

    def test_residual_token_boundaries(self):
        r = {"prompt_length": 2, "token_ids": [1, 2, 3, 4, 5]}
        residual = torch.tensor([[0, 0], [1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.bfloat16)
        f = analyze.aggregate_residual(r, residual)
        np.testing.assert_array_equal(f["prompt_last"], [1, 2])
        np.testing.assert_array_equal(f["response_last"], [7, 8])
        np.testing.assert_array_equal(f["response_mean"], [5, 6])
        self.assertEqual(f["response_mean"].dtype, np.float32)
        with self.assertRaises(ValueError):
            analyze.aggregate_residual(r, residual[:-1])


class SplitTests(unittest.TestCase):
    def test_transitive_problem_and_question_matches(self):
        items = [{"record": {"problem_id": p, "question": q}} for p, q in
                 [("a", "x y"), ("a", "z"), ("b", "x  y"), ("c", "new")]]
        g = analyze.group_items(items)
        self.assertEqual(len(set(g[:3])), 1)
        self.assertNotEqual(g[0], g[3])

    def test_split_is_reproducible_grouped_and_stratified(self):
        labels = np.array([0] * 40 + [1] * 160)
        groups = np.arange(200)
        first = analyze.split_items(labels, groups, 42)
        second = analyze.split_items(labels, groups, 42)
        self.assertEqual([len(first[n]) for n in analyze.SPLIT_NAMES], [120, 40, 40])
        self.assertEqual([int((labels[first[n]] == 0).sum()) for n in analyze.SPLIT_NAMES], [24, 8, 8])
        for n in first:
            np.testing.assert_array_equal(first[n], second[n])
        self.assertEqual(len(set(np.concatenate(list(first.values())))), 200)
        mixed_labels = np.tile([0, 1], 8)
        mixed_groups = np.repeat(np.arange(8), 2)
        grouped = analyze.split_items(mixed_labels, mixed_groups)
        used = []
        for indices in grouped.values():
            self.assertEqual(set(mixed_labels[indices]), {0, 1})
            g = set(mixed_groups[indices])
            self.assertTrue(all(g.isdisjoint(previous) for previous in used))
            used.append(g)

    def test_insufficient_independent_classes_fails(self):
        with self.assertRaisesRegex(ValueError, "Insufficient data"):
            analyze.split_items([0, 0, 1, 1, 1, 1], np.arange(6))
        with self.assertRaisesRegex(ValueError, "independent problem groups"):
            analyze.split_items([0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 2, 3])


class LearningTests(unittest.TestCase):
    def test_scaler_fit_and_models_ignore_test_values_and_labels(self):
        x = np.array([[0., 1], [1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1], [7, 1]])
        y = np.tile([0, 1], 4)
        split = {"train": np.arange(4), "validation": np.array([4, 5]), "test": np.array([6, 7])}
        altered_x, altered_y = x.copy(), y.copy()
        altered_x[split["test"]] = 1e8
        altered_y[split["test"]] = 1 - altered_y[split["test"]]
        for fit in [lambda a, b: analyze.fit_logistic(a, b, split, 1., 42),
                    lambda a, b: analyze.fit_mlp(a, b, split, 42)]:
            first, second = fit(x, y), fit(altered_x, altered_y)
            np.testing.assert_array_equal(first["scaler_mean"].numpy(), [1.5, 1])
            np.testing.assert_array_equal(first["scaler_scale"].numpy(), [np.sqrt(1.25), 1])
            np.testing.assert_array_equal(analyze.predict_scores(first, x), analyze.predict_scores(second, x))
            a = analyze.candidate(first, x, y, split, "model", family="test")
            b = analyze.candidate(second, altered_x, altered_y, split, "model", family="test")
            self.assertEqual(a, b)

    def test_selection_uses_validation_and_stable_tie_break(self):
        a = {"name": "a", "validation": {"auroc": .6, "auprc": .8}, "test": {"auroc": 1.}}
        b = {"name": "b", "validation": {"auroc": .7, "auprc": .6}, "test": {"auroc": 0.}}
        c = {"name": "c", "validation": {"auroc": .7, "auprc": .8}}
        self.assertIs(analyze.select_candidate([a, b, c, copy.deepcopy(c)]), c)
        threshold = analyze.choose_threshold([0, 1, 1], [.1, .6, .8])
        self.assertEqual(threshold, .6)

    def test_single_class_and_empty_metrics_are_serializable(self):
        result = analyze.metrics([0, 0], [.1, .2], .5)
        self.assertIsNone(result["auroc"])
        self.assertIsNone(result["auprc"])
        self.assertIsNone(result["recall"])
        self.assertEqual(result["ranking_metrics_undefined_reason"], "single_class_subset")
        self.assertEqual(result["confusion_matrix"], [[2, 0], [0, 0]])
        json.dumps(result, allow_nan=False)
        empty = analyze.metrics([], [], .5)
        self.assertEqual(empty["n"], 0)
        self.assertIsNone(empty["f1"])


class IntegrationTests(unittest.TestCase):
    def test_end_to_end_and_saved_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "input"
            paths = make_collection(root)
            write_item(root, 20, True, complete=False)
            output = Path(directory) / "analysis"
            manifest = analyze.run_analysis(root, output)
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(len(manifest["items"]), 20)
            self.assertEqual(manifest["excluded_counts"], {"activation_incomplete": 1})
            self.assertEqual(manifest["source_status"], "capturing")
            self.assertTrue((output / "report.md").exists())
            report = (output / "report.md").read_text()
            self.assertIn(manifest["selections"]["probability_baseline"], report)
            self.assertIn("숫자 추출 성공·비잘림 문항에 대한 보조 평가", report)
            candidates = analyze.read_json(output / "candidates.json")
            self.assertEqual(len(candidates), 6 + 4 + 2 * 3 * 4 + 2)
            for c in candidates:
                if c["name"].startswith("mlp_"):
                    linear_name = manifest["selections"][c["timing"] + "_linear"]
                    linear = next(e for e in candidates if e["name"] == linear_name)
                    self.assertEqual((c["layer"], c["pooling"]), (linear["layer"], linear["pooling"]))
            rows = analyze.read_json(output / "test_predictions.json")
            for name in manifest["selections"].values():
                row = next(r for r in rows if r["model"] == name)
                predicted = analyze.predict_item(output / "models" / f"{name}.pt", row["item_dir"])
                self.assertAlmostEqual(predicted["error_score"], row["error_score"], places=6)
                self.assertEqual(predicted["predicted_error"], row["predicted_error"])
            # Prediction does not require labels, reference answers, or process annotations.
            response_path = paths[0] / "response.json"
            record = analyze.read_json(response_path)
            for k in ["correct", "reasoning_annotation"]:
                record.pop(k)
            analyze.write_json(response_path, record)
            logistic_name = manifest["selections"]["post_generation_linear"]
            analyze.predict_item(output / "models" / f"{logistic_name}.pt", paths[0])
            with self.assertRaisesRegex(ValueError, "not empty"):
                analyze.run_analysis(root, output)

    def test_corrupted_completed_tensor_stops_with_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "input"
            paths = make_collection(root, 6)
            broken = paths[2] / "activations/layer_01/chunk_0000.pt"
            broken.write_bytes(b"broken")
            output = Path(directory) / "analysis"
            with self.assertRaisesRegex(ValueError, "train_00002/layer_01"):
                analyze.run_analysis(root, output)
            self.assertEqual(analyze.read_json(output / "analysis_manifest.json")["status"], "failed")
            self.assertFalse((output / "report.md").exists())

    def test_snapshot_exclusions_and_response_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "input"
            paths = make_collection(root, 6)
            r = analyze.read_json(paths[0] / "response.json")
            r["generation_complete"] = False
            analyze.write_json(paths[0] / "response.json", r)
            (paths[1] / "activations/activation.json").unlink()
            _, items, excluded = analyze.snapshot_items(root)
            self.assertEqual(len(items), 4)
            self.assertEqual({e["reason"] for e in excluded}, {"generation_incomplete_or_failed", "activation_missing"})
            r = analyze.read_json(items[0]["item_dir"] / "response.json")
            r["correct"] = not r["correct"]
            analyze.write_json(items[0]["item_dir"] / "response.json", r)
            with self.assertRaisesRegex(ValueError, "changed after"):
                analyze.extract_layer(items, "layer_00")

    def test_validation_does_not_claim_corruption_is_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "input"
            paths = make_collection(root, 6)
            path = paths[0] / "activations/activation.json"
            metadata = analyze.read_json(path)
            metadata["replay_comparison"]["passed"] = False
            analyze.write_json(path, metadata)
            with self.assertRaisesRegex(ValueError, "Completed activation failed"):
                analyze.snapshot_items(root)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
