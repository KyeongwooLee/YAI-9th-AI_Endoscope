"""Small offline tests; no pretrained weights or dataset download required."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

import collect
from read_result import load_layer


class GradingTests(unittest.TestCase):
    def test_exact_numeric_equivalence(self):
        for response in ("Work\n#### +1,200.00", "#### 1200", "#### 1,200\n"):
            self.assertTrue(collect.grade(response, "reference\n#### 1200")["correct"])
        self.assertTrue(collect.grade("#### -.50", "#### -0.5")["correct"])
        self.assertFalse(collect.grade("#### 1200.001", "#### 1200")["correct"])

    def test_strict_final_answer_and_truncation(self):
        for text in ("The answer is 42", "#### 42 dollars", "#### 4,2", "#### 1/2", "#### nan", "#### 1e3"):
            self.assertFalse(collect.grade(text, "#### 42")["correct"])
        self.assertTrue(collect.grade("#### 1\nCorrection\n#### 42", "#### 42")["correct"])
        result = collect.grade("#### 42", "#### 42", truncated=True)
        self.assertTrue(result["numeric_match"])
        self.assertFalse(result["correct"])
        with self.assertRaises(ValueError):
            collect.grade("#### 42", "invalid reference")

    def test_deterministic_problem_selection(self):
        small = collect.selected_indices(100, 3, 42)
        self.assertEqual(small, collect.selected_indices(100, 200, 42)[:3])
        self.assertEqual(len(set(collect.selected_indices(100, 0, 42))), 100)

    def test_byte_offsets_across_unicode_token_boundaries(self):
        text = "<special>Hi 한글é"
        byte_map = collect.byte_unicode_map()
        tokens = ["<special>"] + [byte_map[b] for b in "Hi 한글é".encode()]

        class Tokenizer:
            all_special_ids = [0]

            def convert_ids_to_tokens(self, ids):
                return [tokens[i] for i in ids]

            def decode(self, ids, **kwargs):
                return text

        decoded, offsets = collect.token_offsets(Tokenizer(), list(range(len(tokens))))
        self.assertEqual(decoded, text)
        self.assertEqual(offsets[0]["text_offset"], [0, 9])
        self.assertEqual(offsets[-1]["text_offset"], [len(text) - 1, len(text)])
        for offset in offsets:
            self.assertLessEqual(offset["text_offset"][0], offset["text_offset"][1])
            self.assertLessEqual(offset["text_offset"][1], len(text))


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.config = Qwen2Config(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                  num_attention_heads=2, num_key_value_heads=1)
        self.records = [{"problem_id": str(i), "prompt_length": 4, "token_ids": [1] * (50 + i)}
                        for i in range(25)]

    def test_basic_reserved_before_detailed(self):
        baseline = sum(collect.estimated_capture_bytes(r, self.config, False) for r in self.records)
        plan = collect.make_capture_plan(self.records, self.config, baseline, 42)
        self.assertEqual(plan["detailed_ids"], [])
        self.assertEqual(plan["total_estimated_bytes"], baseline)
        with self.assertRaises(RuntimeError):
            collect.make_capture_plan(self.records, self.config, baseline - 1, 42)

    def test_detail_selection_does_not_use_correctness(self):
        available = 10 ** 9
        plan = collect.make_capture_plan(self.records, self.config, available, 42)
        other = [dict(r, correct=True) for r in self.records]
        self.assertEqual(plan, collect.make_capture_plan(other, self.config, available, 42))
        self.assertEqual(len(plan["detailed_ids"]), 20)
        self.assertLessEqual(plan["total_estimated_bytes"], available)

    def test_refused_write_preserves_existing_data(self):
        with tempfile.TemporaryDirectory() as root:
            budget = collect.DiskBudget(root)
            path = Path(root) / "record.json"
            budget.json(path, {"saved": True})
            with patch.object(budget, "available", return_value=0):
                with self.assertRaises(RuntimeError):
                    budget.json(path, {"saved": False})
            self.assertEqual(collect.read_json(path), {"saved": True})
            self.assertFalse(path.with_name("record.json.tmp").exists())

    def test_failed_atomic_write_keeps_old_record(self):
        with tempfile.TemporaryDirectory() as root:
            budget = collect.DiskBudget(root)
            path = Path(root) / "record.json"
            budget.json(path, {"saved": True})

            def fail(temp):
                temp.write_text("partial")
                raise OSError("simulated interrupted save")

            with self.assertRaises(OSError):
                budget.write(path, fail, 100)
            self.assertEqual(collect.read_json(path), {"saved": True})
            budget.json(path, {"recovered": True})
            self.assertEqual(collect.read_json(path), {"recovered": True})

    def test_manifest_rejects_changed_settings(self):
        manifest = {"settings": {"seed": 42, "dtype": "bfloat16"}}
        collect.check_manifest(manifest, manifest["settings"])
        with self.assertRaises(ValueError):
            collect.check_manifest(manifest, {"seed": 43, "dtype": "bfloat16"})


class CaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        config = Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                             max_position_embeddings=256, eos_token_id=2)
        config._attn_implementation = "eager"
        cls.model = Qwen2ForCausalLM(config).to(dtype=torch.bfloat16).eval()
        prompt = [1, 7, 11, 3]
        cls.ids = prompt[:]
        stats = []
        cache = None
        current = torch.tensor([prompt])
        with torch.inference_mode():
            for _ in range(133):
                result = cls.model(input_ids=current, past_key_values=cache, use_cache=True, logits_to_keep=1)
                logits = result.logits[0, -1:]
                target = logits.argmax(dim=-1)
                stats.extend(collect.distribution_stats(logits, target, [len(cls.ids)]))
                cls.ids.append(target.item())
                current = target.view(1, 1)
                cache = result.past_key_values
        cls.record = {"run_id": "test-run", "problem_id": "gsm8k/train/7", "prompt_length": len(prompt),
                      "token_ids": cls.ids, "generation_token_stats": stats}

    def test_full_token_coverage_and_detailed_values(self):
        expected = {}
        hooks = []

        def remember(name):
            def hook(_module, _args, output):
                expected[name] = (output[0] if isinstance(output, tuple) else output).detach()[0].clone()
            return hook

        def remember_intermediate(_module, args):
            expected["mlp_intermediate"] = args[0].detach()[0].clone()

        layer = self.model.model.layers[0]
        hooks.append(layer.register_forward_hook(remember("residual")))
        hooks.append(layer.self_attn.register_forward_hook(remember("attention_output")))
        hooks.append(layer.mlp.register_forward_hook(remember("mlp_output")))
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(remember_intermediate))
        try:
            with torch.inference_mode():
                self.model(input_ids=torch.tensor([self.ids]), use_cache=False)
        finally:
            for hook in hooks:
                hook.remove()
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "activations"
            budget = collect.DiskBudget(root)
            metadata = collect.capture_activations(self.model, self.record, True, directory, budget)
            self.assertTrue(metadata["replay_comparison"]["passed"])
            self.assertEqual(len(metadata["response_token_stats"]), 133)
            self.assertEqual(metadata["response_token_stats"][0]["position"], 4)
            self.assertEqual(metadata["response_token_stats"][-1]["position"], len(self.ids) - 1)
            collect.verify_capture(directory, self.record, self.model.config, True)
            shards = [s for s in metadata["shards"] if s["group"] == "layer_00"]
            for name in expected:
                restored = torch.cat([torch.load(directory / s["file"], weights_only=True)["tensors"][name]
                                      for s in shards])
                self.assertEqual(restored.shape[0], len(self.ids))
                self.assertEqual(restored.dtype, torch.bfloat16)
                torch.testing.assert_close(restored, expected[name], atol=1e-5, rtol=0.02)
            actual_size = sum((directory / s["file"]).stat().st_size for s in metadata["shards"])
            self.assertLess(actual_size, collect.estimated_capture_bytes(self.record, self.model.config, True))
            budget.json(Path(root) / "response.json", self.record)
            restored_record, neurons = load_layer(root, "layer_00", "mlp_intermediate")
            self.assertEqual(restored_record["token_ids"], self.ids)
            torch.testing.assert_close(neurons, expected["mlp_intermediate"], atol=1e-5, rtol=0.02)
            for module in self.model.modules():
                self.assertFalse(module._forward_hooks)
                self.assertFalse(module._forward_pre_hooks)

    def test_resume_detects_missing_and_corrupt_files(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "activations"
            budget = collect.DiskBudget(root)
            meta = collect.capture_activations(self.model, self.record, False, directory, budget)
            budget.json(Path(root) / "response.json", self.record)
            collect.verify_capture(directory, self.record, self.model.config, False)
            shard = directory / meta["shards"][-1]["file"]
            original = shard.read_bytes()
            shard.write_bytes(b"x" * len(original))
            with self.assertRaises(ValueError):
                collect.verify_capture(directory, self.record, self.model.config, False)
            with self.assertRaises(ValueError):
                load_layer(root, "final_norm", "final_norm")
            shard.write_bytes(original)
            collect.verify_capture(directory, self.record, self.model.config, False)
            changed = copy.deepcopy(self.record)
            changed["token_ids"][-1] = (changed["token_ids"][-1] + 1) % 64
            with self.assertRaises(ValueError):
                collect.verify_capture(directory, changed, self.model.config, False)
            shard.unlink()
            with self.assertRaises(FileNotFoundError):
                collect.verify_capture(directory, self.record, self.model.config, False)

    def test_interrupted_capture_restarts_and_removes_hooks(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "activations"
            budget = collect.DiskBudget(root)
            original_write = budget.write
            calls = 0

            def fail(path, writer, estimate):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise OSError("simulated interruption")
                original_write(path, writer, estimate)

            with patch.object(budget, "write", side_effect=fail):
                with self.assertRaises(OSError):
                    collect.capture_activations(self.model, self.record, False, directory, budget)
            self.assertFalse((directory / "activation.json").exists())
            for module in self.model.modules():
                self.assertFalse(module._forward_hooks)
            collect.capture_activations(self.model, self.record, False, directory, budget)
            collect.verify_capture(directory, self.record, self.model.config, False)

    def test_replay_mismatch_is_not_complete(self):
        changed = copy.deepcopy(self.record)
        changed["generation_token_stats"][0]["log_probability"] -= 1
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "activations"
            with self.assertRaises(RuntimeError):
                collect.capture_activations(self.model, changed, False, directory, collect.DiskBudget(root))
            self.assertFalse(collect.read_json(directory / "activation.json")["complete"])

    def test_long_input_is_preserved_without_model_call(self):
        class Tokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return "long prompt"

            def __call__(self, *args, **kwargs):
                return {"input_ids": [1] * 1025}

        problem = {"question": "Long problem", "reference_answer": "#### 42", "problem_id": "long"}
        with patch.object(self.model, "forward", side_effect=AssertionError("model must not run")):
            record = collect.generate_response(self.model, Tokenizer(), problem, "test-run")
        self.assertEqual(record["generation_status"], "input_too_long")
        self.assertEqual(record["prompt_length"], 1025)
        self.assertEqual(record["reference_answer"], "#### 42")
        self.assertFalse(record["correct"])


if __name__ == "__main__":
    unittest.main()
