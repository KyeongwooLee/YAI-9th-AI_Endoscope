"""Offline correctness probes for validated collection outputs (error is class 1)."""

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import re
import warnings

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from collect import digest_file, read_json
from read_result import load_layer

FEATURE_NAMES = ("mean_entropy", "max_entropy", "mean_log_probability",
                 "min_log_probability", "mean_probability_gap", "min_probability_gap")
POOLINGS = ("prompt_last", "response_last", "response_mean")
CS = (0.01, 0.1, 1.0, 10.0)
SPLIT_NAMES = ("train", "validation", "test")
RATIOS = np.array([0.6, 0.2, 0.2])


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
    temp.replace(path)


def snapshot_items(input_dir):
    """Freeze eligible responses and metadata; never use unannotated process labels."""
    root = Path(input_dir).resolve()
    source = read_json(root / "manifest.json")
    items, excluded = [], []
    for path in sorted((root / "items").glob("*/response.json")):
        try:
            record = read_json(path)
            reason = None
            if not record.get("generation_complete") or record.get("generation_status") != "complete":
                reason = "generation_incomplete_or_failed"
            elif not (path.parent / "activations/activation.json").exists():
                reason = "activation_missing"
            else:
                metadata = read_json(path.parent / "activations/activation.json")
                if not metadata.get("complete"):
                    reason = "activation_incomplete"
                elif not metadata.get("replay_comparison", {}).get("passed"):
                    raise ValueError("Completed activation failed replay validation")
            if reason:
                excluded.append({"item_dir": str(path.parent), "problem_id": record.get("problem_id"),
                                 "reason": reason})
                continue
            if type(record.get("correct")) is not bool:
                raise ValueError("correct must be a stored boolean")
            if record["run_id"] != source["run_id"] or metadata["problem_id"] != record["problem_id"]:
                raise ValueError("Response/activation belongs to another run or problem")
            if metadata["run_id"] != record["run_id"]:
                raise ValueError("Activation belongs to another run")
            token_hash = hashlib.sha256(json.dumps(record["token_ids"]).encode()).hexdigest()
            if metadata["response_sha256"] != token_hash:
                raise ValueError("Activation belongs to another response")
            layers = sorted({s["group"] for s in metadata["shards"]
                             if re.fullmatch(r"layer_\d+", s["group"])})
            if not layers:
                raise ValueError("No residual layers collected")
            expected = source.get("model_config", {}).get("num_hidden_layers")
            if expected is not None and layers != [f"layer_{i:02d}" for i in range(expected)]:
                raise ValueError("Completed activation is missing layers")
            if items and layers != items[0]["layers"]:
                raise ValueError("Items have different residual layers")
            probability_features(record)  # Check token alignment before any learning.
            items.append({"item_dir": path.parent, "record": record, "layers": layers,
                          "response_sha256": digest_file(path),
                          "activation_sha256": digest_file(path.parent / "activations/activation.json")})
        except (KeyError, ValueError, OSError, TypeError) as exc:
            raise ValueError(f"Invalid completed collection item {path.parent}: {exc}") from exc
    if not items:
        raise ValueError("No validated activation items are available")
    return source, items, excluded


def probability_features(record):
    p = record["prompt_length"]
    ids = record["token_ids"]
    stats = record["generation_token_stats"]
    if not 0 < p < len(ids) or len(stats) != len(ids) - p or ids[p:] != record["response_token_ids"]:
        raise ValueError("Response token boundary or probability count is invalid")
    values = []
    for i, s in enumerate(stats):
        top = np.asarray(s["top5_log_probabilities"], dtype=np.float64)
        if (s["position"] != p + i or s["token_id"] != ids[p + i] or len(top) < 2
                or not np.isfinite(top).all() or np.any(top > 1e-6)
                or np.any(np.diff(top) > 1e-6)):
            raise ValueError("Generated token probabilities are misaligned or invalid")
        ent, lp = float(s["entropy"]), float(s["log_probability"])
        if not np.isfinite([ent, lp]).all() or ent < 0 or lp > 1e-6:
            raise ValueError("Invalid entropy/log probability")
        values.append((ent, lp, np.exp(top[0]) - np.exp(top[1])))
    ent, lp, gap = np.asarray(values).T
    return np.array([ent.mean(), ent.max(), lp.mean(), lp.min(), gap.mean(), gap.min()])


def aggregate_residual(record, residual):
    p = record["prompt_length"]
    if residual.ndim != 2 or residual.shape[0] != len(record["token_ids"]) or not 0 < p < len(residual):
        raise ValueError("Residual token boundary/shape is invalid")
    residual = residual.float()
    response = residual[p:]  # Includes the final generated token, including EOS.
    return {"prompt_last": residual[p - 1].numpy().copy(),
            "response_last": response[-1].numpy().copy(),
            "response_mean": response.mean(dim=0).numpy().copy()}


def group_items(items):
    """Union problem-ID and normalized-question matches, including transitive matches."""
    parents = list(range(len(items)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen_id, seen_question = {}, {}
    for i, item in enumerate(items):
        r = item["record"]
        question = " ".join(r["question"].split())
        if not question:
            raise ValueError("Cannot group an empty question")
        for table, key in ((seen_id, r["problem_id"]), (seen_question, question)):
            if key in table:
                parents[find(i)] = find(table[key])
            else:
                table[key] = i
    return np.array([find(i) for i in range(len(items))])


def split_items(labels, groups, seed=42):
    """Group-stratify with guaranteed class coverage and closest attainable ratios."""
    labels, groups = np.asarray(labels), np.asarray(groups)
    if labels.shape != groups.shape or not np.isin(labels, [0, 1]).all():
        raise ValueError("Invalid labels/groups")
    members = [np.flatnonzero(groups == g) for g in sorted(set(groups))]
    counts = np.array([np.bincount(labels[m], minlength=2) for m in members])
    if len(members) < 3 or np.any((counts > 0).sum(axis=0) < 3):
        raise ValueError("Insufficient data: each class must occur in at least three independent problem groups")
    target = RATIOS[:, None] * np.bincount(labels, minlength=2)[None, :]
    rng = np.random.default_rng(seed)
    best = None
    # Mixed-label duplicate groups can cover both classes; then fill missing classes.
    # Multiple deterministic trials reduce deviations caused by differently sized groups.
    for _ in range(128):
        allocation = [[], [], []]
        totals = np.zeros((3, 2), dtype=int)
        mixed = list(rng.permutation(np.flatnonzero((counts > 0).all(axis=1))))
        pure = [list(rng.permutation(np.flatnonzero((counts[:, c] > 0) & (counts[:, 1-c] == 0))))
                for c in (0, 1)]

        def add(part, g):
            allocation[part].append(int(g))
            totals[part] += counts[g]

        for part in range(min(3, len(mixed))):
            add(part, mixed.pop())
        for c in (0, 1):
            for part in range(3):
                if totals[part, c] == 0:
                    add(part, pure[c].pop())
        remaining = mixed + pure[0] + pure[1]
        rng.shuffle(remaining)
        for g in remaining:
            costs = []
            for part in range(3):
                trial = totals.copy()
                trial[part] += counts[g]
                costs.append(float(np.square((trial - target) / np.maximum(target, 1)).sum()))
            add(int(np.argmin(costs)), g)
        cost = float(np.square((totals - target) / np.maximum(target, 1)).sum())
        if best is None or cost < best[0]:
            best = cost, allocation
    return {name: np.sort(np.concatenate([members[g] for g in allocation])).astype(int)
            for name, allocation in zip(SPLIT_NAMES, best[1])}


def choose_threshold(labels, scores):
    # Highest threshold wins an F1 tie. All choices come from validation only.
    candidates = sorted(set(float(s) for s in scores), reverse=True)
    return max(candidates, key=lambda t: f1_score(labels, np.asarray(scores) >= t, zero_division=0))


def metrics(labels, scores, threshold):
    labels, scores = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    if labels.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("Invalid metric inputs")
    both = len(set(labels)) == 2
    pred = scores >= threshold
    n = len(labels)
    reason = None if both else ("empty_subset" if n == 0 else "single_class_subset")
    tp = int(((labels == 1) & pred).sum())
    predicted_positive, actual_positive = int(pred.sum()), int(labels.sum())
    undefined = {}
    if not predicted_positive:
        undefined["precision"] = "no_predicted_errors"
    if not actual_positive:
        undefined["recall"] = "no_actual_errors"
    if not predicted_positive and not actual_positive:
        undefined["f1"] = "no_actual_or_predicted_errors"
    return {"n": n, "correct_count": int((labels == 0).sum()), "error_count": int((labels == 1).sum()),
            "error_prevalence": float(labels.mean()) if n else None,
            "auroc": float(roc_auc_score(labels, scores)) if both else None,
            "auprc": float(average_precision_score(labels, scores)) if both else None,
            "ranking_metrics_undefined_reason": reason,
            "precision": tp / predicted_positive if predicted_positive else None,
            "recall": tp / actual_positive if actual_positive else None,
            "f1": 2 * tp / (predicted_positive + actual_positive) if predicted_positive + actual_positive else None,
            "classification_metrics_undefined_reasons": undefined,
            "confusion_matrix": confusion_matrix(labels, pred, labels=[0, 1]).tolist() if n else [[0, 0], [0, 0]],
            "threshold": float(threshold), "confusion_matrix_label_order": ["correct", "error"]}


def selection_key(candidate):
    m = candidate["validation"]
    return m["auroc"], m["auprc"]


def select_candidate(candidates):
    # max is stable: candidate order is the final tie breaker.
    return max(candidates, key=selection_key)


def scale_features(features, artifact):
    return (np.asarray(features, dtype=np.float64) - artifact["scaler_mean"].numpy()) / artifact["scaler_scale"].numpy()


def fit_logistic(features, labels, split, c, seed):
    train = split["train"]
    scaler = StandardScaler().fit(np.asarray(features[train], dtype=np.float64))
    artifact = {"format_version": 1, "kind": "logistic",
                "scaler_mean": torch.from_numpy(scaler.mean_.copy()),
                "scaler_scale": torch.from_numpy(scaler.scale_.copy()), "C": c}
    x = scale_features(features, artifact)
    model = LogisticRegression(C=c, solver="lbfgs", max_iter=1000, random_state=seed)
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x[train], labels[train])
    artifact.update(weight=torch.from_numpy(model.coef_[0].copy()), bias=float(model.intercept_[0]))
    return artifact


def make_mlp(width):
    return torch.nn.Sequential(torch.nn.Linear(width, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))


def fit_mlp(features, labels, split, seed):
    torch.manual_seed(seed)
    train, val = split["train"], split["validation"]
    scaler = StandardScaler().fit(np.asarray(features[train], dtype=np.float64))
    artifact = {"format_version": 1, "kind": "mlp", "input_width": features.shape[1],
                "scaler_mean": torch.from_numpy(scaler.mean_.copy()),
                "scaler_scale": torch.from_numpy(scaler.scale_.copy())}
    x = torch.tensor(scale_features(features, artifact), dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    model = make_mlp(features.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    for epoch in range(1, 201):
        model.train()
        order = torch.tensor(train)[torch.randperm(len(train))]
        for batch in order.split(32):
            optimizer.zero_grad()
            loss = loss_fn(model(x[batch]).squeeze(-1), y[batch])
            if not torch.isfinite(loss):
                raise ValueError("Non-finite MLP training loss")
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x[val]).squeeze(-1), y[val]))
        if val_loss < best_loss:
            best_loss, best_state, best_epoch, stale = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        if stale >= 20:
            break
    artifact.update(state_dict=best_state, best_epoch=best_epoch, epochs_run=epoch,
                    best_validation_loss=best_loss)
    return artifact


def predict_scores(artifact, features):
    """Predict using only frozen model parameters and label-free numeric features."""
    if artifact["kind"] == "raw_score":
        return np.asarray(features)[:, artifact["feature_index"]] * artifact["direction"]
    x = scale_features(features, artifact)
    with torch.no_grad():
        if artifact["kind"] == "logistic":
            logits = torch.from_numpy(x) @ artifact["weight"] + artifact["bias"]
        elif artifact["kind"] == "mlp":
            model = make_mlp(artifact["input_width"])
            model.load_state_dict(artifact["state_dict"])
            model.eval()
            logits = model(torch.tensor(x, dtype=torch.float32)).squeeze(-1)
        else:
            raise ValueError("Unknown artifact kind")
        return torch.sigmoid(logits).numpy()


def candidate(artifact, features, labels, split, name, **description):
    val = split["validation"]
    scores = predict_scores(artifact, features[val])
    threshold = choose_threshold(labels[val], scores)
    artifact.update(name=name, threshold=threshold, description=description,
                    positive_class="error", label_source="not response.correct")
    return {"name": name, **description, "validation": metrics(labels[val], scores, threshold),
            "threshold": threshold}


def extract_layer(items, layer):
    pooled = {name: [] for name in POOLINGS}
    for item in items:
        directory = item["item_dir"]
        try:
            if (digest_file(directory / "response.json") != item["response_sha256"]
                    or digest_file(directory / "activations/activation.json") != item["activation_sha256"]):
                raise ValueError("Collection item changed after the input snapshot")
            record, residual = load_layer(directory, layer, "residual")
            vectors = aggregate_residual(record, residual)
            for name, vector in vectors.items():
                pooled[name].append(vector)
        except (KeyError, ValueError, OSError, RuntimeError) as exc:
            raise ValueError(f"Cannot read validated activation {directory}/{layer}: {exc}") from exc
    try:
        return {name: np.stack(vectors) for name, vectors in pooled.items()}
    except ValueError as exc:
        raise ValueError(f"Residual widths are inconsistent in {layer}") from exc


def predict_item(model_path, item_dir):
    """Public inference helper; correct/reference answers are never prediction inputs."""
    artifact = torch.load(model_path, map_location="cpu", weights_only=True)
    description = artifact["description"]
    if description["family"] == "probability":
        feature = probability_features(read_json(Path(item_dir) / "response.json"))
    else:
        record, residual = load_layer(item_dir, description["layer"], "residual")
        feature = aggregate_residual(record, residual)[description["pooling"]]
    score = float(predict_scores(artifact, feature[None, :])[0])
    return {"error_score": score,
            "error_probability": None if artifact["kind"] == "raw_score" else score,
            "threshold": artifact["threshold"], "predicted_error": score >= artifact["threshold"]}


def report_text(manifest, evaluated, selections):
    def fmt(value):
        return "—" if value is None else f"{value:.4f}"

    lines = ["# 정답·오답 판별 분석", "",
             f"입력 수집 상태: `{manifest['source_status']}`. 분석 문항 {len(manifest['items'])}개, "
             f"제외 {len(manifest['excluded'])}개. 오답을 양성 클래스로 평가했습니다.", "",
             "`correct`에는 숫자 오답, 답 형식 실패, 잘림이 포함됩니다. 풀이 과정 오류 판정이 아닙니다.", "",
             "| 분할 | 전체 | 정답 | 오답 |", "| --- | ---: | ---: | ---: |"]
    for name, m in manifest["split_counts"].items():
        lines.append(f"| {name} | {m['n']} | {m['correct']} | {m['error']} |")
    lines += ["", "Validation에서만 레이어·집계·C·임계값을 선택했습니다. "
              "AUPRC는 average precision으로 계산하며 무작위 점수의 기준은 오답 비율입니다.", "",
              "| 선택 모델 | 시점 | Test AUROC | Test AUPRC | Precision | Recall | F1 |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, result in evaluated.items():
        m = result["test"]
        timing = result["description"]["timing"]
        lines.append(f"| {name} | {timing} | " + " | ".join(fmt(m[k]) for k in
                     ("auroc", "auprc", "precision", "recall", "f1")) + " |")
    baseline = evaluated[selections["probability_baseline"]]["test"]
    linear = evaluated[selections["post_generation_linear"]]["test"]
    mlp = evaluated[selections["post_generation_mlp"]]["test"]
    lines += ["", f"Validation에서 선택한 확률 기준선은 `{selections['probability_baseline']}`입니다. "
              f"이 기준선 대비 사후 선형 probe의 Test AUROC 차이: "
              f"{linear['auroc'] - baseline['auroc']:+.4f}. "
              f"동일 특징 MLP의 차이: {mlp['auroc'] - baseline['auroc']:+.4f}."]
    max_entropy = evaluated["probability_max_entropy"]["test"]
    combined = evaluated[selections["probability_logistic"]]["test"]
    prevalence = baseline["error_prevalence"]
    lines += ["", f"결합 확률 로지스틱 회귀의 Test AUROC는 {combined['auroc']:.4f}, "
              f"미리 정의한 최대 entropy 점수의 Test AUROC는 {max_entropy['auroc']:.4f}입니다. "
              "위 표는 고정된 개별 기준선도 모두 보고하며 Test 결과로 모델을 다시 선택하지 않습니다.", "",
              f"Test 오답 비율은 {prevalence:.4f}이며, 모든 문항을 오답으로 판정하는 비교값의 "
              f"F1은 {2 * prevalence / (1 + prevalence):.4f}입니다. "
              "분류 F1과 임계값에 무관한 순위 판별 성능을 함께 확인해야 합니다.", "",
              "숫자 추출 성공·비잘림 문항에 대한 보조 평가:", "",
              "| 모델 | 전체 | 정답 | 오답 | AUROC | AUPRC |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, result in evaluated.items():
        m = result["numeric_valid_test"]
        lines.append(f"| {name} | {m['n']} | {m['correct_count']} | {m['error_count']} | "
                     f"{fmt(m['auroc'])} | {fmt(m['auprc'])} |")
    lines += ["",
              "전체 평가와 숫자 추출 성공 집합에서 결과가 다를 수 있으므로, "
              "형식 실패를 포함한 최종 정오 판별을 순수한 숫자 오답 판별이나 풀이 과정 오류 탐지로 해석하지 않습니다.", "",
              "이 차이는 해당 고정 분할의 탐색 결과이며 통계적 유의성이나 일반화를 입증하지 않습니다. "
              "생성 전 probe는 프롬프트만 사용하므로 사후 확률 기준선과 시점이 다릅니다.", "",
              "후보별 validation 지표는 `candidates.json`, test·숫자 추출 성공 보조 지표는 "
              "`metrics.json`, 예측은 `test_predictions.json`에 있습니다. "
              "보조 평가에서는 모델이나 임계값을 다시 선택하지 않았습니다.", "",
              "개별 확률 특징 기준선은 보정되지 않은 오답 점수입니다. "
              "해당 예측의 `error_probability`는 null이며, 학습된 모델만 오답 확률을 출력합니다.", "",
              "문항별 activation은 저장 토큰의 causal replay 값입니다. "
              "EOS를 포함한 응답 마지막 토큰·평균은 사후 진단으로 해석합니다.", ""]
    return "\n".join(lines)


def run_analysis(input_dir, output_dir, seed=42):
    root, output = Path(input_dir).resolve(), Path(output_dir).resolve()
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("Analysis output must be separate from collection input")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty; choose a new directory: {output}")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    snapshot_at = datetime.now(timezone.utc).isoformat()
    source, items, excluded = snapshot_items(root)
    labels = np.array([int(not item["record"]["correct"]) for item in items])
    groups = group_items(items)
    split = split_items(labels, groups, seed)
    output.mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir()
    assignments = {int(i): name for name, indices in split.items() for i in indices}
    manifest = {"schema_version": 1, "status": "running",
                "snapshot_at": snapshot_at,
                "input_dir": str(root), "source_run_id": source["run_id"],
                "source_status": source["status"], "source_settings": source.get("settings"),
                "label": "error = int(not response.correct)",
                "seed": seed, "split_ratios": dict(zip(SPLIT_NAMES, RATIOS.tolist())),
                "group_rule": "same problem_id OR whitespace-normalized question; transitive union",
                "layers": items[0]["layers"], "poolings": list(POOLINGS), "C_candidates": list(CS),
                "feature_names": list(FEATURE_NAMES), "include_generated_eos": True,
                "selection": "validation AUROC, then AUPRC, then fixed candidate order",
                "threshold_selection": "validation max error F1, then highest threshold",
                "standardization": "train only; no refit after model selection",
                "mlp": {"hidden_width": 32, "activation": "ReLU", "optimizer": "Adam",
                        "learning_rate": 0.001, "batch_size": 32, "max_epochs": 200,
                        "patience": 20, "early_stopping": "validation BCE loss", "device": "cpu"},
                "packages": {p: importlib.metadata.version(p) for p in
                             ("numpy", "torch", "scikit-learn", "scipy")},
                "items": [{"item_dir": str(item["item_dir"]), "problem_id": item["record"]["problem_id"],
                           "response_sha256": item["response_sha256"],
                           "activation_sha256": item["activation_sha256"],
                           "correct": item["record"]["correct"], "group": int(groups[i]),
                           "split": assignments[i]} for i, item in enumerate(items)],
                "excluded": excluded, "excluded_counts": dict(Counter(e["reason"] for e in excluded)),
                "answer_status_counts": dict(Counter(item["record"].get("answer_status") for item in items)),
                "truncated_count": sum(bool(item["record"].get("truncated")) for item in items),
                "split_counts": {name: {"n": len(indices), "correct": int((labels[indices] == 0).sum()),
                                        "error": int(labels[indices].sum())} for name, indices in split.items()}}
    write_json(output / "analysis_manifest.json", manifest)
    try:
        candidates, selected = [], {}
        probability = np.stack([probability_features(item["record"]) for item in items])
        baseline_entries = []
        for i, name in enumerate(FEATURE_NAMES):
            artifact = {"format_version": 1, "kind": "raw_score", "feature_index": i,
                        "direction": 1 if i < 2 else -1}
            entry = candidate(artifact, probability, labels, split, f"probability_{name}",
                              family="probability", timing="post_generation", feature=name)
            candidates.append(entry)
            baseline_entries.append((entry, artifact))
            selected[entry["name"]] = (artifact, probability)
        fitted = []
        for c in CS:
            artifact = fit_logistic(probability, labels, split, c, seed)
            entry = candidate(artifact, probability, labels, split, f"probability_logistic_C{c:g}",
                              family="probability", timing="post_generation", C=c)
            candidates.append(entry)
            fitted.append((entry, artifact))
        best_entry = select_candidate([e for e, _ in fitted])
        best_artifact = next(a for e, a in fitted if e is best_entry)
        selected[best_entry["name"]] = best_artifact, probability
        baseline_entries.append((best_entry, best_artifact))
        strongest_baseline = select_candidate([e for e, _ in baseline_entries])
        best_by_timing = {}
        for layer in manifest["layers"]:
            features = extract_layer(items, layer)
            for pooling in POOLINGS:
                timing = "pre_generation" if pooling == "prompt_last" else "post_generation"
                for c in CS:
                    artifact = fit_logistic(features[pooling], labels, split, c, seed)
                    entry = candidate(artifact, features[pooling], labels, split,
                                      f"linear_{layer}_{pooling}_C{c:g}", family="activation",
                                      timing=timing, layer=layer, pooling=pooling, C=c)
                    candidates.append(entry)
                    previous = best_by_timing.get(timing)
                    if previous is None or selection_key(entry) > selection_key(previous[0]):
                        best_by_timing[timing] = entry, artifact, features[pooling]
            write_json(output / "candidates.json", candidates)
            print(f"Validated and probed {layer} ({len(items)} items)", flush=True)
        selections = {"probability_baseline": strongest_baseline["name"],
                      "probability_logistic": best_entry["name"]}
        for timing, (linear_entry, artifact, features) in best_by_timing.items():
            selected[linear_entry["name"]] = artifact, features
            selections[timing + "_linear"] = linear_entry["name"]
            mlp = fit_mlp(features, labels, split, seed)
            entry = candidate(mlp, features, labels, split, f"mlp_{timing}", family="activation",
                              timing=timing, layer=linear_entry["layer"], pooling=linear_entry["pooling"],
                              best_epoch=mlp["best_epoch"], epochs_run=mlp["epochs_run"],
                              best_validation_loss=mlp["best_validation_loss"])
            candidates.append(entry)
            selected[entry["name"]] = mlp, features
            selections[timing + "_mlp"] = entry["name"]
        # Test scores/labels are consumed only after all models and thresholds are frozen.
        evaluated, predictions = {}, []
        test = split["test"]
        numeric = np.array([items[i]["record"].get("answer_status") == "ok"
                            and not items[i]["record"].get("truncated", False) for i in test])
        for name, (artifact, features) in selected.items():
            scores = predict_scores(artifact, features[test])
            threshold = artifact["threshold"]
            evaluated[name] = {"description": artifact["description"],
                               "validation": next(e["validation"] for e in candidates if e["name"] == name),
                               "test": metrics(labels[test], scores, threshold),
                               "numeric_valid_test": metrics(labels[test][numeric], scores[numeric], threshold),
                               "artifact": f"models/{name}.pt"}
            artifact.update(seed=seed, source_run_id=source["run_id"])
            torch.save(artifact, output / "models" / f"{name}.pt")
            for i, score, numeric_valid in zip(test, scores, numeric):
                predictions.append({"model": name, "problem_id": items[i]["record"]["problem_id"],
                                    "item_dir": str(items[i]["item_dir"]), "correct": bool(labels[i] == 0),
                                    "error_label": int(labels[i]), "error_score": float(score),
                                    "error_probability": None if artifact["kind"] == "raw_score" else float(score),
                                    "threshold": threshold, "predicted_error": bool(score >= threshold),
                                    "numeric_valid": bool(numeric_valid)})
        write_json(output / "candidates.json", candidates)
        write_json(output / "metrics.json", {"selections": selections, "models": evaluated})
        write_json(output / "test_predictions.json", predictions)
        (output / "report.md").write_text(report_text(manifest, evaluated, selections), encoding="utf-8")
        manifest.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat(), selections=selections)
        write_json(output / "analysis_manifest.json", manifest)
        return manifest
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        write_json(output / "analysis_manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="pilot")
    parser.add_argument("--output-dir", default="analysis/pilot")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        manifest = run_analysis(args.input_dir, args.output_dir, args.seed)
    except (ValueError, OSError, RuntimeError, ConvergenceWarning) as exc:
        parser.exit(1, f"Analysis failed: {exc}\n")
    print(f"Analysis complete: {len(manifest['items'])} items; report: {Path(args.output_dir) / 'report.md'}")


if __name__ == "__main__":
    main()
