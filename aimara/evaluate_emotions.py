"""Evaluate segment emotions against transcript-aligned first/third target turns.

The evaluator uses only the standard library once the AIMARA package is loaded,
and consults neither reference timing nor predicted speaker identity.
Run ``python -m aimara.evaluate_emotions --help`` for the CLI.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .emotion_alignment import align_video


DEFAULT_MAPPING = Path(__file__).with_name("emotion_circumplex.json")
REQUIRED_COLUMNS = {"video_file", "text", "emotion"}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load_mapping(path: Path = DEFAULT_MAPPING) -> dict:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if mapping["angles"].get("neutral", "missing") is not None:
        raise ValueError("Neutral must have no angle (JSON null)")
    for label, angle in mapping["angles"].items():
        if angle is not None and (
            not isinstance(angle, (int, float)) or not math.isfinite(angle)
        ):
            raise ValueError(f"Invalid angle for {label}: {angle}")
    for alias, canonical in mapping.get("aliases", {}).items():
        if canonical not in mapping["angles"]:
            raise ValueError(f"Alias {alias} refers to unknown emotion {canonical}")
    return mapping


def canonical_label(label: str, mapping: dict) -> str:
    key = label.strip().casefold()
    key = mapping.get("aliases", {}).get(key, key)
    if key not in mapping["angles"]:
        raise ValueError(f"Emotion {label!r} has no circumplex mapping")
    return key


def score_emotions(reference: str, prediction: str, mapping: dict) -> dict:
    """Return pure angular quantities and an explicitly neutral-aware score."""
    expected = canonical_label(reference, mapping)
    if not prediction.strip():
        return dict(
            ground_truth_label=expected,
            predicted_label="",
            ground_truth_angle=mapping["angles"][expected],
            predicted_angle=None,
            angular_error_deg=None,
            angular_score=None,
            score=0.0,
            score_type="missing_prediction",
            categorical_correct=0,
        )
    actual = canonical_label(prediction, mapping)
    first, second = mapping["angles"][expected], mapping["angles"][actual]
    correct = int(expected == actual)
    if first is None or second is None:
        error, angular = None, None
        score, kind = float(correct), "neutral_exact" if correct else "neutral_mismatch"
    else:
        difference = abs(first - second) % 360.0
        error = min(difference, 360.0 - difference)
        angular = score = 1.0 - error / 180.0
        kind = "angular"
    return dict(
        ground_truth_label=expected,
        predicted_label=actual,
        ground_truth_angle=first,
        predicted_angle=second,
        angular_error_deg=error,
        angular_score=angular,
        score=score,
        score_type=kind,
        categorical_correct=correct,
    )


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not REQUIRED_COLUMNS <= set(reader.fieldnames or []):
            raise ValueError(f"{path}: required columns {sorted(REQUIRED_COLUMNS)}")
        rows = list(reader)
    for row in rows:
        # Both Unix and Windows path strings are accepted, but matching is by
        # exact basename. Case remains significant to avoid accidental joins.
        row["video_file"] = row["video_file"].replace("\\", "/").rsplit("/", 1)[-1]
        if not row["video_file"]:
            raise ValueError(f"{path}: empty video filename")
    return rows


def grouped(rows: list[dict]) -> dict[str, list[dict]]:
    result = defaultdict(list)
    for row in rows:
        result[row["video_file"]].append(row)
    return dict(result)


def evaluate_model(
    reference: list[dict],
    prediction: list[dict],
    *,
    model: str,
    supported: set[str],
    common: set[str],
    mapping: dict,
    prediction_source: str = "",
) -> tuple[list[dict], list[dict], list[dict]]:
    references, predictions = grouped(reference), grouped(prediction)
    details, targets, unused = [], [], []
    for video, ref_rows in references.items():
        if len(ref_rows) != 3:
            raise ValueError(
                f"{video}: expected exactly three ground-truth utterances, got {len(ref_rows)}"
            )
        pred_rows = predictions.get(video, [])
        aligned = align_video(ref_rows, pred_rows)
        used = set()
        for match in aligned:
            turn = match["reference_turn"]
            if turn not in (1, 3):
                raise ValueError("Alignment returned a non-target ground-truth turn")
            segment = match["prediction_segment"]
            emotion = pred_rows[segment - 1]["emotion"] if segment is not None else ""
            if segment is not None:
                used.add(segment)
            scores = score_emotions(ref_rows[turn - 1]["emotion"], emotion, mapping)
            details.append(
                dict(
                    model=model,
                    video_file=video,
                    **match,
                    ground_truth_emotion=ref_rows[turn - 1]["emotion"],
                    predicted_emotion=emotion,
                    **scores,
                    eligible_supported=scores["ground_truth_label"] in supported,
                    eligible_common=scores["ground_truth_label"] in common,
                    prediction_source=prediction_source,
                )
            )
        for turn in (1, 3):
            parts = [
                r
                for r in details
                if r["video_file"] == video and r["reference_turn"] == turn
            ]
            if not parts:
                raise ValueError(f"Alignment omitted {video} target turn {turn}")
            n_reference = parts[0]["reference_tokens"]
            matched = sum(p["matched_tokens"] for p in parts)
            if matched > n_reference:
                raise ValueError("Alignment double-counted a reference token")

            def weighted(field):
                return (
                    sum(p["matched_tokens"] * p[field] for p in parts) / matched
                    if matched
                    else 0.0
                )

            directional = [p for p in parts if p["angular_score"] is not None]
            directional_tokens = sum(p["matched_tokens"] for p in directional)
            angular_score = (
                sum(p["matched_tokens"] * p["angular_score"] for p in directional)
                / directional_tokens
                if directional_tokens
                else None
            )
            ground_truth = canonical_label(ref_rows[turn - 1]["emotion"], mapping)
            coverage = matched / n_reference if n_reference else 0.0
            targets.append(
                dict(
                    model=model,
                    video_file=video,
                    reference_turn=turn,
                    ground_truth_emotion=ref_rows[turn - 1]["emotion"],
                    ground_truth_label=ground_truth,
                    eligible_supported=ground_truth in supported,
                    eligible_common=ground_truth in common,
                    reference_text=ref_rows[turn - 1]["text"],
                    reference_tokens=n_reference,
                    matched_tokens=matched,
                    text_coverage=coverage,
                    matched_segments=sum(
                        p["prediction_segment"] is not None for p in parts
                    ),
                    score=weighted("score"),
                    coverage_adjusted_score=weighted("score") * coverage,
                    categorical_accuracy=weighted("categorical_correct"),
                    angular_score=angular_score,
                    directional_prediction_fraction=directional_tokens / matched
                    if matched
                    else 0.0,
                    neutral_mismatch_fraction=(
                        sum(
                            p["matched_tokens"]
                            for p in parts
                            if p["score_type"] == "neutral_mismatch"
                        )
                        / matched
                        if matched
                        else 0.0
                    ),
                    alignment_status=";".join(
                        sorted({p["alignment_status"] for p in parts})
                    ),
                    prediction_spans_phone_response=any(
                        p.get("prediction_spans_phone_response", False) for p in parts
                    ),
                )
            )
        for i, row in enumerate(pred_rows, 1):
            if i not in used:
                unused.append(
                    dict(
                        model=model,
                        video_file=video,
                        prediction_segment=i,
                        prediction_text=row["text"],
                        predicted_emotion=row["emotion"],
                        reason="no_target_overlap_phone_or_extra_text",
                    )
                )
    for video in predictions.keys() - references.keys():
        for i, row in enumerate(predictions[video], 1):
            unused.append(
                dict(
                    model=model,
                    video_file=video,
                    prediction_segment=i,
                    prediction_text=row["text"],
                    predicted_emotion=row["emotion"],
                    reason="video_absent_from_ground_truth",
                )
            )
    return details, targets, unused


def summarize(targets: list[dict], model: str, scope: str) -> dict:
    eligible = [r for r in targets if r["model"] == model and r[f"eligible_{scope}"]]
    count = len(eligible)

    def mean(key):
        return sum(r[key] for r in eligible) / count if count else None

    per_emotion = defaultdict(list)
    for row in eligible:
        per_emotion[row["ground_truth_label"]].append(row["score"])
    angular = [r["angular_score"] for r in eligible if r["angular_score"] is not None]
    emotional = [r for r in eligible if r["ground_truth_label"] != "neutral"]
    return dict(
        model=model,
        scope=scope,
        eligible_videos=len({r["video_file"] for r in eligible}),
        eligible_targets=count,
        excluded_targets=sum(r["model"] == model for r in targets) - count,
        matched_targets=sum(r["matched_tokens"] > 0 for r in eligible),
        matched_segment_pairs=sum(r["matched_segments"] for r in eligible),
        mean_score=mean("score"),
        macro_emotion_score=(
            sum(sum(v) / len(v) for v in per_emotion.values()) / len(per_emotion)
            if per_emotion
            else None
        ),
        mean_coverage_adjusted_score=mean("coverage_adjusted_score"),
        categorical_accuracy=mean("categorical_accuracy"),
        mean_text_coverage=mean("text_coverage"),
        mean_angular_score_when_defined=sum(angular) / len(angular)
        if angular
        else None,
        targets_with_defined_angle=len(angular),
        emotional_target_score=(
            sum(r["score"] for r in emotional) / len(emotional) if emotional else None
        ),
        neutral_targets=sum(r["ground_truth_label"] == "neutral" for r in eligible),
        mean_neutral_mismatch_fraction=mean("neutral_mismatch_fraction"),
        mean_directional_prediction_fraction=mean("directional_prediction_fraction"),
        targets_needing_alignment_review=sum(
            "ambiguous" in r["alignment_status"] or "unmatched" in r["alignment_status"]
            for r in eligible
        ),
        merged_phone_targets=sum(
            r["prediction_spans_phone_response"] for r in eligible
        ),
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )


def named_paths(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or name in result:
            raise ValueError(f"Expected unique NAME=PATH, got {value!r}")
        result[name] = Path(path).resolve()
    return result


def run(args: argparse.Namespace) -> dict:
    mapping = load_mapping(args.mapping)
    paths = named_paths(args.prediction)
    inventories = {}
    for value in args.labels:
        name, separator, labels = value.partition("=")
        if not separator or name not in paths or name in inventories:
            raise ValueError(
                f"Expected one NAME=label,label inventory per prediction, got {value!r}"
            )
        inventories[name] = {
            canonical_label(label, mapping) for label in labels.split(",")
        }
    if set(inventories) != set(paths):
        raise ValueError(
            "Supply --labels for every model; never infer eligibility from its predicted labels"
        )
    common = set.intersection(*inventories.values())
    references = read_rows(args.reference)
    if not references:
        raise ValueError("Ground truth is empty")
    details, targets, unused, provenance = [], [], [], {}
    segmentation_fingerprints = {}
    for model, path in paths.items():
        manifest = path.with_name("run_manifest.json")
        run_manifest = json.loads(manifest.read_text()) if manifest.is_file() else None
        if run_manifest:
            if run_manifest.get("status") != "passed":
                raise ValueError(f"{model}: model run did not pass")
            recorded_inventory = {
                canonical_label(label, mapping)
                for label in run_manifest["label_inventory"]
            }
            if recorded_inventory != inventories[model]:
                raise ValueError(
                    f"{model}: declared inventory differs from run manifest"
                )
            if run_manifest.get("predictions_csv_sha256") != digest(path):
                raise ValueError(
                    f"{model}: prediction CSV differs from its run manifest"
                )
        predictions = read_rows(path)
        for row in predictions:
            if (
                row["emotion"].strip()
                and canonical_label(row["emotion"], mapping) not in inventories[model]
            ):
                raise ValueError(
                    f"{model}: prediction outside declared checkpoint inventory: {row['emotion']}"
                )
        pairs, turns, extra = evaluate_model(
            references,
            predictions,
            model=model,
            supported=inventories[model],
            common=common,
            mapping=mapping,
            prediction_source=str(path),
        )
        for record in pairs + turns:
            record.update(
                mapping_version=mapping["version"],
                dataset=run_manifest["dataset"] if run_manifest else "unrecorded",
                checkpoint=run_manifest.get("checkpoint", "") if run_manifest else "",
                checkpoint_sha256=run_manifest.get("checkpoint_sha256", "")
                if run_manifest
                else "",
            )
        details.extend(pairs)
        targets.extend(turns)
        unused.extend(extra)
        provenance[model] = dict(
            prediction_csv=str(path),
            prediction_sha256=digest(path),
            supported_canonical_labels=sorted(inventories[model]),
            run_manifest=run_manifest,
        )
        segmentation = [
            {
                key: row.get(key, "")
                for key in ("video_file", "start", "end", "speaker", "text")
            }
            for row in predictions
        ]
        segmentation_fingerprints[model] = hashlib.sha256(
            json.dumps(segmentation, sort_keys=True).encode()
        ).hexdigest()
    summary = [
        summarize(targets, model, scope)
        for model in paths
        for scope in ("supported", "common")
    ]
    by_emotion = []
    for model in paths:
        for emotion in sorted(
            {r["ground_truth_label"] for r in targets if r["model"] == model}
        ):
            selected = [r for r in targets if r["ground_truth_label"] == emotion]
            by_emotion.append(
                dict(emotion=emotion, **summarize(selected, model, "supported"))
            )
    report = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        reference_csv=str(args.reference.resolve()),
        reference_sha256=digest(args.reference),
        mapping=mapping,
        mapping_sha256=digest(args.mapping),
        source_hashes={
            name: digest(Path(__file__).with_name(name))
            for name in ("evaluate_emotions.py", "emotion_alignment.py")
        },
        methodology=dict(
            target_turns=[1, 3],
            alignment="monotonic normalized transcript tokens, all three ground-truth turns as anchors; video basename first; no timing or speaker matching",
            angular_score="1 - shortest angular distance in degrees / 180; defined only when both labels have angles",
            neutral_policy="score 1 for neutral-neutral, 0 for exactly one neutral; angular_error_deg/angular_score blank in both cases",
            aggregation="Each target: matched-token-weighted mean of pair scores. Unmatched targets score zero. Overall: equal weight per eligible target. Missing words reported as text coverage; optional coverage_adjusted_score penalizes them.",
            categorical_accuracy="Canonical-label correctness, weighted within each target by matched tokens; joy and happy are synonyms",
            macro_emotion_score="Equal weight per canonical emotion class; Happy.mp4 and Joy.mp4 belong to the same happy class",
            mean_angular_score_when_defined="Within each target, discard neutral/missing-angle portions and renormalize by directional matched-token count. Average those target means equally only across targets with at least one directional prediction. Even a small directional portion gives one target weight; report emotional_target_score and directional/neutral fractions alongside this conditional measure.",
            eligibility="Ground-truth label in declared checkpoint label inventory; common uses canonical inventory intersection",
            exclusions="Unsupported labels retained in pair/target sheets but excluded from supported/common summaries",
            uncertainty="Prototype angles are an engineering heuristic. Small synthetic sample, no confidence intervals or generalization claims.",
        ),
        common_labels=sorted(common),
        same_segmentation_across_models=len(set(segmentation_fingerprints.values()))
        == 1,
        segmentation_sha256=segmentation_fingerprints,
        provenance=provenance,
        summary=summary,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "reference.csv").write_bytes(args.reference.read_bytes())
    (args.output_dir / "circumplex_mapping.json").write_bytes(args.mapping.read_bytes())
    for name, rows in (
        ("matched_segments.csv", details),
        ("target_utterances.csv", targets),
        ("summary.csv", summary),
        ("by_emotion.csv", by_emotion),
        ("unused_predictions.csv", unused),
    ):
        write_csv(args.output_dir / name, rows)
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--prediction", action="append", required=True, metavar="NAME=CSV"
    )
    parser.add_argument(
        "--labels", action="append", required=True, metavar="NAME=LABEL,LABEL"
    )
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            dict(output_dir=str(args.output_dir.resolve()), summary=report["summary"]),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
