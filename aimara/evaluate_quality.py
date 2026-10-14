"""Score pretrained observations against annotated turns, without loading models."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

from scipy.optimize import linear_sum_assignment

from .batch import _atomic_json
from .batch_resume import file_digest
from .prepare_manifest import read_reference_csv


LABELS = ("happy", "sad", "neutral", "angry", "excited", "frustrated")
DEFAULT_MAPPING = {
    **{label: label for label in LABELS},
    "anger": "angry",
    "sadness": "sad",
}


def tokens(text: str) -> list[str]:
    """Casefold and remove punctuation, retaining internal apostrophes."""
    text = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    return re.findall(r"[^\W_]+(?:'[^\W_]+)*", text, flags=re.UNICODE)


def word_errors(reference: str, hypothesis: str) -> dict:
    left, right = tokens(reference), tokens(hypothesis)
    # Each cell stores distance and S/D/I counts. Ties prefer substitutions,
    # then deletions, then insertions; total WER is independent of this choice.
    previous = [(j, 0, 0, j) for j in range(len(right) + 1)]
    for i, word in enumerate(left, 1):
        current = [(i, 0, i, 0)]
        for j, predicted in enumerate(right, 1):
            distance, substitutions, deletions, insertions = previous[j - 1]
            mismatch = int(word != predicted)
            diagonal = (
                distance + mismatch,
                substitutions + mismatch,
                deletions,
                insertions,
            )
            d, s, removed, added = previous[j]
            deletion = (d + 1, s, removed + 1, added)
            d, s, removed, added = current[-1]
            insertion = (d + 1, s, removed, added + 1)
            current.append(
                min((diagonal, deletion, insertion), key=lambda cell: cell[0])
            )
        previous = current
    errors, substitutions, deletions, insertions = previous[-1]
    return dict(
        reference_words=len(left),
        hypothesis_words=len(right),
        errors=errors,
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        wer=errors / len(left) if left else None,
    )


def validate_intervals(intervals: list[dict]) -> None:
    previous_end = 0.0
    for interval in intervals:
        start, end = interval["start"], interval["end"]
        if not (
            math.isfinite(start)
            and math.isfinite(end)
            and 0 <= start < end
            and start >= previous_end - 1e-9
        ):
            raise ValueError(
                "Intervals must be finite, positive, ordered and nonoverlapping"
            )
        previous_end = end


def overlap(left, right) -> float:
    return max(0.0, min(left["end"], right["end"]) - max(left["start"], right["start"]))


def speaker_score(references: list[dict], predicted: list[dict]) -> dict:
    """Score annotated-turn duration, with optimal identity permutation per video.

    Annotation spans include pauses. This is deliberately not a standard DER
    benchmark: there is no word-level speech mask, overlap scoring or collar.
    """
    validate_intervals(references)
    validate_intervals(predicted)
    identities = sorted({r["speaker"] for r in references})
    voices = sorted({p["speaker"] for p in predicted} - {"UNKNOWN"})
    weights = [
        [
            sum(
                overlap(r, p)
                for r in references
                for p in predicted
                if r["speaker"] == identity and p["speaker"] == voice
            )
            for voice in voices
        ]
        for identity in identities
    ]
    mapping, correct = {}, 0.0
    if identities and voices:
        rows, columns = linear_sum_assignment(weights, maximize=True)
        for row, column in zip(rows, columns):
            mapping[voices[column]] = identities[row]
            correct += weights[row][column]
    total = sum(r["end"] - r["start"] for r in references)
    covered = sum(overlap(r, p) for r in references for p in predicted)
    unknown = sum(
        overlap(r, p)
        for r in references
        for p in predicted
        if p["speaker"] == "UNKNOWN"
    )
    return dict(
        reference_seconds=total,
        correct_seconds=correct,
        uncovered_seconds=max(0.0, total - covered),
        unknown_seconds=unknown,
        confusion_seconds=max(0.0, covered - unknown - correct),
        accuracy=correct / total if total else None,
        coverage=covered / total if total else None,
        identity_mapping=mapping,
    )


def boundary_score(
    references: list[dict], predicted: list[dict], tolerance: float
) -> dict:
    """One-to-one change-point matching; UNKNOWN changes count as predictions."""
    expected = [
        b["start"]
        for a, b in zip(references, references[1:])
        if a["speaker"] != b["speaker"]
    ]
    observed = [
        b["start"]
        for a, b in zip(predicted, predicted[1:])
        if a["speaker"] != b["speaker"]
    ]
    differences = []
    if expected and observed:
        # A cost larger than the sum of all valid distances prioritizes match
        # count over distance; unlike greedy matching, competing points are safe.
        penalty = (len(expected) + len(observed) + 1) * (tolerance + 1)
        costs = [
            [abs(a - b) if abs(a - b) <= tolerance else penalty for b in observed]
            for a in expected
        ]
        rows, columns = linear_sum_assignment(costs)
        differences = [
            costs[r][c] for r, c in zip(rows, columns) if costs[r][c] <= tolerance
        ]
    matched = len(differences)
    return dict(
        reference_boundaries=len(expected),
        predicted_boundaries=len(observed),
        matched=matched,
        precision=matched / len(observed) if observed else None,
        recall=matched / len(expected) if expected else None,
        absolute_error_seconds=differences,
    )


def classification(rows: list[dict]) -> dict:
    scored = [r for r in rows if r["target"] is not None]
    supported = sorted({r["target"] for r in scored})
    confusion = {
        label: dict(
            Counter(
                r["predicted"] or "NO_PREDICTION"
                for r in scored
                if r["target"] == label
            )
        )
        for label in supported
    }
    per_class = {}
    for label in supported:
        tp = sum(r["target"] == label and r["predicted"] == label for r in scored)
        fp = sum(r["target"] != label and r["predicted"] == label for r in scored)
        fn = sum(r["target"] == label and r["predicted"] != label for r in scored)
        per_class[label] = dict(
            support=tp + fn, f1=2 * tp / (2 * tp + fp + fn), recall=tp / (tp + fn)
        )
    correct = sum(r["target"] == r["predicted"] for r in scored)
    return dict(
        scored_turns=len(scored),
        correct_turns=correct,
        accuracy=correct / len(scored) if scored else None,
        macro_f1=sum(p["f1"] for p in per_class.values()) / len(supported)
        if supported
        else None,
        macro_f1_classes=supported,
        per_class=per_class,
        confusion=confusion,
        excluded_labels=dict(
            Counter(r["reference_emotion"] for r in rows if r["target"] is None)
        ),
        missing_predictions=sum(r["predicted"] is None for r in scored),
    )


def evaluate_video(video: dict, references, mapping: dict, tolerance: float) -> dict:
    reference = [dict(start=r.start, end=r.end, speaker=r.speaker) for r in references]
    predictions = video["predictions"]
    exported = [
        dict(start=p["start_time"], end=p["end_time"], speaker=p["speaker_id"])
        for p in predictions
    ]
    validate_intervals(reference)
    validate_intervals(exported)
    timeline = [
        dict(
            start=p["start_sample"] / 16000,
            end=p["end_sample"] / 16000,
            speaker=p["speaker_id"],
        )
        for p in video["speaker_timeline"]
    ]
    validate_intervals(timeline)
    for prediction in predictions:
        scores = prediction["probabilities"]
        if (
            set(scores) != set(LABELS)
            or any(not math.isfinite(p) or not 0 <= p <= 1 for p in scores.values())
            or not math.isclose(sum(scores.values()), 1, rel_tol=0, abs_tol=1e-6)
        ):
            raise ValueError("Invalid emotion probabilities")
    text = " ".join(r.text for r in references)
    emotion_rows = []
    for r, interval in zip(references, reference):
        weights = [overlap(interval, segment) for segment in exported]
        duration = sum(weights)
        scores = (
            {
                label: sum(
                    weight * prediction["probabilities"][label]
                    for weight, prediction in zip(weights, predictions)
                )
                / duration
                for label in LABELS
            }
            if duration
            else {}
        )
        emotion_rows.append(
            dict(
                source_row=r.source_row,
                start=r.start,
                end=r.end,
                reference_emotion=r.emotion,
                target=mapping.get(r.emotion),
                predicted=max(scores, key=scores.get) if scores else None,
                covered_fraction=duration / (r.end - r.start),
                probabilities=scores,
            )
        )
    return dict(
        video_file=video["video_file"],
        source_sha256=video["source_sha256"],
        asr=word_errors(text, " ".join(c["text"] for c in video["asr_calls"])),
        exported_transcript=word_errors(
            text, " ".join(p["transcript"] for p in predictions)
        ),
        timeline_speakers=speaker_score(reference, timeline),
        exported_speakers=speaker_score(reference, exported),
        timeline_boundaries=boundary_score(reference, timeline, tolerance),
        exported_boundaries=boundary_score(reference, exported, tolerance),
        emotion=classification(emotion_rows),
        emotion_turns=emotion_rows,
        segment_count=len(predictions),
    )


def aggregate(videos: list[dict]) -> dict:
    result = dict(
        video_count=len(videos), segment_count=sum(v["segment_count"] for v in videos)
    )
    for kind in ("asr", "exported_transcript"):
        counts = {
            key: sum(v[kind][key] for v in videos)
            for key in (
                "reference_words",
                "hypothesis_words",
                "errors",
                "substitutions",
                "deletions",
                "insertions",
            )
        }
        counts["wer"] = (
            counts["errors"] / counts["reference_words"]
            if counts["reference_words"]
            else None
        )
        result[kind] = counts
    for kind in ("timeline_speakers", "exported_speakers"):
        counts = {
            key: sum(v[kind][key] for v in videos)
            for key in (
                "reference_seconds",
                "correct_seconds",
                "uncovered_seconds",
                "unknown_seconds",
                "confusion_seconds",
            )
        }
        counts["accuracy"] = (
            counts["correct_seconds"] / counts["reference_seconds"]
            if counts["reference_seconds"]
            else None
        )
        result[kind] = counts
    for kind in ("timeline_boundaries", "exported_boundaries"):
        counts = {
            key: sum(v[kind][key] for v in videos)
            for key in ("reference_boundaries", "predicted_boundaries", "matched")
        }
        errors = [e for v in videos for e in v[kind]["absolute_error_seconds"]]
        counts.update(
            precision=counts["matched"] / counts["predicted_boundaries"]
            if counts["predicted_boundaries"]
            else None,
            recall=counts["matched"] / counts["reference_boundaries"]
            if counts["reference_boundaries"]
            else None,
            mean_absolute_error_seconds=sum(errors) / len(errors) if errors else None,
        )
        result[kind] = counts
    result["emotion"] = classification([r for v in videos for r in v["emotion_turns"]])
    return result


def evaluate_reports(
    paths: list[Path], reference_csv: Path, mapping: dict, tolerance: float
) -> dict:
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Boundary tolerance must be finite and nonnegative")
    if not isinstance(mapping, dict) or any(
        label not in LABELS for label in mapping.values()
    ):
        raise ValueError(f"Emotion mappings must target one of {LABELS}")
    reference_hash = file_digest(reference_csv)
    references = read_reference_csv(reference_csv)
    runs, names, source_hashes = [], set(), {}
    for path in paths:
        report = json.loads(path.read_text())
        name = path.parent.name
        if name in names:
            raise ValueError(f"Duplicate configuration name: {name}")
        names.add(name)
        if (
            not report.get("passed")
            or report.get("status") != "passed"
            or report.get("failures")
        ):
            raise ValueError(f"Incomplete or failed smoke report: {path}")
        if report["reference_sha256"] != reference_hash:
            raise ValueError(f"Reference checksum mismatch: {path}")
        names_in_report = [v["video_file"] for v in report["videos"]]
        if (
            not names_in_report
            or len(set(names_in_report)) != len(names_in_report)
            or set(names_in_report) != set(report["selected_videos"])
        ):
            raise ValueError(f"Incomplete or duplicate selected videos: {path}")
        videos = []
        for video in report["videos"]:
            if video["status"] != "passed":
                raise ValueError(f"Failed video: {video['video_file']}")
            name_in_video = video["video_file"]
            digest = video["source_sha256"]
            if (
                name_in_video in source_hashes
                and digest != source_hashes[name_in_video]
            ):
                raise ValueError(f"Source checksum mismatch for {name_in_video}")
            source_hashes[name_in_video] = digest
            selected = sorted(
                (r for r in references if r.video_file == name_in_video),
                key=lambda r: r.start,
            )
            if not selected:
                raise ValueError(f"Missing annotations for {name_in_video}")
            videos.append(evaluate_video(video, selected, mapping, tolerance))
        runs.append(
            dict(
                name=name,
                source_report_sha256=file_digest(path),
                configuration=report["configuration"],
                versions=report["versions"],
                videos=videos,
                summary=aggregate(videos),
            )
        )
    if not runs:
        raise ValueError("No reports supplied")
    comparisons = []
    for left, right in itertools.combinations(runs, 2):
        common = sorted(
            {v["video_file"] for v in left["videos"]}
            & {v["video_file"] for v in right["videos"]}
        )
        if common:
            comparisons.append(
                dict(
                    left=left["name"],
                    right=right["name"],
                    common_videos=common,
                    left_summary=aggregate(
                        [v for v in left["videos"] if v["video_file"] in common]
                    ),
                    right_summary=aggregate(
                        [v for v in right["videos"] if v["video_file"] in common]
                    ),
                )
            )
    return dict(
        schema_version=1,
        validation_kind="annotated_turn_quality",
        reference_sha256=reference_hash,
        emotion_mapping=mapping,
        boundary_tolerance_seconds=tolerance,
        methods=dict(
            asr="Micro WER over chronological raw ASR calls; Unicode casefold, punctuation removal, internal apostrophes retained",
            speakers="Optimal one-to-one identity mapping per video; duration on annotated turn spans, including pauses; UNKNOWN never mapped; not standard DER",
            boundaries="One-to-one speaker change matching; next interval start; UNKNOWN changes included; no word-level timestamp ground truth",
            emotion="Duration-weighted mean segment probabilities per annotated turn; uncovered turns are errors; unmapped labels excluded; macro F1 over reference-supported classes",
            comparisons="Only shared videos, verified source and reference checksums; observational ablations on a small set, not held-out tuning",
        ),
        runs=runs,
        paired_comparisons=comparisons,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-report", type=Path, action="append", required=True)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--emotion-map", type=Path, help="JSON mapping replacing the default aliases"
    )
    parser.add_argument("--boundary-tolerance", type=float, default=0.5)
    args = parser.parse_args()
    mapping = (
        json.loads(args.emotion_map.read_text())
        if args.emotion_map
        else DEFAULT_MAPPING
    )
    report = evaluate_reports(
        args.smoke_report, args.reference_csv, mapping, args.boundary_tolerance
    )
    _atomic_json(args.report, report)
    for run in report["runs"]:
        metrics = run["summary"]
        accuracy = metrics["emotion"]["accuracy"]
        emotion = f"{accuracy:.1%}" if accuracy is not None else "no matching labels"
        wer = metrics["asr"]["wer"]
        word_error_rate = f"{wer:.1%}" if wer is not None else "no reference words"
        print(
            f"{run['name']}: {metrics['video_count']} videos; WER {word_error_rate}; emotion accuracy {emotion}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
