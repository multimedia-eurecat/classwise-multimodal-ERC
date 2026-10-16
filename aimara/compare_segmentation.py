"""Export saved segmentation and trace boundary losses without rerunning models."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .batch import _atomic_json
from .batch_resume import file_digest
from .diarization import SpeakerInterval, speaker_for_region
from .evaluate_quality import speaker_score, tokens
from .prepare_manifest import read_reference_csv
from .segmentation import SpeechRegion
from .speaker_segmentation import SpeakerSegmentationConfig, SpeakerSegmenter
from .speech_interfaces import Transcript, TranscriptSegment, TranscriptWord


RATE = 16000
REFERENCE_FIELDS = ["video_file", "start", "end", "speaker", "text", "emotion"]
NUMBER_ALIASES = dict(
    zip(
        (
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
        ),
        map(str, range(11)),
    )
)


def align(
    reference: list[str], hypothesis: list[str]
) -> list[tuple[int | None, int | None, str]]:
    """Global edit alignment. Matching text locates turns, not ground-truth times."""
    costs = [list(range(len(hypothesis) + 1))]
    for i, word in enumerate(reference, 1):
        row = [i]
        for j, other in enumerate(hypothesis, 1):
            row.append(
                min(costs[-1][j - 1] + (word != other), costs[-1][j] + 1, row[-1] + 1)
            )
        costs.append(row)
    i, j, result = len(reference), len(hypothesis), []
    while i or j:
        if (
            i
            and j
            and costs[i][j]
            == costs[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1])
        ):
            result.append(
                (
                    i - 1,
                    j - 1,
                    "exact"
                    if reference[i - 1] == hypothesis[j - 1]
                    else "substitution",
                )
            )
            i, j = i - 1, j - 1
        elif i and costs[i][j] == costs[i - 1][j] + 1:
            result.append((i - 1, None, "deletion"))
            i -= 1
        else:
            result.append((None, j - 1, "insertion"))
            j -= 1
    return list(reversed(result))


def align_reference(reference, hypothesis):
    """Treat number spelling as equivalent only for structural text alignment.

    This prevents equally cheap edit paths from shifting '4' into the next turn
    when the reference says 'four. All right' and ASR says '4. Alright'. It does
    not change the separately recorded WER or the original CSV/transcript text.
    """
    result = align(
        [NUMBER_ALIASES.get(t, t) for t in reference],
        [NUMBER_ALIASES.get(t, t) for t in hypothesis],
    )
    return [
        (
            r,
            h,
            "numeric_equivalent"
            if status == "exact" and reference[r] != hypothesis[h]
            else status,
        )
        for r, h, status in result
    ]


def regions_for_calls(video: dict):
    regions = sorted(
        set(
            (p["asr_input_start_time"], p["asr_input_end_time"])
            for p in video["predictions"]
        )
    )
    if len(regions) != len(video["asr_calls"]):
        raise ValueError(
            "Cannot associate every saved ASR call with an unambiguous source input interval"
        )
    for region, call in zip(regions, video["asr_calls"]):
        if not math.isclose(
            region[1] - region[0], call["input_duration"], abs_tol=1 / RATE
        ):
            raise ValueError("ASR input duration does not match saved source interval")
    return regions


def restore_transcript(call: dict) -> Transcript:
    return Transcript(
        call["text"],
        call.get("language"),
        tuple(
            TranscriptSegment(
                s["start_time"],
                s["end_time"],
                s["text"],
                tuple(TranscriptWord(**w) for w in s["words"]),
            )
            for s in call["segments"]
        ),
    )


class MajoritySentences(SpeakerSegmenter):
    """Diagnostic only: remove the credible-trailing-speaker override."""

    def _credible(self, spans, sample_rate):
        return False


def replay(video, regions, segmenter, timeline):
    rows = []
    for (start, end), call in zip(regions, video["asr_calls"]):
        groups = segmenter.split(
            SpeechRegion(round(start * RATE), round(end * RATE)),
            restore_transcript(call),
            RATE,
            timeline,
        )
        for group in groups:
            rows.append(
                dict(
                    segment_id=len(rows),
                    start_time=group.region.start_time(RATE),
                    end_time=group.region.end_time(RATE),
                    speaker_id=group.speaker_id,
                    transcript=group.text,
                )
            )
    return rows


def export_owners(words, groups):
    exported, owners = [], []
    for group in groups:
        values = tokens(group["transcript"])
        exported.extend(values)
        owners.extend([group["segment_id"]] * len(values))
    expected = [w["token"] for w in words]
    if exported != expected:
        raise ValueError(
            "Grouping changed normalized ASR text; cannot trace exact word ownership"
        )
    return owners


def diagnose_boundary(left, right, owner_sets):
    if left is None or right is None:
        return "insufficient_aligned_words"
    if owner_sets["actual"][0] != owner_sets["actual"][1]:
        if left.get("actual_speaker") == right.get("actual_speaker") and left.get(
            "actual_speaker"
        ) not in (None, "UNKNOWN"):
            return "groups_split_but_final_speaker_identity_merged"
        return (
            "split_exists_identity_unresolved"
            if "UNKNOWN"
            in (
                left.get("actual_speaker", left["raw_speaker"]),
                right.get("actual_speaker", right["raw_speaker"]),
            )
            else "split_exists"
        )
    if owner_sets["majority_only"][0] != owner_sets["majority_only"][1]:
        return "grouping_trailing_override_removes_split"
    if left["raw_speaker"] == right["raw_speaker"] and left["raw_speaker"] != "UNKNOWN":
        return "raw_diarization_merges_boundary_words"
    if "UNKNOWN" in (left["raw_speaker"], right["raw_speaker"]):
        return "raw_diarization_unresolved_at_boundary"
    if owner_sets["raw_unsmoothed"][0] != owner_sets["raw_unsmoothed"][1]:
        return "sentence_grouping_or_smoothing_removes_split"
    return "grouping_requires_review"


def audit_video(video, references, config):
    regions = regions_for_calls(video)
    timeline = [
        SpeakerInterval(p["start_sample"], p["end_sample"], p["speaker_id"])
        for p in video["speaker_timeline"]
    ]
    mapping = speaker_score(
        [dict(start=r.start, end=r.end, speaker=r.speaker) for r in references],
        [
            dict(
                start=p.start_sample / RATE,
                end=p.end_sample / RATE,
                speaker=p.speaker_id,
            )
            for p in timeline
        ],
    )["identity_mapping"]
    variants = {
        "actual": replay(video, regions, SpeakerSegmenter(config), timeline),
        "majority_only": replay(video, regions, MajoritySentences(config), timeline),
        "raw_unsmoothed": replay(
            video,
            regions,
            SpeakerSegmenter(SpeakerSegmentationConfig(False, 1, 0)),
            timeline,
        ),
        "reference_timeline": replay(
            video,
            regions,
            SpeakerSegmenter(config),
            [
                SpeakerInterval(
                    round(r.start * RATE), round(r.end * RATE), "ref" + r.speaker
                )
                for r in references
            ],
        ),
    }
    recorded = [
        {
            key: p[key]
            for key in (
                "segment_id",
                "start_time",
                "end_time",
                "speaker_id",
                "transcript",
            )
        }
        for p in video["predictions"]
    ]
    if variants["actual"] != recorded:
        raise ValueError(
            f"Actual grouping replay does not reproduce recorded output for {video['video_file']}"
        )
    words, stages = [], []
    for call_index, ((start, end), call) in enumerate(zip(regions, video["asr_calls"])):
        stages.append(
            dict(
                stage="vad_asr_input",
                stage_id=call_index,
                start=start,
                end=end,
                speaker="",
                text=call["text"],
            )
        )
        for segment_index, segment in enumerate(call["segments"]):
            stages.append(
                dict(
                    stage="asr_backend_segment",
                    stage_id=f"{call_index}:{segment_index}",
                    start=start + segment["start_time"],
                    end=min(end, start + segment["end_time"]),
                    speaker="",
                    text=segment["text"],
                )
            )
            for word_index, word in enumerate(segment["words"]):
                absolute_start = min(end, start + word["start_time"])
                absolute_end = min(end, start + word["end_time"])
                point = min(round(absolute_start * RATE), round(end * RATE) - 1)
                speaker = speaker_for_region(
                    SpeechRegion(point, max(point + 1, round(absolute_end * RATE))),
                    timeline,
                )
                for token in tokens(word["text"]):
                    words.append(
                        dict(
                            token=token,
                            word_text=word["text"].strip(),
                            start=absolute_start,
                            end=absolute_end,
                            call_index=call_index,
                            asr_segment_index=segment_index,
                            word_index=word_index,
                            raw_speaker=speaker,
                            reference_turn=None,
                            reference_token="",
                            alignment="insertion",
                        )
                    )
    # The exact text path is required; a partial backend alignment must be reviewed,
    # rather than silently inventing timings for words absent from the observations.
    if [t for call in video["asr_calls"] for t in tokens(call["text"])] != [
        w["token"] for w in words
    ]:
        raise ValueError(
            "Saved word observations do not cover the complete ASR transcript"
        )
    ownership = {
        name: export_owners(words, groups) for name, groups in variants.items()
    }
    ref_tokens, ref_owners = [], []
    for index, reference in enumerate(references):
        values = tokens(reference.text)
        ref_tokens.extend(values)
        ref_owners.extend([index] * len(values))
    alignment = align_reference(ref_tokens, [w["token"] for w in words])
    for ref_index, hyp_index, status in alignment:
        if hyp_index is not None:
            words[hyp_index]["alignment"] = status
            if ref_index is not None:
                words[hyp_index]["reference_turn"] = ref_owners[ref_index]
                words[hyp_index]["reference_token"] = ref_tokens[ref_index]
    for index, word in enumerate(words):
        for name, owners in ownership.items():
            word[name + "_group"] = owners[index]
        word["raw_evaluation_mapped_reference_speaker"] = mapping.get(
            word["raw_speaker"], "UNKNOWN"
        )
        word["actual_speaker"] = video["predictions"][word["actual_group"]][
            "speaker_id"
        ]
    for index, interval in enumerate(video["speaker_timeline"]):
        stages.append(
            dict(
                stage="raw_diarization",
                stage_id=index,
                start=interval["start_sample"] / RATE,
                end=interval["end_sample"] / RATE,
                speaker=interval["speaker_id"],
                text="",
            )
        )
    for p in video["predictions"]:
        stages.append(
            dict(
                stage="final_erc_group",
                stage_id=p["segment_id"],
                start=p["start_time"],
                end=p["end_time"],
                speaker=p["speaker_id"],
                text=p["transcript"],
            )
        )
    turns = []
    for index, reference in enumerate(references):
        aligned = [w for w in words if w["reference_turn"] == index]
        ids = sorted({w["actual_group"] for w in aligned})
        groups = [video["predictions"][i] for i in ids]
        turns.append(
            dict(
                reference_turn=index,
                reference_start=reference.start,
                reference_end=reference.end,
                reference_speaker=reference.speaker,
                reference_text=reference.text,
                reference_emotion=reference.emotion,
                reference_token_count=len(tokens(reference.text)),
                aligned_asr_tokens=len(aligned),
                exact_asr_tokens=sum(w["alignment"] == "exact" for w in aligned),
                numeric_equivalent_asr_tokens=sum(
                    w["alignment"] == "numeric_equivalent" for w in aligned
                ),
                asr_word_start=aligned[0]["start"] if aligned else None,
                asr_word_end=aligned[-1]["end"] if aligned else None,
                actual_group_ids=ids,
                actual_group_intervals=[
                    [p["start_time"], p["end_time"]] for p in groups
                ],
                actual_group_speakers=[p["speaker_id"] for p in groups],
                evaluation_mapped_reference_speakers=[
                    mapping.get(p["speaker_id"], "UNKNOWN") for p in groups
                ],
                actual_group_texts=[p["transcript"] for p in groups],
                reference_minus_vad_coverage_seconds=(reference.end - reference.start)
                - sum(
                    max(0, min(reference.end, end) - max(reference.start, start))
                    for start, end in regions
                ),
            )
        )
    boundaries = []
    for index, (before, after) in enumerate(zip(references, references[1:])):
        if before.speaker == after.speaker:
            continue
        li = [i for i, w in enumerate(words) if w["reference_turn"] == index]
        ri = [i for i, w in enumerate(words) if w["reference_turn"] == index + 1]
        left, right = (words[li[-1]] if li else None), (words[ri[0]] if ri else None)
        owners = (
            {
                name: (values[li[-1]], values[ri[0]])
                for name, values in ownership.items()
            }
            if left and right
            else {}
        )
        row = dict(
            reference_boundary=index,
            reference_time=after.start,
            previous_reference_end=before.end,
            reference_before_speaker=before.speaker,
            reference_after_speaker=after.speaker,
            reference_before_text=before.text,
            reference_after_text=after.text,
            diagnosis=diagnose_boundary(left, right, owners),
            edge_alignment="exact"
            if left and right and left["alignment"] == right["alignment"] == "exact"
            else "review_lexical_alignment",
            left_asr_word=left["word_text"] if left else None,
            right_asr_word=right["word_text"] if right else None,
            left_word_end=left["end"] if left else None,
            right_word_start=right["start"] if right else None,
            raw_before_speaker=left["raw_speaker"] if left else None,
            raw_after_speaker=right["raw_speaker"] if right else None,
            final_before_speaker=left["actual_speaker"] if left else None,
            final_after_speaker=right["actual_speaker"] if right else None,
        )
        if left and right:
            row.update(
                asr_estimated_gap_seconds=right["start"] - left["end"],
                reference_distance_from_asr_gap_seconds=max(
                    0,
                    min(left["end"], right["start"]) - after.start,
                    after.start - max(left["end"], right["start"]),
                ),
                asr_backend_segments_separate=(
                    left["call_index"],
                    left["asr_segment_index"],
                )
                != (right["call_index"], right["asr_segment_index"]),
            )
            for name, (a, b) in owners.items():
                row[name + "_separates"] = a != b
            if owners["actual"][0] != owners["actual"][1]:
                a, b = [video["predictions"][i] for i in owners["actual"]]
                midpoint = (a["end_time"] + b["start_time"]) / 2
                row.update(
                    actual_gap_or_cut_midpoint=midpoint,
                    midpoint_minus_reference_seconds=midpoint - after.start,
                    reference_inside_output_gap=a["end_time"]
                    <= after.start
                    <= b["start_time"],
                )
        boundaries.append(row)
    return dict(
        words=words,
        stages=stages,
        turns=turns,
        boundaries=boundaries,
        variants=variants,
        evaluation_identity_mapping=mapping,
        summary=dict(
            reference_turns=len(references),
            final_groups=len(video["predictions"]),
            vad_inputs=len(regions),
            asr_backend_segments=sum(len(c["segments"]) for c in video["asr_calls"]),
            reference_speaker_changes=len(boundaries),
            changes_separated_by_final_groups=sum(
                bool(b.get("actual_separates")) for b in boundaries
            ),
            changes_separated_by_asr_backend_segments=sum(
                bool(b.get("asr_backend_segments_separate")) for b in boundaries
            ),
            majority_only_changes_separated=sum(
                bool(b.get("majority_only_separates")) for b in boundaries
            ),
            reference_timeline_changes_separated=sum(
                bool(b.get("reference_timeline_separates")) for b in boundaries
            ),
        ),
    )


def write_csv(path: Path, rows, fields=None):
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--original-reference-csv", type=Path)
    parser.add_argument("--smoke-report", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    references = read_reference_csv(args.reference_csv)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.reference_csv, output / "reference_current.csv")
    results, sources, originals = [], [], {}
    if args.original_reference_csv:
        shutil.copy2(args.original_reference_csv, output / "reference_original.csv")
        for row in read_reference_csv(args.original_reference_csv):
            originals.setdefault(row.video_file, []).append(asdict(row))
    combined, words, stages, turns, boundaries, variants, summaries = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for path in args.smoke_report:
        report = json.loads(path.read_text())
        if not report.get("passed"):
            raise ValueError(f"Cannot audit failed/incomplete report: {path}")
        options = report["requested_options"]
        name = Path(options.get("output_dir", str(path.parent))).name
        if (
            options["diarization_backend"] == "none"
            or not options["snap_splits_to_sentences"]
        ):
            raise ValueError(
                "This diagnostic expects the saved sentence-mode diarization configurations"
            )
        config = SpeakerSegmentationConfig(
            options["snap_splits_to_sentences"],
            options["min_speaker_run_tokens"],
            options["min_speaker_run_duration"],
        )
        rows = []
        for video in report["videos"]:
            selected = sorted(
                (r for r in references if r.video_file == video["video_file"]),
                key=lambda r: r.start,
            )
            if not selected:
                raise ValueError(f"No current reference for {video['video_file']}")
            audit = audit_video(video, selected, config)
            prefix = dict(system=name, video_file=video["video_file"])
            for p in video["predictions"]:
                row = dict(
                    video_file=video["video_file"],
                    start=p["start_time"],
                    end=p["end_time"],
                    speaker=p["speaker_id"],
                    text=p["transcript"],
                    emotion=p["emotion"],
                )
                rows.append(row)
                combined.append(
                    dict(
                        system=name,
                        **row,
                        segment_id=p["segment_id"],
                        emotion_source="model_prediction",
                    )
                )
            for target, key in (
                (words, "words"),
                (stages, "stages"),
                (turns, "turns"),
                (boundaries, "boundaries"),
            ):
                target.extend(dict(**prefix, **row) for row in audit[key])
            for variant, groups in audit["variants"].items():
                variants.extend(
                    dict(
                        **prefix,
                        variant=variant,
                        **row,
                        emotion="",
                        emotion_source="not_recomputed",
                    )
                    for row in groups
                )
            summaries.append(dict(**prefix, **audit["summary"]))
            results.append(dict(**prefix, **audit))
        write_csv(
            output / f"new_pipeline_{options['asr_backend']}.csv",
            rows,
            REFERENCE_FIELDS,
        )
        sources.append(
            dict(
                system=name,
                source_report=str(path.resolve()),
                sha256=file_digest(path),
                measured_reference_sha256=report["reference_sha256"],
                runner_sha256=report["runner_sha256"],
                actual_replay_matches_saved_output=True,
            )
        )
    for filename, rows in (
        ("all_pipeline_segments.csv", combined),
        ("word_trace.csv", words),
        ("segmentation_stages.csv", stages),
        ("comparison_by_reference_turn.csv", turns),
        ("boundary_diagnosis.csv", boundaries),
        ("diagnostic_replays.csv", variants),
        ("summary.csv", summaries),
    ):
        write_csv(output / filename, rows)
    changes = []
    for video in sorted({r.video_file for r in references}):
        current = [asdict(r) for r in references if r.video_file == video]
        old = originals.get(video, [])

        def clean(values):
            return [{k: v for k, v in r.items() if k != "source_row"} for r in values]

        if originals and clean(current) != clean(old):
            changes.append(dict(video_file=video, original=old, current=current))
    report = dict(
        schema_version=1,
        generated_at=datetime.now(timezone.utc).isoformat(),
        diagnostic_source_sha256=file_digest(Path(__file__)),
        grouping_source_sha256=file_digest(
            Path(__file__).with_name("speaker_segmentation.py")
        ),
        reference_path=str(args.reference_csv.resolve()),
        reference_sha256=file_digest(args.reference_csv),
        reference_turns=len(references),
        original_reference_turns=sum(len(v) for v in originals.values()) or None,
        changed_reference_videos=changes,
        sources=sources,
        summary=summaries,
        boundaries=boundaries,
        methods=dict(
            reference="Current weak reference; transcript/timing provenance unverified; back-to-back intervals include pauses",
            alignment="Global normalized token edit alignment locates reference text in ASR observations; English zero..ten and digits are treated as equivalent only for this structural comparison, without changing WER; estimated word times are not independent ground truth",
            boundary="A reference text-turn boundary is retained when its neighboring aligned words belong to different final groups; gaps do not imply missing speech",
            replay="Actual groups must reproduce the saved labels, text and source bounds exactly; diagnostic variants do not recompute ASR, encoders or emotions",
            majority_only="Same recorded words/timeline and sentence mode, with only credible-trailing-speaker override disabled",
            raw_unsmoothed="Same words/timeline; raw word grouping with minimum run 1 token / 0 seconds",
            reference_timeline="Counterfactual using annotation speaker spans instead of diarization, retaining recorded ASR words/times; not an independent model result",
        ),
    )
    _atomic_json(output / "diagnosis.json", report)
    write_html(output / "index.html", report, results)
    print(f"Wrote {output}")
    for row in summaries:
        print(
            row["system"],
            row["video_file"],
            f"groups={row['final_groups']}; text-turn changes retained={row['changes_separated_by_final_groups']}/{row['reference_speaker_changes']}; majority-only={row['majority_only_changes_separated']}",
        )
    return 0


def write_html(path, report, results):
    """Static diagnostic tables remain usable through the existing review server."""

    def esc(value):
        return html.escape(str(value))

    content = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Segmentation diagnosis</title><style>body{font-family:system-ui;color:#193149;background:#f6f8fc;margin:30px}p{line-height:1.6;max-width:1100px}table{border-collapse:collapse;background:white;width:100%;margin:20px 0}td,th{text-align:left;padding:10px;border:1px solid #dce4ed;font-size:13px}th{background:#193149;color:white}a{color:#08788a}details{background:white;padding:18px;margin:15px 0}summary{cursor:pointer;font-weight:bold}.wrap{overflow:auto}video{width:480px;max-width:100%}.bad{background:#fff0d8}.links a{display:inline-block;margin:6px 16px 6px 0}</style></head><body>",
        "<h1>Segmentation comparison and stage diagnosis</h1><p><a href='../index.html'>Back to playable segmentation viewer</a></p>",
        f"<p>Current reference: {report['reference_turns']} turns. Earlier evaluation reference: {report['original_reference_turns']} turns. Changed video: {', '.join(c['video_file'] for c in report['changed_reference_videos']) or 'none'}.</p>",
        "<p>Reference intervals include pauses and have unverified timing provenance. These tables track whether reference text turns are kept separate; a pause excluded by VAD is not automatically missed speech. Word alignment is lexical and word times are ASR estimates. The diagnostic replays reuse saved observations; they do not rerun any neural model or calculate new emotion scores.</p>",
        "<div class='links'>",
    ]
    for filename in (
        "new_pipeline_whisper.csv",
        "new_pipeline_parakeet.csv",
        "comparison_by_reference_turn.csv",
        "boundary_diagnosis.csv",
        "word_trace.csv",
        "segmentation_stages.csv",
        "diagnostic_replays.csv",
        "summary.csv",
        "reference_current.csv",
        "reference_original.csv",
        "diagnosis.json",
    ):
        if (path.parent / filename).exists():
            content.append(f"<a href='{filename}'>{filename}</a>")
    content.append(
        "</div><h2>Per-video summary</h2><div class='wrap'><table><tr><th>System</th><th>Video</th><th>Reference turns</th><th>VAD inputs</th><th>ASR backend segments</th><th>Final groups</th><th>Reference text changes retained</th><th>Without trailing override</th><th>With reference speaker timeline</th></tr>"
    )
    for r in report["summary"]:
        values = [
            r[k]
            for k in (
                "system",
                "video_file",
                "reference_turns",
                "vad_inputs",
                "asr_backend_segments",
                "final_groups",
            )
        ]
        values += [
            f"{r[k]}/{r['reference_speaker_changes']}"
            for k in (
                "changes_separated_by_final_groups",
                "majority_only_changes_separated",
                "reference_timeline_changes_separated",
            )
        ]
        content.append("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in values) + "</tr>")
    content.append("</table></div><h2>Boundary evidence</h2>")
    for video in sorted({r["video_file"] for r in results}):
        content.append(
            f"<details open><summary>{esc(video)}</summary><video controls preload='none' src='../assets/videos/{esc(video)}'></video>"
        )
        for audit in (r for r in results if r["video_file"] == video):
            content.append(
                f"<h3>{esc(audit['system'])}</h3><p>Evaluation-only identity mapping for the current reference: {esc(audit['evaluation_identity_mapping'])}</p><div class='wrap'><table><tr><th>Reference time</th><th>Neighboring ASR words / estimated times</th><th>Raw speaker labels</th><th>Final speaker labels</th><th>ASR segments separate?</th><th>Final groups separate?</th><th>Majority-only separates?</th><th>Evidence category</th></tr>"
            )
            for b in audit["boundaries"]:
                values = [
                    b["reference_time"],
                    f"{b['left_asr_word']} ends {b['left_word_end']} → {b['right_asr_word']} starts {b['right_word_start']}",
                    f"{b['raw_before_speaker']} → {b['raw_after_speaker']}",
                    f"{b['final_before_speaker']} → {b['final_after_speaker']}",
                    b.get("asr_backend_segments_separate"),
                    b.get("actual_separates"),
                    b.get("majority_only_separates"),
                    b["diagnosis"],
                ]
                content.append(
                    ("<tr>" if b.get("actual_separates") else "<tr class='bad'>")
                    + "".join(f"<td>{esc(v)}</td>" for v in values)
                    + "</tr>"
                )
            content.append("</table></div>")
        content.append("</details>")
    content.append(
        "<h2>Interpretation rules</h2><p>“Raw diarization merges boundary words” means the aligned words on each side received the same known raw identity; it does not prove whether an ASR word timestamp or the diarizer's change timing is acoustically wrong. “Grouping trailing override removes split” is a controlled replay result: disabling only that rule restores the split with the same observed words and diarizer timeline. The reference-timeline replay is an annotation-based counterfactual, not a deployed result.</p></body></html>"
    )
    path.write_text("\n".join(content), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
