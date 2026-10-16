"""Portable inspection page for ROB exports and downstream ERC results."""

from __future__ import annotations

import html
import json
import math
import shutil
from dataclasses import asdict
from pathlib import Path

from .batch_resume import file_digest
from .prepare_manifest import read_reference_csv
from .speaker_ids import normalize_speaker_id


def _clip(part, duration):
    first, last = float(part["start"]), float(part["end"])
    if not math.isfinite(first) or not math.isfinite(last):
        raise ValueError("Timeline bounds must be finite")
    first, last = max(0.0, first), min(duration, last)
    return {**part, "start": first, "end": last} if first < last else None


def activity_timeline(events, source_start, duration):
    """Retrospective projection of saved activity events, not an ASR state dump.

    Preserve delivery order: REDI can backdate a label, including a final repeat
    after a stop. Each completed activity run replaces earlier evidence over
    its covered interval. Heartbeats extend a run; inactive events close it.
    Uncovered time remains uncovered, rather than guessing a speaker or silence.
    """
    spans, current = [], None

    def finish(run, end):
        nonlocal spans
        part = _clip({**run, "end": end}, duration)
        if part is None:
            return
        retained = []
        for previous in spans:
            if previous["end"] <= part["start"] or previous["start"] >= part["end"]:
                retained.append(previous)
            else:
                if previous["start"] < part["start"]:
                    retained.append({**previous, "end": part["start"]})
                if previous["end"] > part["end"]:
                    retained.append({**previous, "start": part["end"]})
        spans = sorted([*retained, part], key=lambda item: item["start"])

    for event in events:
        at = float(event["stamp_unix"]) - source_start
        if not math.isfinite(at):
            raise ValueError("Activity event stamp must be finite")
        speaker = event["speaker"]
        if event["active"]:
            if current is not None and current["speaker"] == speaker:
                current["start"] = min(current["start"], at)
                current["end"] = max(current["end"], at)
                current["confidence"] = event.get("confidence", -1)
            else:
                if current is not None:
                    finish(current, at)
                current = {
                    "start": at,
                    "end": at,
                    "speaker": speaker,
                    "label": speaker,
                    "confidence": event.get("confidence", -1),
                }
        elif current is not None and current["speaker"] == speaker:
            finish(current, at)
            current = None
    if current is not None:
        # No recorded stop: show only up to the final observed heartbeat.
        finish(current, current["end"])
    return spans


def timeline_data(capture, bounds, references):
    duration = capture["source_samples"] / capture["sample_rate"]
    clock = capture["source_start_unix"]
    inputs = {}
    groups = sorted(capture["groups"], key=lambda item: item["start_unix"])
    if len(groups) != len(bounds):
        raise ValueError("ROB groups and exported ERC bounds differ in length")
    final = []
    for index, (group, bound) in enumerate(zip(groups, bounds)):
        first, last = group.get("input_start_unix"), group.get("input_end_unix")
        if first is not None and last is not None:
            part = _clip(
                {
                    "start": first - clock,
                    "end": last - clock,
                    "label": "ASR input",
                    "speaker": "input",
                },
                duration,
            )
            if part is not None:
                inputs[(first, last)] = part
        final.append(
            {
                "start": bound["erc_start"],
                "end": bound["erc_end"],
                "speaker": normalize_speaker_id(group["speaker"]),
                "label": f"#{index} {normalize_speaker_id(group['speaker'])}",
                "text": group["text"],
            }
        )
    refs = []
    for reference in references:
        part = _clip(
            {
                "start": reference["start"],
                "end": reference["end"],
                "speaker": reference["speaker"],
                "label": f"ref {reference['speaker']}",
                "text": reference["text"],
            },
            duration,
        )
        if part is not None:
            refs.append(part)
    return {
        "duration": duration,
        "reference": refs,
        "asr_inputs": sorted(inputs.values(), key=lambda item: item["start"]),
        "raw_speakers": activity_timeline(capture.get("activity", []), clock, duration),
        "final_groups": final,
        "raw_method": "retrospective_activity_event_projection",
        "asr_input_scope": "Distinct inputs associated with published groups; empty/unpublished calls were not captured.",
    }


def _lane(title, parts, duration, video_id, *, reference=False):
    buttons, ends = [], []
    palette = ("#087f8c", "#4966c9", "#8457b5", "#b65332", "#407744")
    for part in parts:
        first, last = part["start"], part["end"]
        level = next(
            (index for index, end in enumerate(ends) if end <= first), len(ends)
        )
        if level == len(ends):
            ends.append(last)
        else:
            ends[level] = last
        speaker = part.get("speaker", "")
        number = speaker.removeprefix("speaker")
        color = (
            palette[(int(number) - 1) % len(palette)] if number.isdigit() else "#687789"
        )
        if speaker.casefold() == "unknown":
            color = "#a46b16"
        if reference:
            color = "#687789"
        tooltip = f"{first:.3f}–{last:.3f} s | {part['label']}"
        if part.get("text"):
            tooltip += " | " + part["text"]
        buttons.append(
            f'<button class="slice" data-video="{video_id}" data-start="{first}" data-end="{last}" '
            f'title="{html.escape(tooltip, quote=True)}" style="left:{100 * first / duration:.6f}%;width:{100 * (last - first) / duration:.6f}%;top:{level * 38}px;background:{color}">'
            f"{html.escape(part['label'])}</button>"
        )
    empty = '<span class="empty-lane">No recorded intervals</span>' if not parts else ""
    return (
        f'<div class="lane"><div class="lane-label">{html.escape(title)}</div>'
        f'<div class="track" style="height:{max(1, len(ends)) * 38}px">{empty}{"".join(buttons)}<i class="playhead"></i></div></div>'
    )


def _timeline(data, video_id):
    lanes = [
        ("Reference turns", "reference"),
        ("ASR input regions", "asr_inputs"),
        ("Raw speaker timeline", "raw_speakers"),
        ("Final ERC groups", "final_groups"),
    ]
    result = [
        f'<div class="timeline" data-timeline-for="{video_id}" data-duration="{data["duration"]}">'
    ]
    for title, name in lanes:
        result.append(
            _lane(
                title,
                data[name],
                data["duration"],
                video_id,
                reference=name == "reference",
            )
        )
    ticks = [float(value) for value in range(math.ceil(data["duration"]))] + [
        data["duration"]
    ]
    result.append(
        '<div class="lane"><div class="lane-label">Source time</div><div class="ruler">'
    )
    for tick in ticks:
        result.append(
            f'<span style="left:{100 * tick / data["duration"]:.6f}%">{tick:g}s</span>'
        )
    result.append("</div></div></div>")
    return "".join(result)


def write_review(output_dir, summary, reference_csv=None):
    output = Path(output_dir)
    snapshot = output / "reference.csv"
    if reference_csv is not None:
        source = Path(reference_csv)
        if source.resolve() != snapshot.resolve():
            shutil.copy2(source, snapshot)
    references = (
        [asdict(row) for row in read_reference_csv(snapshot)]
        if snapshot.is_file()
        else []
    )
    metadata = {
        "schema_version": 1,
        "renderer_sha256": file_digest(Path(__file__)),
        "reference_turns": len(references),
        "reference_sha256": file_digest(snapshot) if snapshot.is_file() else None,
        "videos": {},
    }
    cards = []
    for record in summary["videos"]:
        video = Path(record["video"])
        target = output / video.stem
        capture = json.loads((target / "rob_capture.json").read_text())
        bounds = json.loads((target / "erc_bounds.json").read_text())
        predictions_path = target / "predictions.json"
        predictions = (
            json.loads(predictions_path.read_text())
            if record["erc_status"] == "passed"
            else []
        )
        emotions = {item["segment_id"]: item["emotion"] for item in predictions}
        if video.is_file() and video.resolve() != (target / "source.mp4").resolve():
            shutil.copy2(video, target / "source.mp4")
        if not (target / "source.mp4").is_file():
            raise FileNotFoundError(f"Missing source video for review: {video}")
        video_id = f"v{len(cards)}"
        refs = [row for row in references if row["video_file"] == video.name]
        timeline = timeline_data(capture, bounds, refs)
        metadata["videos"][video.name] = {
            **timeline,
            "capture_sha256": file_digest(target / "rob_capture.json"),
            "bounds_sha256": file_digest(target / "erc_bounds.json"),
        }
        rows = []
        for index, (group, bound) in enumerate(
            zip(sorted(capture["groups"], key=lambda item: item["start_unix"]), bounds)
        ):
            first, last = bound["erc_start"], bound["erc_end"]
            rows.append(
                f'<tr><td><button data-video="v{len(cards)}" data-start="{first}" data-end="{last}">{first:.3f}–{last:.3f}s</button></td>'
                f"<td>{html.escape(group['speaker'])}</td><td>{html.escape(group['text'])}</td>"
                f"<td>{html.escape(emotions.get(index, '—'))}</td>"
                f'<td><audio controls preload="none" src="{html.escape(video.stem)}/audio/segment_{index:04d}.wav"></audio></td></tr>'
            )
        failure = html.escape(record.get("erc_failure", ""))
        if failure:
            failure = f'<p class="failure">ERC unavailable: {failure}</p>'
        cards.append(
            f"<section><h2>{html.escape(video.name)} — {len(rows)} ROB groups</h2>"
            f"<p>ERC: {html.escape(record['erc_status'])} · Maximum audio-delivery lateness: {record['max_replay_lateness_seconds'] * 1000:.1f} ms "
            f"· VAD maximum backlog: {capture.get('vad_delivery_max_backlog', 'unrecorded')} chunks</p>{failure}"
            f'<video id="v{len(cards)}" controls preload="metadata" src="{html.escape(video.stem)}/source.mp4"></video>'
            f'<p class="clock" id="clock-{video_id}">0.000 s</p>{_timeline(timeline, video_id)}'
            '<p class="note">Raw speakers: reconstructed from ROB activity events, including backdated updates. '
            "This is a retrospective view, not a dump of every internal ASR timeline state. "
            "Blank raw intervals mean no recorded active identity; they may be silence or unresolved speech. "
            "ASR inputs show only calls associated with published groups. Overlaps are stacked.</p>"
            f'<p><a href="{html.escape(video.stem)}/rob_capture.json">Raw ROB capture</a> · '
            f'<a href="{html.escape(video.stem)}/erc_bounds.json">Media-boundary adjustments</a> · '
            f'<a href="{html.escape(video.stem)}/rob.log">Engine log</a></p>'
            "<table><thead><tr><th>Play source interval</th><th>ROB speaker</th><th>ROB text</th><th>ERC emotion</th><th>Extracted audio</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></section>"
        )
    document = (
        """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Actual ROB engines → ERC</title>
<style>body{font:16px system-ui;max-width:1400px;margin:36px auto;padding:0 20px;background:#f5f7fa;color:#172432}section{background:white;padding:24px;margin:24px 0;border-radius:12px}video{width:480px;max-width:100%}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:12px;border-bottom:1px solid #ddd}audio{width:220px}button{cursor:pointer;white-space:nowrap;padding:8px}.failure{color:#a22020}a{color:#175cb6}.timeline{margin:22px 0}.lane{display:grid;grid-template-columns:180px minmax(0,1fr);gap:12px;margin:10px 0}.lane-label{font-size:14px;font-weight:650;padding-top:8px}.track{position:relative;background:#eef2f6;border-radius:5px}.slice{position:absolute;height:34px;border:0;border-radius:4px;color:white;text-align:left;overflow:hidden;font-size:12px;padding:7px}.slice:hover,.slice:focus-visible{outline:3px solid #172432;z-index:3}.playhead{position:absolute;left:0;top:0;bottom:0;border-left:2px solid #de364c;pointer-events:none;z-index:4}.ruler{position:relative;height:25px}.ruler span{position:absolute;transform:translateX(-50%);font-size:12px}.note{font-size:13px;line-height:1.6;color:#52687b}.clock{font-variant-numeric:tabular-nums}.empty-lane{font-size:12px;color:#52687b;padding:10px;display:block}@media(max-width:700px){.lane{grid-template-columns:110px minmax(0,1fr);gap:6px}.lane-label{font-size:12px}section{padding:12px}table{display:block;overflow-x:auto}}</style>
<h1>Actual ROB engines → ERC</h1>
<p>ROB owns VAD, speaker identity, transcription, and final grouping. ERC receives those groups without regrouping. This is real-time native-engine replay with direct callbacks, not a ROS transport test. Each video starts with fresh engines and a session-only identity gallery, VAD silence warmup and a recorded silence pre-roll. Robot speech and shared-gallery persistence are excluded.</p>
<p>Source intervals use ROB's internal wall-clock bounds and ASR offsets. Those bounds are not verified word times. Edge clipping and sample rounding are recorded; overlaps and gaps are retained. Click a time interval to inspect the original video. Browser stopping is approximate; extracted WAV bounds use the recorded samples.</p>
<p>Reference turns are supplied annotations, including pauses. ASR input regions are audio chunks passed to transcription, which can end on silence or a speaker change. Raw speakers are diarization identity/activity observations before sentence grouping. Final ERC groups are the resulting slices supplied to emotion inference. Reference 0/1 and ROB speaker1/2/3 are separate identity naming systems; matching colors do not establish ground-truth identity.</p>
<p><a href="rob_native.csv">Six-column CSV</a> · <a href="run_report.json">Run report, including ERC failures</a></p>
"""
        + (
            f'<p>Reference snapshot: <a href="reference.csv">{len(references)} turns</a> · SHA-256 <code>{metadata["reference_sha256"]}</code>. '
            '<a href="review_timelines.json">Timeline data and reconstruction provenance</a></p>'
            if references
            else "<p>No reference CSV was supplied. Rebuild with --reference-csv to fill the reference lane.</p>"
        )
        + "".join(cards)
        + """<script>
document.querySelectorAll('button[data-video]').forEach(button=>button.addEventListener('click',()=>{
 document.querySelectorAll('video').forEach(video=>video.pause());
 document.querySelectorAll('audio').forEach(audio=>audio.pause());
 const video=document.getElementById(button.dataset.video);video.currentTime=Number(button.dataset.start);video.dataset.stop=button.dataset.end;video.play().catch(()=>{});
}));
document.querySelectorAll('video').forEach(video=>video.addEventListener('timeupdate',()=>{
 document.getElementById('clock-'+video.id).textContent=video.currentTime.toFixed(3)+' s';
 const timeline=document.querySelector('[data-timeline-for="'+video.id+'"]');
 timeline.querySelectorAll('.playhead').forEach(head=>head.style.left=(100*Math.min(Number(timeline.dataset.duration),video.currentTime)/Number(timeline.dataset.duration))+'%');
 if(video.dataset.stop && video.currentTime>=Number(video.dataset.stop)){video.pause();delete video.dataset.stop;}
}));
document.querySelectorAll('audio').forEach(audio=>audio.addEventListener('play',()=>{
 document.querySelectorAll('video').forEach(video=>video.pause());
 document.querySelectorAll('audio').forEach(other=>{if(other!==audio)other.pause()});
}));
</script></html>"""
    )
    (output / "review_timelines.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "index.html").write_text(document, encoding="utf-8")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Rebuild the native ROB review from saved captures; no inference."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path)
    args = parser.parse_args()
    summary = json.loads((args.output_dir / "run_report.json").read_text())
    write_review(args.output_dir, summary, reference_csv=args.reference_csv)


if __name__ == "__main__":
    main()
