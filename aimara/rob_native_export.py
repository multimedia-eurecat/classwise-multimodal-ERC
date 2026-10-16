"""Observe ROB's final groups without changing its attribution decisions.

``observe_groups`` can also be installed on an engine owned by ROB's ROS node.
The existing transcript callback still receives precisely its original arguments.
"""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from types import MethodType

from .audio_io import decode_audio, write_wav
from .batch_resume import file_digest
from .domain import SpeechSegment
from .speech_metadata import ASRMetadata


def observe_groups(engine, on_group):
    """Tap final published groups, including the final UNKNOWN-grace retry.

    ROB groups and publishes on the same thread under its publication lock.
    Thread-local state also protects concurrent transcription callers. No clocks,
    thresholds, speaker resolution, text, or callback arguments are replaced.
    Returns an uninstall function; install only once, before feeding audio.
    """
    if getattr(engine, "_aimara_group_observer", False):
        raise ValueError("ROB engine already has a group observer")
    original_group = engine._group_segments_by_speaker
    original_ready = engine._on_transcript_ready
    state = threading.local()

    def group(self, segments, start_time, end_time):
        groups = original_group(segments, start_time, end_time)
        state.groups = [dict(item) for item in groups]
        state.index = 0
        state.start = start_time
        state.end = end_time
        return groups

    def ready(*args):
        item = state.groups[state.index]
        state.index += 1
        if item["text"] != args[0] or item["speaker"] != args[1]:
            raise RuntimeError("ROB publication differs from the observed final group")
        if state.start is None:
            raise RuntimeError("ROB published a group without an audio timestamp")
        on_group(
            {
                "text": args[0],
                "speaker": args[1],
                "language": args[2],
                "transcript_confidence": args[3],
                "processing_ms": args[4],
                "vad_wait_ms": args[5],
                "audio_duration_ms": args[6],
                "realtime_factor": args[7],
                "speaker_confidence": args[8] if len(args) > 8 else -1.0,
                "input_start_unix": state.start,
                "input_end_unix": state.end,
                "start_offset": item["start_offset"],
                "end_offset": item["end_offset"],
                "start_unix": state.start + item["start_offset"],
                "end_unix": state.start + item["end_offset"],
            }
        )
        return original_ready(*args)

    engine._group_segments_by_speaker = MethodType(group, engine)
    engine._on_transcript_ready = ready
    engine._aimara_group_observer = True

    def uninstall():
        engine._group_segments_by_speaker = original_group
        engine._on_transcript_ready = original_ready
        engine._aimara_group_observer = False

    return uninstall


class ROBExportProcessor:
    """Materialize exported ROB groups for ERC; never regroup or relabel them.

    Group bounds are ROB's own wall-clock bounds mapped onto the source clock.
    Clipping at the physical media edges and sample rounding are recorded. Gaps
    and overlaps are preserved, rather than repaired with invented cut points.
    """

    def __init__(self, capture_path):
        self.capture_path = Path(capture_path)

    def process_video(self, video_path, output_dir):
        capture = json.loads(self.capture_path.read_text())
        if capture.get("schema_version") != 1 or capture.get("status") != "passed":
            raise ValueError("A complete, successful ROB capture is required")
        source = Path(video_path).resolve()
        if file_digest(source) != capture["source_video_sha256"]:
            raise ValueError("ROB capture does not belong to this source video")
        rate = capture["sample_rate"]
        if rate != 16000:
            raise ValueError("ROB exports must use 16 kHz audio")
        audio = decode_audio(source, sample_rate=rate)
        if len(audio) != capture["source_samples"]:
            raise ValueError("ROB capture source audio length differs")
        start_clock = float(capture["source_start_unix"])
        if not math.isfinite(start_clock):
            raise ValueError("Invalid source clock")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        segments, bounds = [], []
        # Publication order is retained in the capture. ERC consumes time order.
        groups = sorted(capture["groups"], key=lambda item: item["start_unix"])
        for index, group in enumerate(groups):
            start = float(group["start_unix"]) - start_clock
            end = float(group["end_unix"]) - start_clock
            if not (math.isfinite(start) and math.isfinite(end) and end > start):
                raise ValueError(
                    f"Invalid ROB bounds for group {index}: {start}, {end}"
                )
            first = max(0, min(len(audio), round(start * rate)))
            last = max(0, min(len(audio), round(end * rate)))
            if first >= last:
                raise ValueError(f"ROB group {index} falls entirely outside the media")
            metadata = capture["asr"]
            confidence = group["transcript_confidence"]
            # Upstream Whisper's 0 is an unavailable-score sentinel, not 0% accuracy.
            is_parakeet = metadata["backend"] == "parakeet"
            segment = SpeechSegment(
                conversation_id=source.stem,
                segment_id=index,
                source_video=source,
                audio_path=write_wav(
                    output / "audio" / f"segment_{index:04d}.wav",
                    audio[first:last].copy(),
                    sample_rate=rate,
                ),
                start_time=first / rate,
                end_time=last / rate,
                speaker_id=group["speaker"],
                transcript=group["text"],
                language=group["language"] or None,
                speaker_id_confidence=group["speaker_confidence"],
                asr_metadata=ASRMetadata(
                    backend=metadata["backend"],
                    model=metadata["model"],
                    device=metadata.get("device"),
                    compute_type=metadata.get("compute_type"),
                    transcript_confidence=confidence if is_parakeet else None,
                    confidence_method="rob_nemo_word_mean" if is_parakeet else None,
                ),
            )
            segments.append(segment)
            bounds.append(
                {
                    "segment_id": index,
                    "rob_start": start,
                    "rob_end": end,
                    "erc_start": segment.start_time,
                    "erc_end": segment.end_time,
                    "edge_clipped": start < 0 or end > len(audio) / rate,
                }
            )
        (output / "erc_bounds.json").write_text(json.dumps(bounds, indent=2) + "\n")
        return segments
