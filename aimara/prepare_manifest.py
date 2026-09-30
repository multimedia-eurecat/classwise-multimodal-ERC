"""Convert AIMARA reference annotations into an emotion-pipeline manifest."""

from __future__ import annotations

import argparse
import csv
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


REQUIRED_COLUMNS = {"video_file", "start", "end", "speaker", "text", "emotion"}
OUTPUT_COLUMNS = [
    "Dialogue_ID",
    "Utterance_ID",
    "Speaker",
    "Utterance",
    "Emotion",
    "Video_Path",
    "Audio_Path",
    "Start_Time",
    "End_Time",
    "Source_Video",
    "Source_Row",
]


@dataclass(frozen=True)
class ReferenceUtterance:
    video_file: str
    start: float
    end: float
    speaker: str
    text: str
    emotion: str
    source_row: int


def _detect_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;").delimiter
    except csv.Error as exc:
        raise ValueError("Could not determine whether the input CSV uses ',' or ';'") from exc


def read_reference_csv(path: Path) -> list[ReferenceUtterance]:
    """Read either AIMARA CSV variant and validate its required fields."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(8192)
        handle.seek(0)
        reader = csv.DictReader(handle, delimiter=_detect_delimiter(sample))

        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - fieldnames
        if missing:
            raise ValueError(f"{path}: missing required columns: {sorted(missing)}")

        utterances = []
        for row_number, row in enumerate(reader, start=2):
            try:
                start = float(row["start"])
                end = float(row["end"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{row_number}: start and end must be numbers") from exc

            if start < 0 or end <= start:
                raise ValueError(
                    f"{path}:{row_number}: invalid interval {start:g}-{end:g}"
                )

            values = {
                name: (row[name] or "").strip()
                for name in ("video_file", "speaker", "text", "emotion")
            }
            empty = [name for name, value in values.items() if not value]
            if empty:
                raise ValueError(f"{path}:{row_number}: empty fields: {empty}")

            utterances.append(
                ReferenceUtterance(
                    video_file=values["video_file"],
                    start=start,
                    end=end,
                    speaker=values["speaker"],
                    text=values["text"],
                    emotion=values["emotion"].lower(),
                    source_row=row_number,
                )
            )

    if not utterances:
        raise ValueError(f"{path}: no utterances found")
    return utterances


def extract_wav(
    video_path: Path,
    output_path: Path,
    start: float,
    end: float,
    *,
    ffmpeg: str = "ffmpeg",
) -> None:
    """Extract a mono 16 kHz PCM WAV for one utterance."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.6f}",
        "-i",
        str(video_path),
        "-t",
        f"{end - start:.6f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"ffmpeg executable not found: {ffmpeg}") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffmpeg failed for {video_path} interval {start:g}-{end:g}"
        ) from exc


def build_manifest(
    utterances: Iterable[ReferenceUtterance],
    *,
    video_dir: Path,
    output_dir: Path,
    selected_videos: Sequence[str] = (),
    extract_audio: bool = True,
    ffmpeg: str = "ffmpeg",
) -> list[dict[str, object]]:
    """Build deterministic dialogue/utterance rows and optionally extract WAVs."""
    grouped: dict[str, list[ReferenceUtterance]] = defaultdict(list)
    for utterance in utterances:
        grouped[utterance.video_file].append(utterance)

    requested = set(selected_videos)
    unknown = requested - set(grouped)
    if unknown:
        raise ValueError(f"Videos not present in the input CSV: {sorted(unknown)}")

    video_names = sorted(requested or grouped)
    rows: list[dict[str, object]] = []
    audio_dir = output_dir / "audio"

    for dialogue_id, video_name in enumerate(video_names):
        video_path = (video_dir / video_name).absolute()
        if not video_path.is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")

        ordered = sorted(
            grouped[video_name], key=lambda item: (item.start, item.end, item.source_row)
        )
        for utterance_id, utterance in enumerate(ordered):
            audio_path = (audio_dir / f"dia{dialogue_id}_utt{utterance_id}.wav").absolute()
            if extract_audio:
                extract_wav(
                    video_path,
                    audio_path,
                    utterance.start,
                    utterance.end,
                    ffmpeg=ffmpeg,
                )

            rows.append(
                {
                    "Dialogue_ID": dialogue_id,
                    "Utterance_ID": utterance_id,
                    "Speaker": utterance.speaker,
                    "Utterance": utterance.text,
                    "Emotion": utterance.emotion,
                    "Video_Path": str(video_path),
                    "Audio_Path": str(audio_path) if extract_audio else "",
                    "Start_Time": f"{utterance.start:.6f}",
                    "End_Time": f"{utterance.end:.6f}",
                    "Source_Video": video_name,
                    "Source_Row": utterance.source_row,
                }
            )

    return rows


def write_manifest(rows: Sequence[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare AIMARA reference annotations for emotion inference."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--video",
        action="append",
        default=[],
        help="Process only this filename; repeat to select multiple videos.",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Write the manifest without extracting utterance WAV files.",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    utterances = read_reference_csv(args.input_csv)
    rows = build_manifest(
        utterances,
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        selected_videos=args.video,
        extract_audio=not args.no_audio,
        ffmpeg=args.ffmpeg,
    )
    manifest_path = args.output_dir / "manifest.csv"
    write_manifest(rows, manifest_path)
    print(
        f"Wrote {len(rows)} utterances from "
        f"{len({row['Dialogue_ID'] for row in rows})} dialogue(s) to {manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

