import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aimara.prepare_manifest import (
    build_manifest,
    extract_wav,
    read_reference_csv,
    write_manifest,
)


class PrepareManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.video_dir = self.root / "videos"
        self.video_dir.mkdir()
        (self.video_dir / "Anger.mp4").touch()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_csv(self, text: str) -> Path:
        path = self.root / "input.csv"
        path.write_text(text, encoding="utf-8-sig")
        return path

    def test_reads_comma_csv_and_orders_utterances(self):
        path = self._write_csv(
            "video_file,start,end,speaker,text,emotion\n"
            "Anger.mp4,3.3,4.94,1,Wait,neutral\n"
            "Anger.mp4,0,3.3,0,Unacceptable,ANGER\n"
        )

        rows = build_manifest(
            read_reference_csv(path),
            video_dir=self.video_dir,
            output_dir=self.root / "output",
            extract_audio=False,
        )

        self.assertEqual([row["Utterance_ID"] for row in rows], [0, 1])
        self.assertEqual([row["Utterance"] for row in rows], ["Unacceptable", "Wait"])
        self.assertEqual(rows[0]["Emotion"], "anger")
        self.assertEqual(rows[0]["Audio_Path"], "")

    def test_reads_semicolon_csv_and_selects_video(self):
        (self.video_dir / "Neutral.mp4").touch()
        path = self._write_csv(
            "video_file;start;end;speaker;text;emotion\n"
            "Anger.mp4;0;1;0;No;anger\n"
            "Neutral.mp4;0;1;1;Okay;neutral\n"
        )

        rows = build_manifest(
            read_reference_csv(path),
            video_dir=self.video_dir,
            output_dir=self.root / "output",
            selected_videos=["Neutral.mp4"],
            extract_audio=False,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Source_Video"], "Neutral.mp4")
        self.assertEqual(rows[0]["Dialogue_ID"], 0)

    def test_rejects_invalid_interval(self):
        path = self._write_csv(
            "video_file,start,end,speaker,text,emotion\n"
            "Anger.mp4,2,1,0,No,anger\n"
        )

        with self.assertRaisesRegex(ValueError, "invalid interval"):
            read_reference_csv(path)

    def test_write_manifest_uses_expected_columns(self):
        path = self._write_csv(
            "video_file,start,end,speaker,text,emotion\n"
            "Anger.mp4,0,1,0,No,anger\n"
        )
        rows = build_manifest(
            read_reference_csv(path),
            video_dir=self.video_dir,
            output_dir=self.root / "output",
            extract_audio=False,
        )
        manifest_path = self.root / "output" / "manifest.csv"

        write_manifest(rows, manifest_path)

        with manifest_path.open(encoding="utf-8", newline="") as handle:
            saved = list(csv.DictReader(handle))
        self.assertEqual(saved[0]["Dialogue_ID"], "0")
        self.assertEqual(saved[0]["Video_Path"], str((self.video_dir / "Anger.mp4").absolute()))

    @patch("aimara.prepare_manifest.subprocess.run")
    def test_extract_wav_invokes_ffmpeg_without_a_shell(self, run):
        extract_wav(
            self.video_dir / "Anger.mp4",
            self.root / "audio" / "dia0_utt0.wav",
            1.0,
            2.5,
        )

        command = run.call_args.args[0]
        self.assertEqual(command[0], "ffmpeg")
        self.assertIn("1.500000", command)
        run.assert_called_once_with(command, check=True)


if __name__ == "__main__":
    unittest.main()

