import tempfile
import unittest
from pathlib import Path

import numpy as np

from aimara import (
    OfflineROBProcessor,
    ROBProcessorConfig,
    SpeakerInterval,
    SpeechRegion,
    Transcript,
)


class FakeVADProcessor:
    def __init__(self, regions):
        self.config = ROBProcessorConfig(sample_rate=16000)
        self.regions = regions
        self.audio = None

    def process(self, audio):
        self.audio = audio
        return self.regions


class FakeTranscriber:
    def __init__(self, results):
        self.results = iter(results)
        self.calls = []

    def transcribe(self, audio, sample_rate):
        self.calls.append((audio.copy(), sample_rate))
        return next(self.results)


class OfflineROBProcessorTests(unittest.TestCase):
    def test_uses_diarization_timeline_without_changing_audio_or_transcript_boundaries(
        self,
    ):
        audio = np.arange(32000, dtype=np.float32)
        regions = [SpeechRegion(0, 16000), SpeechRegion(16000, 32000)]
        calls = []

        class Diarizer:
            def diarize(self, samples, rate, speech_regions):
                calls.append((samples, rate, speech_regions))
                return [
                    SpeakerInterval(0, 16000, "speaker1"),
                    SpeakerInterval(16000, 32000, "speaker2"),
                ]

        transcriber = FakeTranscriber([Transcript("first"), Transcript("second")])
        processor = OfflineROBProcessor(
            vad_processor=FakeVADProcessor(regions),
            transcriber=transcriber,
            diarizer=Diarizer(),
            audio_decoder=lambda path, sample_rate: audio,
        )
        with tempfile.TemporaryDirectory() as directory:
            segments = processor.process_video("two-speakers.mp4", directory)
            self.assertEqual([s.speaker_id for s in segments], ["speaker1", "speaker2"])
            self.assertEqual(
                [(s.start_time, s.end_time) for s in segments], [(0, 1), (1, 2)]
            )
            self.assertEqual([s.transcript for s in segments], ["first", "second"])
            import wave

            with wave.open(str(segments[1].audio_path)) as wav:
                self.assertEqual(wav.getnframes(), 16000)
        self.assertIs(calls[0][0], audio)
        self.assertEqual(calls[0][1:], (16000, regions))
        np.testing.assert_array_equal(transcriber.calls[1][0], audio[16000:])

    def test_empty_diarization_is_unknown_and_does_not_guess_the_configured_speaker(
        self,
    ):
        from unittest.mock import Mock

        diarizer = Mock()
        diarizer.diarize.return_value = []
        processor = OfflineROBProcessor(
            vad_processor=FakeVADProcessor([SpeechRegion(0, 16000)]),
            transcriber=FakeTranscriber([Transcript("hello")]),
            diarizer=diarizer,
            speaker_id="fallback-person",
            audio_decoder=lambda path, sample_rate: np.ones(16000, dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as directory:
            segments = processor.process_video("video.mp4", directory)
        self.assertEqual(segments[0].speaker_id, "UNKNOWN")

    def test_invalid_timeline_fails_before_transcription(self):
        from unittest.mock import Mock

        for timeline in [
            [SpeakerInterval(0, 16001, "speaker1")],
            [
                SpeakerInterval(0, 12000, "speaker1"),
                SpeakerInterval(10000, 16000, "speaker2"),
            ],
        ]:
            diarizer = Mock()
            diarizer.diarize.return_value = timeline
            transcriber = FakeTranscriber([])
            processor = OfflineROBProcessor(
                vad_processor=FakeVADProcessor([SpeechRegion(0, 16000)]),
                transcriber=transcriber,
                diarizer=diarizer,
                audio_decoder=lambda path, sample_rate: np.ones(
                    16000, dtype=np.float32
                ),
            )
            with (
                tempfile.TemporaryDirectory() as directory,
                self.assertRaises(ValueError),
            ):
                processor.process_video("video.mp4", directory)
            self.assertEqual(transcriber.calls, [])

    def test_silent_video_skips_diarization(self):
        from unittest.mock import Mock

        diarizer = Mock()
        processor = OfflineROBProcessor(
            vad_processor=FakeVADProcessor([]),
            transcriber=FakeTranscriber([]),
            diarizer=diarizer,
            audio_decoder=lambda path, sample_rate: np.zeros(16000, dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(processor.process_video("quiet.mp4", directory), [])
        diarizer.diarize.assert_not_called()

    def test_creates_ordered_speech_segments_and_wavs(self):
        audio = np.arange(12, dtype=np.float32)
        vad = FakeVADProcessor([SpeechRegion(1, 5), SpeechRegion(7, 12)])
        transcriber = FakeTranscriber(
            [
                Transcript("first text", "en"),
                Transcript("second text", "en"),
            ]
        )
        decoder_calls = []
        writer_calls = []

        def decoder(path, *, sample_rate):
            decoder_calls.append((path, sample_rate))
            return audio

        def writer(path, segment_audio, *, sample_rate):
            writer_calls.append((path, segment_audio.copy(), sample_rate))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            return path

        processor = OfflineROBProcessor(
            vad_processor=vad,
            transcriber=transcriber,
            audio_decoder=decoder,
            wav_writer=writer,
        )

        with tempfile.TemporaryDirectory() as directory:
            segments = processor.process_video(
                "video.mp4",
                directory,
                conversation_id="conversation-1",
            )

            self.assertEqual(len(segments), 2)
            self.assertEqual(segments[0].conversation_id, "conversation-1")
            self.assertEqual(segments[0].segment_id, 0)
            self.assertEqual(segments[0].speaker_id, "UNKNOWN")
            self.assertEqual(segments[0].transcript, "first text")
            self.assertEqual(segments[0].start_time, 1 / 16000)
            self.assertEqual(segments[0].end_time, 5 / 16000)
            self.assertEqual(segments[1].audio_path.name, "segment_0001.wav")
            self.assertTrue(all(segment.audio_path.is_file() for segment in segments))

        self.assertEqual(decoder_calls[0][1], 16000)
        self.assertIs(vad.audio, audio)
        self.assertEqual(len(transcriber.calls), 2)
        np.testing.assert_array_equal(transcriber.calls[0][0], audio[1:5])
        np.testing.assert_array_equal(writer_calls[1][1], audio[7:12])

    def test_no_vad_regions_returns_empty_result(self):
        audio = np.zeros(16, dtype=np.float32)
        vad = FakeVADProcessor([])
        transcriber = FakeTranscriber([])
        processor = OfflineROBProcessor(
            vad_processor=vad,
            transcriber=transcriber,
            audio_decoder=lambda path, sample_rate: audio,
        )

        with tempfile.TemporaryDirectory() as directory:
            segments = processor.process_video("quiet.mp4", directory)
            self.assertTrue((Path(directory) / "audio").is_dir())

        self.assertEqual(segments, [])
        self.assertEqual(transcriber.calls, [])

    def test_rejects_empty_default_speaker_id(self):
        with self.assertRaisesRegex(ValueError, "speaker_id"):
            OfflineROBProcessor(
                vad_processor=FakeVADProcessor([]),
                transcriber=FakeTranscriber([]),
                speaker_id=" ",
            )


if __name__ == "__main__":
    unittest.main()
