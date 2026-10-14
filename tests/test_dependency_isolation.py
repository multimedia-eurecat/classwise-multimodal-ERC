"""Core imports and both runner parsers work without robot/optional ML stacks."""

import subprocess
import sys
import unittest


class DependencyIsolationTests(unittest.TestCase):
    def test_imports_and_cli_parsing_do_not_load_optional_backends(self):
        code = """
import importlib.abc
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'nemo', 'rclpy', 'diart', 'pyannote', 'pymongo', 'faster_whisper', 'torch'}:
            raise ImportError('Optional runtime imported eagerly: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
import aimara.rob
from aimara.run_pipeline import build_parser as single
from aimara.run_batch import build_parser as batch
single().parse_args(['--video', 'v.mp4', '--output-dir', 'out', '--checkpoint', 'c.pt',
                    '--vad-weights', 'vad', '--asr-weights', 'asr'])
batch().parse_args(['--video-dir', 'videos', '--output-dir', 'out', '--checkpoint', 'c.pt',
                   '--vad-weights', 'vad', '--asr-weights', 'asr', '--asr-backend', 'parakeet'])
"""
        completed = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
