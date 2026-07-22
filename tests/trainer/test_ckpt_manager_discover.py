import os
import shutil
import tempfile
import unittest

from deepspec.trainer.ckpt_manager import discover_latest_checkpoint


class TestDiscoverLatestCheckpoint(unittest.TestCase):
    def setUp(self):
        self.checkpoint_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.checkpoint_dir, ignore_errors=True)

    def test_no_link_returns_none(self):
        self.assertIsNone(discover_latest_checkpoint(self.checkpoint_dir))

    def test_healthy_symlink_returns_resolved_dir(self):
        target = os.path.join(self.checkpoint_dir, "step_10")
        os.makedirs(target)
        link = os.path.join(self.checkpoint_dir, "step_latest")
        os.symlink(target, link)
        self.assertEqual(
            os.path.realpath(link),
            discover_latest_checkpoint(self.checkpoint_dir),
        )

    def test_real_directory_returns_realpath(self):
        link = os.path.join(self.checkpoint_dir, "step_latest")
        os.makedirs(link)
        self.assertEqual(
            os.path.realpath(link),
            discover_latest_checkpoint(self.checkpoint_dir),
        )

    def test_dangling_symlink_raises(self):
        target = os.path.join(self.checkpoint_dir, "step_99")
        os.makedirs(target)
        link = os.path.join(self.checkpoint_dir, "step_latest")
        os.symlink(target, link)
        shutil.rmtree(target)
        with self.assertRaises(RuntimeError) as ctx:
            discover_latest_checkpoint(self.checkpoint_dir)
        message = str(ctx.exception)
        self.assertIn(link, message)
        self.assertIn(os.path.realpath(target), message)


if __name__ == "__main__":
    unittest.main()
