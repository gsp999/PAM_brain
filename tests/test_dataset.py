import tempfile
import unittest
from pathlib import Path

from pam_mosaic.dataset import index_dataset


class DatasetTests(unittest.TestCase):
    def test_custom_names_and_optional_scan_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("sample_12.dat", "sample_2.dat", "ignore.dat"):
                (root / name).touch()
            records = index_dataset(root, filename_regex=r"sample_(?P<index>\d+)\.dat")
            self.assertEqual([item.index for item in records], [2, 12])
            self.assertEqual([item.scan for item in records], [1, 1])
            self.assertTrue(all(item.path.is_absolute() for item in records))

    def test_duplicate_index_across_scans_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "1_Col2_tdT_58_scan1.dat").touch()
            (root / "1_Col2_tdT_58_scan2.dat").touch()
            with self.assertRaisesRegex(ValueError, "Duplicate acquisition indices"):
                index_dataset(root)

    def test_custom_regex_requires_index(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "named 'index'"):
                index_dataset(Path(directory), filename_regex=r"sample_\d+\.dat")


if __name__ == "__main__":
    unittest.main()
