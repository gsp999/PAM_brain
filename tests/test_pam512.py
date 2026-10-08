import struct
import tempfile
import unittest
from pathlib import Path
import numpy as np
from pam_mosaic.preprocessing.formats.pam512 import read_frame
from pam_mosaic.frontend.view import matching_image, registration_view
from pam_mosaic.pipeline import run
from pam_mosaic.preprocessing.dataset import Record


class Pam512Tests(unittest.TestCase):
    def test_pipeline_optional_denoise_cache_and_report(self):
        raw = np.full((512,1024),.03,dtype=np.float64)
        raw[100,600]=.5
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            path=root/'302.dat'
            with path.open('wb') as output:
                output.write(struct.pack('<ii',1024,512))
                output.write(raw.astype('<f8').tobytes(order='F'))
            record=Record(302,1,path)
            report=run([record],root/'filtered',input_format='pam512',background_correction=False)
            self.assertTrue(report['suppress_speckles'])
            self.assertFalse(report['validated_roundtrip'])
            self.assertEqual(report['speckle_changes'][0]['changed_pixels'],1)
            image=np.load(root/'filtered/.run_cache/frame_0000.npy')
            self.assertAlmostEqual(float(image[100,89]),.03,places=6)
            plain=run([record],root/'plain',input_format='pam512',suppress_speckles=False,background_correction=False)
            self.assertFalse(plain['suppress_speckles'])
            np.testing.assert_array_equal(np.load(root/'plain/.run_cache/frame_0000.npy'),
                                          raw[:,511:1024].astype(np.float32))
            adjusted=run([record],root/'adjusted',input_format='pam512',pam512_speckle_excess=.6,background_correction=False)
            self.assertEqual(adjusted['speckle_parameters']['excess'],.6)
            self.assertEqual(adjusted['speckle_changes'][0]['changed_pixels'],0)
            np.testing.assert_array_equal(np.load(root/'adjusted/.run_cache/frame_0000.npy'),
                                          raw[:,511:1024].astype(np.float32))
            with self.assertRaisesRegex(ValueError,'finite and positive'):
                run([record],root/'invalid',input_format='pam512',pam512_speckle_excess=float('nan'))
            corrected=run([record],root/'corrected',input_format='pam512',suppress_speckles=False)
            self.assertTrue(corrected['background_correction'])
            self.assertAlmostEqual(corrected['frame_decisions'][0]['background_baseline'],.03,places=6)
            c=np.load(root/'corrected/.run_cache/frame_0000.npy')
            self.assertAlmostEqual(float(c[100,89]),.47,places=6)
            self.assertAlmostEqual(float(c[0,0]),0,places=6)

    def test_matlab_layout_and_inclusive_column_selection(self):
        raw = np.arange(512*1024, dtype=np.float64).reshape(512,1024,order='F') - 1000
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'302.dat'
            with path.open('wb') as output:
                output.write(struct.pack('<ii',1024,512))
                output.write(raw.astype('<f8').tobytes(order='F'))
            image = read_frame(path)
            self.assertEqual(image.shape,(512,513))
            np.testing.assert_array_equal(image,raw[:,511:1024].astype(np.float32))
            with path.open('r+b') as output:
                output.write(struct.pack('<ii',512,1024))
            with self.assertRaisesRegex(ValueError,'header'):
                read_frame(path)

    def test_matching_window_does_not_change_source_values(self):
        source = np.array([[-.01,0,.03,.06,.1]],dtype=np.float32)
        original=source.copy()
        view = registration_view(matching_image(source,(0,.06)))
        np.testing.assert_array_equal(view[:,[0,1,3,4]],np.array([[0,0,255,255]],dtype=np.uint8))
        self.assertIn(int(view[0,2]),(127,128))
        np.testing.assert_array_equal(source,original)
        self.assertIs(matching_image(source,(.02,.25)),source)

    def test_truncated_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'302.dat'
            path.write_bytes(struct.pack('<ii',1024,512))
            with self.assertRaisesRegex(ValueError,'size'):
                read_frame(path)


if __name__=='__main__':
    unittest.main()
