import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from pam_mosaic.checkpoint import Checkpoint
from pam_mosaic.preprocessing.dataset import Record
from pam_mosaic.pipeline import run
from pam_mosaic.frontend.registration import Match
from pam_mosaic.preprocessing.reconstruction import Calibration
from pam_mosaic.preprocessing.roundtrip import PairDecision


class CheckpointTests(unittest.TestCase):
    def test_identity_and_frame_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            data=root/'sample.dat'; data.touch()
            records=[Record(1,1,data)]
            cache=Checkpoint(root,records,{'mode':'a'})
            image=np.ones((4,4),np.float32)
            cache.save_frame(0,image,{'file_index':1});cache.flush_frames()
            loaded=Checkpoint(root,records,{'mode':'a'})
            np.testing.assert_array_equal(loaded.frame(0)[0],image)
            self.assertEqual(loaded.frame(0)[1],{'file_index':1})
            with self.assertRaises(ValueError):
                Checkpoint(root,records,{'mode':'different'})

    def test_resumed_pipeline_preserves_measurements_and_poses(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);records=[]
            for i in range(2):
                path=root/f'{i}.dat';path.touch();records.append(Record(i+1,1,path))
            output=root/'output'
            frame=np.ones((20,20),np.float32)*.1
            match=Match(np.eye(2,3),30,.1,1.,True)
            def fuse(frames,poses,destination,**kwargs):
                destination.mkdir(parents=True,exist_ok=True)
                return (20,20)
            with patch('pam_mosaic.preprocessing.formats.bidirectional500.reconstruct',return_value=(frame,frame)) as decode, \
                 patch('pam_mosaic.preprocessing.formats.bidirectional500.calibrate',return_value=Calibration()), \
                 patch('pam_mosaic.preprocessing.formats.bidirectional500.fuse_if_aligned',return_value=(frame,PairDecision(True,30,.1,0.,0.,'fused'))) as roundtrip, \
                 patch('pam_mosaic.pipeline.register',return_value=match) as register, \
                 patch('pam_mosaic.pipeline.candidate_pairs',return_value=[(0,1)]), \
                 patch('pam_mosaic.pipeline.fuse',side_effect=fuse):
                first=run(records,output)
                self.assertTrue(first['background_correction'])
                self.assertEqual(first['intensity_range'],[0.,.25])
                self.assertAlmostEqual(first['frame_decisions'][0]['background_baseline'],.1,places=6)
                np.testing.assert_allclose(np.load(output/'.run_cache/frame_0000.npy'),0,atol=1e-7)
                first_poses=np.load(output/'component_001/poses_xytheta.npy')
                register.reset_mock();decode.reset_mock();roundtrip.reset_mock()
                second=run(records,output)
                self.assertEqual(register.call_count,0)
                self.assertEqual(decode.call_count,0)
                self.assertEqual(roundtrip.call_count,0)
                self.assertEqual(first['registrations'],second['registrations'])
                self.assertEqual(first['frame_decisions'],second['frame_decisions'])
                np.testing.assert_array_equal(first_poses,np.load(output/'component_001/poses_xytheta.npy'))
                legacy=run(records,root/'legacy',background_correction=False)
                self.assertFalse(legacy['background_correction'])
                self.assertEqual(legacy['intensity_range'],[.02,.25])
                np.testing.assert_array_equal(np.load(root/'legacy/.run_cache/frame_0000.npy'),frame)


if __name__=='__main__':
    unittest.main()
