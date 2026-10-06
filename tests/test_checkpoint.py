import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from pam_mosaic.checkpoint import Checkpoint
from pam_mosaic.dataset import Record
from pam_mosaic.pipeline import run
from pam_mosaic.registration import Match


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
            def fuse(frames,poses,destination):
                destination.mkdir(parents=True,exist_ok=True)
                return (20,20)
            with patch('pam_mosaic.pipeline.reconstruct',return_value=(frame,frame)) as decode, \
                 patch('pam_mosaic.pipeline.register',return_value=match) as register, \
                 patch('pam_mosaic.pipeline.candidate_pairs',return_value=[(0,1)]), \
                 patch('pam_mosaic.pipeline.fuse',side_effect=fuse):
                first=run(records,output,reverse_even=False,use_backward=False)
                first_poses=np.load(output/'component_001/poses_xytheta.npy')
                register.reset_mock();decode.reset_mock()
                second=run(records,output,reverse_even=False,use_backward=False)
                self.assertEqual(register.call_count,0)
                self.assertEqual(decode.call_count,2) # baseline calibration prepass only
                self.assertEqual(first['registrations'],second['registrations'])
                np.testing.assert_array_equal(first_poses,np.load(output/'component_001/poses_xytheta.npy'))


if __name__=='__main__':
    unittest.main()
