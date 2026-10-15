import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from path_tracking_overlay import paired_overlay, source_transform
from tracking_history import TrackingHistory


class TrackingOverlayTests(unittest.TestCase):
    def fields(self):
        return dict(frame='enu', source_stamp_ns=10_000_000_000, diagnostic_only=True,
                    tracking_valid=True, targets_valid=False, state='sensor_path_only',
                    vehicle_transform_verified=False, lateral_error_m=.2, heading_error_rad=.1,
                    reference_xyz=[[10, 20, 2], [10, 25, 2]], lookahead_xyz=[10, 23, 2],
                    projection_xyz=[10, 20, 2], ego_xyz=[10, 20, 2])

    def test_enu_rotation_and_sensor_only_label(self):
        transform=np.eye(4)
        transform[:3, :3]=[[0,-1,0],[1,0,0],[0,0,1]]
        transform[:3, 3]=[10,20,2]
        overlay, summary=paired_overlay((self.fields(),1.),transform,10_000_000_000,10_100_000_000,1.1)
        np.testing.assert_allclose(overlay['target'],[3,0,0])
        self.assertIn('axle transform UNSET',summary)
        self.assertIn('steering unavailable',summary)

    def test_expired_unpaired_invalid_and_missing_clear_geometry(self):
        for change in ({'source_stamp_ns':9_000_000_000},{'tracking_valid':False},
                       {'frame':'map'},{'reference_xyz':[[float('nan'),0,0]]},
                       {'source_stamp_ns':10_300_000_000}):
            fields={**self.fields(),**change}
            overlay,_=paired_overlay((fields,1.),np.eye(4),10_000_000_000,10_100_000_000,1.1)
            self.assertIsNone(overlay)
        self.assertIsNone(paired_overlay((self.fields(),1.),np.eye(4),10_000_000_000,10_100_000_000,2.)[0])
        self.assertIsNone(paired_overlay(None,np.eye(4),10_000_000_000,10_100_000_000,1.1)[0])

    def test_saved_pose_uses_map_to_enu_but_live_pose_already_enu(self):
        alignment=np.eye(4); alignment[0,3]=100
        pose=dict(transform=np.eye(4),query_stamp_ns=123)
        saved=SimpleNamespace(enu_from_map=alignment)
        transform,stamp=source_transform(saved,None,pose,None)
        self.assertEqual(transform[0,3],100)
        self.assertEqual(stamp,123)
        transform,_=source_transform(None,object(),None,pose)
        self.assertEqual(transform[0,3],0)
        self.assertEqual(source_transform(SimpleNamespace(enu_from_map=None),None,pose,None),(None,None))


class TrackingHistoryTests(unittest.TestCase):
    def fields(self, stamp, curvature=.1, speed=2.):
        return dict(tracking_valid=True, targets_valid=True, source_stamp_ns=stamp,
                    curvature_per_m=curvature, target_speed_mps=speed)

    def test_repeated_render_does_not_duplicate_a_target_sample(self):
        history=TrackingHistory()
        history.update(1_000_000_000,self.fields(1_000_000_000))
        history.update(1_100_000_000,self.fields(1_000_000_000))
        self.assertEqual(len(history.samples),1)
        self.assertEqual(history.latest,(.1,2.))

    def test_invalid_or_stale_targets_create_gaps_instead_of_zero(self):
        history=TrackingHistory()
        history.update(1_000_000_000,self.fields(1_000_000_000))
        fields={**self.fields(1_100_000_000),'targets_valid':False}
        history.update(1_100_000_000,fields)
        self.assertEqual(history.latest,(None,None))
        self.assertEqual(history.samples[-1][1:],(None,None))
        history.update(2_000_000_000,self.fields(1_000_000_000))
        self.assertFalse(history.available)

    def test_window_expiry_and_clock_rewind_clear_old_values(self):
        history=TrackingHistory(window_s=10.)
        history.update(1_000_000_000,self.fields(1_000_000_000))
        history.update(12_000_000_000,self.fields(12_000_000_000, -.1,3.))
        self.assertEqual(len(history.samples),1)
        history.update(2_000_000_000,self.fields(2_000_000_000,.2,1.))
        self.assertEqual(list(history.samples),[(2_000_000_000,.2,1.)])

    def test_vehicle_and_desired_are_independent_source_timestamped_series(self):
        history=TrackingHistory()
        fields={**self.fields(1_000_000_000), 'vehicle_curvature_valid':True,
                'vehicle_transform_verified':True, 'vehicle_curvature_source':'localization_pose_estimate',
                'vehicle_curvature_per_m':-.05}
        history.update(1_100_000_000,fields)
        self.assertEqual(list(history.samples),[(1_000_000_000,.1,2.)])
        self.assertEqual(list(history.vehicle_samples),[(1_000_000_000,-.05)])
        fields={**fields,'source_stamp_ns':1_200_000_000,'vehicle_curvature_valid':False}
        history.update(1_200_000_000,fields)
        self.assertEqual(history.latest,(.1,2.))
        self.assertIsNone(history.latest_vehicle)
        self.assertEqual(history.vehicle_samples[-1],(1_200_000_000,None))
        history.update(2_000_000_000,fields)
        self.assertFalse(history.vehicle_available)
        history.update(500_000_000,{**fields,'source_stamp_ns':500_000_000})
        self.assertEqual(len(history.vehicle_samples),1)

    def test_vehicle_estimate_rejects_invalid_values_or_unknown_source(self):
        for change in ({'vehicle_curvature_per_m':float('nan')},
                       {'vehicle_curvature_per_m':None},
                       {'vehicle_transform_verified':False},
                       {'vehicle_curvature_source':'unknown'}):
            fields={**self.fields(1_000_000_000), 'vehicle_curvature_valid':True,
                    'vehicle_transform_verified':True,'vehicle_curvature_per_m':.03,
                    'vehicle_curvature_source':'localization_pose_estimate',**change}
            history=TrackingHistory().update(1_000_000_000,fields)
            self.assertIsNone(history.latest_vehicle)


if __name__=='__main__':
    unittest.main()
