"""Data-free tests; toy matrices below are not research observations."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
B=ROOT/"13_CMIP6_GridPath_Figures"
sys.path.insert(0,str(B/"72_Supplier_Switch_Experiment_20261111"))
sys.path.insert(0,str(B/"32_NCC2023_IndirectExposure"))
sys.path.insert(0,str(ROOT/"tools"))

from analyse_switch import evaluate
from market import cap_allocate
from test_hem_model import TestHEM
from check_release import check
from run_stage import STAGES


class ReleaseTests(unittest.TestCase):
    def test_release_audit(self):
        self.assertEqual(check(),0)

    def test_cap_allocation_conserves_and_respects_capacity(self):
        out=cap_allocate(np.array([3.,1.,2.]),np.array([1.,4.,3.]),6.)
        self.assertAlmostEqual(out.sum(),6.)
        self.assertTrue(np.all(out>=0))
        self.assertTrue(np.all(out<=np.array([1.,4.,3.])))

    def test_allocation_under_insufficient_supply(self):
        out=cap_allocate(np.zeros(3),np.array([1.,2.,3.]),9.)
        np.testing.assert_allclose(out,[1.,2.,3.])

    def test_supplier_switch_excludes_additional_filled_demand(self):
        x={"base":np.array([[60.],[40.]]),"states":{"A":{"cost":np.zeros((2,1))}}}
        zero=np.zeros((1,1))
        neutral=np.zeros((1,1),dtype=int)
        before=np.array([[[50.,30.]]])
        after=np.array([[[40.,60.]]])
        r=evaluate(x,2,3,before,after,zero,zero,neutral,neutral,.20)
        np.testing.assert_equal(r["volume"],[[10.]])
        np.testing.assert_equal(r["switched"],[[True]])
        np.testing.assert_equal(r["avoided"],[[4.]])

    def test_stage_commands_exist(self):
        for stage in STAGES.values():
            for cmd in stage["commands"]:
                self.assertTrue((ROOT/cmd[0]).is_file(),cmd[0])


if __name__=="__main__":
    unittest.main()
