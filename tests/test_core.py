import os
import unittest

import pandas as pd
import mofax

TEST_MODEL = os.path.join(os.path.dirname(__file__), "mofa2_test_model.hdf5")


class TestMofaModelConnection(unittest.TestCase):
    def test_connection(self):
        self.assertIsInstance(mofax.mofa_model(TEST_MODEL), mofax.mofa_model)


class TestMofaModelMethods(unittest.TestCase):
    def setUp(self):
        self.model = mofax.mofa_model(TEST_MODEL)

    def test_nfactors(self):
        self.assertIsInstance(self.model.nfactors, int)

    def test_shape(self):
        self.assertIsInstance(self.model.shape, tuple)

    def test_cells(self):
        self.assertIsInstance(self.model.get_cells(), pd.DataFrame)

    def tearDown(self):
        self.model.close()


class TestR2(unittest.TestCase):
    def setUp(self):
        self.model = mofax.mofa_model(TEST_MODEL)

    def test_factor_r2(self):
        self.assertIsInstance(self.model.get_factor_r2(factor_index=0), pd.DataFrame)

    def tearDown(self):
        self.model.close()


if __name__ == "__main__":
    unittest.main()
