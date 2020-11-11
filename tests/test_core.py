import os
import unittest

import pandas as pd
import mofax

from mofax.plot import *

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

    def test_get_data(self):
        self.model.get_data()
        self.model.get_data(view=0, groups=0, df=True)

    def test_dimred(self):
        self.model.run_umap()

    def test_variance_explained(self):
        self.model.calculate_variance_explained(factors=[1,2,3])
        self.model.get_variance_explained()

    def tearDown(self):
        self.model.close()


# class TestR2(unittest.TestCase):
#     def setUp(self):
#         self.model = mofax.mofa_model(TEST_MODEL)

#     def test_factor_r2(self):
#         self.assertIsInstance(self.model.get_factor_r2(factor_index=0), pd.DataFrame)

#     def tearDown(self):
#         self.model.close()

class TestPlotting(unittest.TestCase):
    def setUp(self):
        self.model = mofax.mofa_model(TEST_MODEL)

    # def test_plot_factors(self):
    def test_plot_weights(self):
        plot_weights(self.model, factors=0, views=0)
        # plot_weights_ranked(self.model, factors=0, views=0)

    def tearDown(self):
        self.model.close()


if __name__ == "__main__":
    unittest.main()
