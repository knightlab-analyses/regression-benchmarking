import pandas as pd
import numpy as np
import json
from os import path
import biom
from qiime2 import Artifact
from qiime2.plugin.testing import TestPluginBase
from sklearn.model_selection import ParameterGrid
from q2_mlab.learningtask import ClassificationTask, RegressionTask


class UnitBenchmarkTests(TestPluginBase):
    package = "q2_mlab"

    def setUp(self):
        super().setUp()
        self.unit_benchmark = self.plugin.methods["unit_benchmark"]

        TEST_DIR = path.split(__file__)[0]
        table = path.join(
            TEST_DIR, "movingpicturestest/filtered_rarefied_table.qza"
        )
        metadata = path.join(
            TEST_DIR, "movingpicturestest/filtered_metadata.qza"
        )
        distance_matrix = path.join(
            TEST_DIR, "movingpicturestest/aitchison_distance_matrix.qza"
        )

        self.table = Artifact.load(table).view(biom.Table)
        self.metadata = Artifact.load(metadata).view(pd.Series)
        self.distance_matrix = Artifact.load(distance_matrix)
        self.n_repeats = 1
        n_classes = self.metadata.nunique()
        self.ncols_classification = 10 + n_classes
        self.ncols_regression = 8

        LinearSVR_grids = {
            "C": [1e-4, 1e-2, 1e-1, 1e1, 1e2, 1e4],
            "epsilon": [1e-2, 1e-1, 0, 1],
            "loss": ["squared_epsilon_insensitive", "epsilon_insensitive"],
            "random_state": [2018],
            "max_iter": [20000],
        }

        LinearSVC_grids = {
            "penalty": ["l2"],
            "tol": [1e-4, 1e-2, 1e-1],
            "random_state": [2018],
            "max_iter": [20000],
        }

        self.svr_params = json.dumps(list(ParameterGrid(LinearSVR_grids))[0])
        self.svc_params = json.dumps(list(ParameterGrid(LinearSVC_grids))[0])
        self.default_params = "{}"

        self.task = ClassificationTask(
            table=self.table,
            metadata=self.metadata,
            algorithm="LinearSVC",
            params=self.svc_params,
            n_repeats=self.n_repeats,
        )

    def testContainsNaNorInfinity(self):
        no_nans = [1, 0, 1, 0]
        one_nan = [1, None, 1, 0]
        np_nan = [1, np.nan, 1, 0]
        all_nans = [None, None, None, None]
        self.assertFalse(self.task.contains_nan(no_nans))
        self.assertTrue(self.task.contains_nan(one_nan))
        self.assertTrue(self.task.contains_nan(np_nan))
        self.assertTrue(self.task.contains_nan(all_nans))

    def testRegressionTaskInit(self):
        task = RegressionTask(
            table=self.table,
            metadata=self.metadata,
            algorithm="LinearSVR",
            params=self.svr_params,
            n_repeats=self.n_repeats,
        )

        task.results
        for key in task.results:
            self.assertEqual(len(task.results[key]), task.table_size)

        self.assertEqual(len(task.results), self.ncols_regression)

    def testClassificationTaskInit(self):
        task = ClassificationTask(
            table=self.table,
            metadata=self.metadata,
            algorithm="LinearSVC",
            params=self.svc_params,
            n_repeats=self.n_repeats,
        )

        task.results
        for key in task.results:
            self.assertEqual(len(task.results[key]), task.table_size)

        self.assertEqual(len(task.results), self.ncols_classification)
