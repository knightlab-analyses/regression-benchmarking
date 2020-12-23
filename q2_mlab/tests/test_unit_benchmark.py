import pandas as pd
import numpy as np
import biom
import json
from os import path
from qiime2 import Artifact
from qiime2.plugin.testing import TestPluginBase
from sklearn.model_selection import ParameterGrid
import q2_mlab


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

        self.table = Artifact.load(table)
        self.metadata = Artifact.load(metadata)
        self.distance_matrix = Artifact.load(distance_matrix)
        self.n_repeats = 1
        self.n_classes = self.metadata.view(pd.Series).nunique()
        self.ncols_classification = 10 + self.n_classes
        self.ncols_regression = 8
        self.expected_classification_cols = [
            "CV_IDX",
            "SAMPLE_ID",
            "Y_PRED",
            "Y_TRUE",
            "RUNTIME",
            "ACCURACY",
            "AUPRC",
            "AUROC",
            "F1",
            "BALANCED_ACCURACY",
        ] + ["PROB_CLASS_" + str(i) for i in range(self.n_classes)]

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

    def testRegressionTask(self):

        (results,) = self.unit_benchmark(
            table=self.table,
            metadata=self.metadata,
            algorithm="LinearSVR",
            params=self.svr_params,
            n_repeats=self.n_repeats,
            distance_matrix=self.distance_matrix,
        )

        table_df = self.table.view(pd.DataFrame)
        results_df = results.view(pd.DataFrame)

        # Assert format and content of results
        expected_shape = (
            table_df.shape[0] * self.n_repeats,
            self.ncols_regression,
        )
        self.assertTupleEqual(results_df.shape, expected_shape)

    def testClassificationNoProba(self):

        (results,) = self.unit_benchmark(
            table=self.table,
            metadata=self.metadata,
            algorithm="LinearSVC",
            params=self.svc_params,
            n_repeats=self.n_repeats,
            distance_matrix=self.distance_matrix,
        )

        table_df = self.table.view(pd.DataFrame)
        results_df = results.view(pd.DataFrame)

        # Assert format and content of results
        expected_shape = (
            table_df.shape[0] * self.n_repeats,
            self.ncols_classification,
        )
        self.assertTupleEqual(results_df.shape, expected_shape)

        # Assert columns exist and are null
        self.assertCountEqual(
            list(results_df.columns), self.expected_classification_cols
        )

    def testClassificationWithProba(self):

        (results,) = self.unit_benchmark(
            table=self.table,
            metadata=self.metadata,
            algorithm="RandomForestClassifier",
            params=self.default_params,
            n_repeats=self.n_repeats,
            distance_matrix=self.distance_matrix,
        )

        table_df = self.table.view(pd.DataFrame)
        results_df = results.view(pd.DataFrame)

        # Assert format and content of results
        expected_shape = (
            table_df.shape[0] * self.n_repeats,
            self.ncols_classification,
        )
        self.assertTupleEqual(results_df.shape, expected_shape)

        self.assertCountEqual(
            list(results_df.columns), self.expected_classification_cols
        )

    def testNoDistanceMatrix(self):

        (results,) = self.unit_benchmark(
            table=self.table,
            metadata=self.metadata,
            algorithm="LinearSVC",
            params=self.svc_params,
            n_repeats=self.n_repeats,
        )

        table_df = self.table.view(pd.DataFrame)
        results_df = results.view(pd.DataFrame)

        # Assert format and content of results
        expected_shape = (
            table_df.shape[0] * self.n_repeats,
            self.ncols_classification,
        )
        self.assertTupleEqual(results_df.shape, expected_shape)

        # Assert columns exist and are null
        self.assertCountEqual(
            list(results_df.columns), self.expected_classification_cols
        )

    def testReturnBestModel(self):

        results, best_model, best_accuracy = q2_mlab._unit_benchmark(
            table=self.table.view(biom.Table),
            metadata=self.metadata.view(pd.Series),
            algorithm="LinearSVC",
            params=self.svc_params,
            n_repeats=self.n_repeats,
        )

        self.assertTrue(best_model)
        self.assertEqual(results['ACCURACY'].max(), best_accuracy)

    def test_result_table(self):

        n_features = 10
        n_samples = 30
        test_data = np.arange(n_features*n_samples).reshape(
            n_features, n_samples
        )
        sample_ids = [f"S{i}" for i in range(n_samples)]
        obs_ids = [f"F{i}" for i in range(n_features)]
        test_table = biom.Table(test_data, obs_ids, sample_ids)

        continuous_target = pd.Series(
            [i for i in range(n_samples)],
            index=sample_ids
        )
        discrete_target = pd.Series(
            np.tile([0, 1], int(n_samples/2)),
            index=sample_ids
        )

        # Test Regression table
        results, _, _ = q2_mlab._unit_benchmark(
            table=test_table,
            metadata=continuous_target,
            algorithm="LinearSVR",
            params=self.svr_params,
            n_repeats=1
        )
        for sampleid in results.SAMPLE_ID:
            self.assertTrue(sampleid in sample_ids)
        for y in results.Y_TRUE:
            self.assertTrue(y in list(test_data.flatten()))

        # Test Classification table
        results, _, _ = q2_mlab._unit_benchmark(
            table=test_table,
            metadata=discrete_target,
            algorithm="LinearSVC",
            params=self.svc_params,
            n_repeats=1
        )
        for sampleid in results.SAMPLE_ID:
            self.assertTrue(sampleid in sample_ids)
        for y in results.Y_TRUE:
            self.assertTrue(y in list(test_data.flatten()))

    def testKNNDistanceMatrix(self):
        pass
