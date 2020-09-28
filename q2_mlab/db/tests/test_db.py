import tempfile
import unittest
import pandas as pd
from qiime2 import Artifact
from q2_mlab.db.schema import Parameters, RegressionScore
from q2_mlab.db.maint import (
    create,
    create_engine,
    add,
    add_from_qza,
)
from sqlalchemy.engine import Engine


class DBTestCase(unittest.TestCase):

    def test_create_in_memory(self):
        engine = create(echo=False)
        self.assertIsInstance(engine, Engine)

    def test_create_in_file(self):
        fh = tempfile.NamedTemporaryFile()
        engine = create(fh.name, echo=False)
        self.assertIsInstance(engine, Engine)
        fh.close()

    def test_add(self):
        engine = create(echo=False)
        parameters = {'max_features_STRING': 'log2'}
        results = pd.DataFrame([
            {'CV_IDX': 0, 'RUNTIME': 3.5, 'MAE': 2.4, 'RMSE': 7.24, 'R2': 0.7},
            {'CV_IDX': 1, 'RUNTIME': 2.3, 'MAE': 7.4, 'RMSE': 8.25, 'R2': 0.3},
        ])
        dataset = 'FINRISK'
        algorithm = 'RandomForestRegressor'
        level = 'MG'
        target = 'AGE'
        add(engine=engine, results=results, parameters=parameters,
            dataset=dataset, algorithm=algorithm, level=level, target=target,
            artifact_uuid='some-uuid',
            )

        param_df = pd.read_sql_table(Parameters.__tablename__, con=engine)
        self.assertEqual(1, len(param_df))
        score_df = pd.read_sql_table(RegressionScore.__tablename__, con=engine)
        self.assertEqual(2, len(score_df))

        add(engine=engine, results=results, parameters=parameters,
            dataset=dataset, algorithm=algorithm, level=level, target=target,
            artifact_uuid='some-uuid',
            )
        param_df = pd.read_sql_table(Parameters.__tablename__, con=engine)
        self.assertEqual(1, len(param_df))
        score_df = pd.read_sql_table(RegressionScore.__tablename__, con=engine)
        self.assertEqual(4, len(score_df))

        add(engine=engine, results=results, parameters={'activation': 'relu'},
            dataset=dataset, algorithm=algorithm, level=level, target=target,
            artifact_uuid='some-uuid',
            )
        param_df = pd.read_sql_table(Parameters.__tablename__, con=engine)
        self.assertEqual(2, len(param_df))
        score_df = pd.read_sql_table(RegressionScore.__tablename__, con=engine)
        self.assertEqual(6, len(score_df))

    def test_add_from_qza(self):
        format_name = "SampleData[Results]"

        results = pd.DataFrame([
            {'CV_IDX': 0, 'SAMPLE_ID': 'SAMPLE-1', 'Y_PRED': 4, 'Y_TRUE': 4,
             'RUNTIME': 3.5, 'MAE': 2.4, 'RMSE': 7.24, 'R2': 0.7},
            {'CV_IDX': 0, 'SAMPLE_ID': 'SAMPLE-2', 'Y_PRED': 5, 'Y_TRUE': 6,
             'RUNTIME': 3.5, 'MAE': 2.4, 'RMSE': 7.24, 'R2': 0.7},
            {'CV_IDX': 0, 'SAMPLE_ID': 'SAMPLE-3', 'Y_PRED': 5, 'Y_TRUE': 4,
             'RUNTIME': 3.5, 'MAE': 2.4, 'RMSE': 7.24, 'R2': 0.7},
            {'CV_IDX': 1, 'SAMPLE_ID': 'SAMPLE-1', 'Y_PRED': 4, 'Y_TRUE': 4,
             'RUNTIME': 2.3, 'MAE': 7.4, 'RMSE': 8.25, 'R2': 0.3},
        ])

        imported_artifact = Artifact.import_data(format_name,
                                                 results
                                                 )

        parameters = {"max_features": "log2", "gamma": 0.01, "normalize": True}

        dataset = 'FINRISK'
        algorithm = 'RandomForestRegressor'
        level = 'MG'
        target = 'AGE'

        engine = add_from_qza(imported_artifact, parameters, dataset, target,
                              level, algorithm,
                              engine_creator=create,
                              echo=False,
                              )

        param_df = pd.read_sql_table(Parameters.__tablename__, con=engine)
        self.assertEqual(1, len(param_df))
        self.assertListEqual(param_df['gamma_NUMBER'].values.tolist(), [0.01])
        self.assertListEqual(param_df['max_features_STRING'].values.tolist(),
                             ['log2'])

        score_df = pd.read_sql_table(RegressionScore.__tablename__, con=engine)
        self.assertEqual(2, len(score_df))
        self.assertCountEqual(score_df['R2'].values.tolist(), [0.7, 0.3])
        self.assertCountEqual(score_df['RUNTIME'].values.tolist(), [3.5, 2.3])
        self.assertEqual(score_df['parameters_id'].values.tolist(), [1, 1])

    def test_add_from_qza_None_parameter(self):
        format_name = "SampleData[Results]"

        results = pd.DataFrame([
            {'CV_IDX': 0, 'SAMPLE_ID': 'SAMPLE-1', 'Y_PRED': 4, 'Y_TRUE': 4,
             'RUNTIME': 3.5, 'MAE': 2.4, 'RMSE': 7.24, 'R2': 0.7},
        ])

        imported_artifact = Artifact.import_data(format_name,
                                                 results
                                                 )

        parameters = {"max_features": None, "gamma": 0.01, "normalize": True}

        dataset = 'FINRISK'
        algorithm = 'RandomForestRegressor'
        level = 'MG'
        target = 'AGE'

        engine = add_from_qza(imported_artifact, parameters, dataset, target,
                              level, algorithm,
                              engine_creator=create,
                              echo=False,
                              )

        param_df = pd.read_sql_table(Parameters.__tablename__, con=engine)
        self.assertEqual(1, len(param_df))
        self.assertListEqual(param_df['gamma_NUMBER'].values.tolist(), [0.01])
        self.assertListEqual(param_df['max_features_STRING'].values.tolist(),
                             [None])

        score_df = pd.read_sql_table(RegressionScore.__tablename__, con=engine)
        self.assertEqual(1, len(score_df))
        self.assertCountEqual(score_df['R2'].values.tolist(), [0.7])
        self.assertCountEqual(score_df['RUNTIME'].values.tolist(), [3.5])

    def test_add_parameters_thrice(self):
        format_name = "SampleData[Results]"

        results = pd.DataFrame([
            {'CV_IDX': 0, 'SAMPLE_ID': 'SAMPLE-1', 'Y_PRED': 4, 'Y_TRUE': 4,
             'RUNTIME': 3.5, 'MAE': 2.4, 'RMSE': 7.24, 'R2': 0.7},
        ])

        imported_artifact = Artifact.import_data(format_name,
                                                 results
                                                 )

        dataset = 'FINRISK'
        algorithm = 'RandomForestRegressor'
        level = 'MG'
        target = 'AGE'

        fh = tempfile.NamedTemporaryFile()
        db_file = fh.name

        parameters = {'bootstrap': True, 'criterion': 'mse', 'max_depth': 7,
                      'min_samples_leaf': 0.41, 'min_samples_split': 0.2,
                      'n_estimators': 100, 'n_jobs': -1, 'random_state': 2020,
                      }
        engine = add_from_qza(imported_artifact,
                              parameters,
                              dataset,
                              target,
                              level, algorithm,
                              db_file=db_file,
                              engine_creator=create,
                              echo=False,
                              )

        parameters = {'bootstrap': True, 'criterion': 'mse', 'max_depth': 7,
                      'max_features': None, 'max_samples': None,
                      'min_samples_leaf': 0.41, 'min_samples_split': 0.2,
                      'n_estimators': 100, 'n_jobs': -1, 'random_state': 2020,
                      }

        for _ in range(3):
            engine = add_from_qza(imported_artifact,
                                  parameters,
                                  dataset,
                                  target,
                                  level, algorithm,
                                  db_file=db_file,
                                  engine_creator=create_engine,
                                  echo=False,
                                  )

        param_df = pd.read_sql_table(Parameters.__tablename__, con=engine)
        self.assertEqual(1, len(param_df))

        fh.close()


if __name__ == '__main__':
    unittest.main()
