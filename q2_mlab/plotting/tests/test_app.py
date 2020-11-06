import unittest
import pandas as pd
import numpy.testing as npt
from q2_mlab.plotting.app import _get_standardized_mae


class TestAppUtils(unittest.TestCase):

    def test_get_standardize_mae(self):
        df = pd.DataFrame({
            'MAE': [1, 2, 3, 4],
            'target': ['bmi', 'age', 'bmi', 'age'],
            'dataset': ['finrisk', 'finrisk', 'sol', 'sol'],
            'level': ['16S', '16S', 'MG', 'MG'],
            'CV_IDX': [0, 0, 1, 0]
        })
        norms = {
            "(finrisk, age, 16S, 0)": 2,
            "(sol, bmi, MG, 1)": 0.5,
        }
        obs0 = _get_standardized_mae(df.iloc[0, :], norms)
        self.assertEqual(1, obs0)
        obs1 = _get_standardized_mae(df.iloc[1, :], norms)
        self.assertEqual(1, obs1)
        obs2 = _get_standardized_mae(df.iloc[2, :], norms)
        self.assertEqual(6, obs2)
        obs3 = _get_standardized_mae(df.iloc[3, :], norms)
        self.assertEqual(4, obs3)

        obs_series = df.apply(_get_standardized_mae, axis=1, args=(norms,))
        npt.assert_array_equal([1, 1, 6, 4], obs_series.values)

