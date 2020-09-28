import pandas as pd
import biom
import pkg_resources
from skbio.stats.distance import DistanceMatrix
from .learningtask import RegressionTask, ClassificationTask


def iter_entry_points():
    for entry_point in pkg_resources.iter_entry_points(
            group='q2_mlab.models'):
        yield entry_point


def _unit_benchmark(
    table: biom.Table,
    metadata: pd.Series,
    algorithm: str,
    params: str,
    distance_matrix: DistanceMatrix = None,
    n_repeats: int = 3,
) -> pd.DataFrame:

    # Add any custom algorithms from entry points
    for entry_point in iter_entry_points():
        name = entry_point.name
        method = entry_point.load()
        # Check what algorithm type the custom method/pipeline has
        if method._estimator_type == "regressor":
            RegressionTask.algorithms.update({name: method})
        if method._estimator_type == "classifier":
            ClassificationTask.algorithms.update({name: method})
        if (method._estimator_type not in ["regressor", "classifier"]):
            msg = (
                "Linked custom model is not a valid estimator type. "
                "Check attribute _estimator_type. "
                "Valid types are: classifier, regressor"
            )
            raise TypeError(msg)

    if algorithm in RegressionTask.algorithms:
        worker = RegressionTask(
            table, metadata, algorithm, params, n_repeats, distance_matrix
        )
    elif algorithm in ClassificationTask.algorithms:
        worker = ClassificationTask(
            table, metadata, algorithm, params, n_repeats, distance_matrix
        )
    else:
        raise ValueError(algorithm + " is not an accepted algorithm")

    for train_index, test_index in worker.splits:
        worker.cv_fold(train_index, test_index)

    results_table = worker.tabularize()

    return results_table, worker.best_model, worker.best_accuracy


def unit_benchmark(
    table: biom.Table,
    metadata: pd.Series,
    algorithm: str,
    params: str,
    distance_matrix: DistanceMatrix = None,
    n_repeats: int = 3,
) -> pd.DataFrame:

    results_table, _, _ = _unit_benchmark(
        table=table,
        metadata=metadata,
        algorithm=algorithm,
        params=params,
        distance_matrix=distance_matrix,
        n_repeats=n_repeats,
    )

    return results_table
