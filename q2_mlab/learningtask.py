import json
import numpy as np
import pandas as pd
import time
from abc import ABC

# CV Methods
from sklearn.model_selection import RepeatedStratifiedKFold
from calour.training import RepeatedSortedStratifiedKFold

# Metrics
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    f1_score,
    balanced_accuracy_score
)

# Algorithms
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import (
    RidgeClassifier,
    Ridge,
    LogisticRegression,
    LinearRegression,
)
from xgboost import XGBRegressor, XGBClassifier
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.svm import SVC, SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import ElasticNet, Lasso


# Not all regressors in Scikit-Learn have corresponding classifiers,
# e.g. "Lasso" classification is implemented in LogisticRegression with L1
# penalty.
# This ModelFactory enforces the names of models correspond appropriately to
# the defining parameters, e.g. any model using named RadialSVR must have
# 'rbf' as the kernel.
class ModelFactory:
    @staticmethod
    def get_model(model_class, **outer_kwargs):
        class SpecificModel(model_class):
            def __init__(self, **kwargs):
                super().__init__(**outer_kwargs, **kwargs)

        return SpecificModel


# scikit-learn has some asymmetries between classifiers and regressors, e.g.,
# `Lasso`'s classification equivalent is `LogisticRegression` with
# `penalty='l1'`. You can use ModelFactory.get_model to create Estimator
# classes that set specific parameters.
LogisticRegressionLasso = ModelFactory.get_model(
    LogisticRegression, penalty="l1"
)
LogisticRegressionElasticNet = ModelFactory.get_model(
    LogisticRegression, penalty="elasticnet"
)
RadialSVR = ModelFactory.get_model(SVR, kernel="rbf")
SigmoidSVR = ModelFactory.get_model(SVR, kernel="sigmoid")
RadialSVC = ModelFactory.get_model(SVC, kernel="rbf")
SigmoidSVC = ModelFactory.get_model(SVC, kernel="sigmoid")
LGBMRegressorGBDT = ModelFactory.get_model(LGBMRegressor, boosting_type="gbdt")
LGBMRegressorRF = ModelFactory.get_model(LGBMRegressor, boosting_type="rf")
LGBMClassifierGBDT = ModelFactory.get_model(
    LGBMClassifier, boosting_type="gbdt"
)
LGBMClassifierRF = ModelFactory.get_model(LGBMClassifier, boosting_type="rf")


class LearningTask(ABC):
    algorithms = {}

    def __init__(
        self,
        table,
        metadata,
        algorithm,
        params,
        n_repeats,
        distance_matrix=None,
    ):

        self.learner = self.algorithms[algorithm]
        print(params)
        self.params = json.loads(params)
        if isinstance(self.learner, Pipeline):
            # Assumes that the last step in the pipeline is the model:
            prefix = list(self.learner.named_steps)[-1] + "__"
            # And adds the prefix of that last step to our param dict's keys
            # so we can access that step's parameters.
            newparams = {prefix + key: val for key, val in self.params.items()}
            self.params = newparams
        self.X = table.transpose().matrix_data
        self.metadata = metadata
        self.y = self.metadata.to_numpy()
        self.distance_matrix = distance_matrix
        self.cv_idx = 0
        self.idx = 0
        self.n_repeats = n_repeats
        self.n_classes = self.metadata.nunique()
        self.table_size = self.n_repeats * self.y.shape[0]
        self.best_accuracy = -1
        self.best_model = None

        self.results = {}
        self.results["CV_IDX"] = np.full(
            self.table_size, fill_value=np.nan, dtype=int
        )
        self.results["SAMPLE_ID"] = np.full(
            self.table_size, fill_value=np.nan, dtype=object
        )
        self.results["Y_PRED"] = np.full(
            self.table_size, fill_value=np.nan, dtype=float
        )
        self.results["Y_TRUE"] = np.full(
            self.table_size, fill_value=np.nan, dtype=object
        )
        self.results["RUNTIME"] = np.full(
            self.table_size, fill_value=np.nan, dtype=float
        )

        # Check for sample id agreement between table and metadata
        if list(metadata.index) != list(table.ids()):
            raise ValueError(
                "Table and Metadata Sample IDs do not match in contents "
                "and/or order"
            )

    def contains_nan(self, y_pred):
        if (np.any(pd.isnull(y_pred))) or (not np.all(np.isfinite(y_pred))):
            return True
        else:
            return False

    def tabularize(self):
        results_table = pd.DataFrame.from_dict(self.results)
        return results_table


class ClassificationTask(LearningTask):

    algorithms = {
        "KNeighborsClassifier": KNeighborsClassifier,
        "RandomForestClassifier": RandomForestClassifier,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "XGBClassifier": XGBClassifier,
        "LogisticRegression": LogisticRegression,
        "AdaBoostClassifier": AdaBoostClassifier,
        "BaggingClassifier": BaggingClassifier,
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
        "MLPClassifier": MLPClassifier,
        "LinearSVC": LinearSVC,
        "RadialSVC": RadialSVC,
        "SigmoidSVC": SigmoidSVC,
        "RidgeClassifier": RidgeClassifier,
        "LogisticRegression_ElasticNet": LogisticRegressionElasticNet,
        "LogisticRegression_Lasso": LogisticRegressionLasso,
        "LGBMClassifier_GBDT": LGBMClassifierGBDT,
        "LGBMClassifier_RF": LGBMClassifierRF,
    }

    def __init__(
        self,
        table,
        metadata,
        algorithm,
        params,
        n_repeats,
        distance_matrix=None,
    ):
        super().__init__(
            table, metadata, algorithm, params, n_repeats, distance_matrix
        )

        kfold = RepeatedStratifiedKFold(
            n_splits=5, n_repeats=self.n_repeats, random_state=2020
        )
        self.splits = kfold.split(X=self.X, y=self.y)

        for n in list(range(self.n_classes)):
            colname = "PROB_CLASS_" + str(n)
            self.results[colname] = np.full(
                self.table_size, fill_value=np.nan, dtype=float
            )
        self.results["ACCURACY"] = np.full(
            self.table_size, fill_value=np.nan, dtype=float
        )
        self.results["AUPRC"] = np.full(
            self.table_size, fill_value=np.nan, dtype=float
        )
        self.results["AUROC"] = np.full(
            self.table_size, fill_value=np.nan, dtype=float
        )
        self.results["F1"] = np.full(
            self.table_size, fill_value=np.nan, dtype=float
        )
        self.results["BALANCED_ACCURACY"] = np.full(
            self.table_size, fill_value=np.nan, dtype=float
        )

    def cv_fold(self, train_index, test_index):
        X_train, X_test = self.X[train_index], self.X[test_index]
        y_train, y_test = self.y[train_index], self.y[test_index]
        y_test_ids = self.metadata.index[test_index]

        # Start timing
        start = time.process_time()
        model = self.learner(**self.params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # End timimg
        end = time.process_time()

        runtime = end - start
        nrows = len(y_pred)
        curr_indices = list(range(self.idx, self.idx + nrows))
        use_probabilities = hasattr(self.learner, "predict_proba")

        if use_probabilities:
            probas = model.predict_proba(X_test)
        else:
            probas = np.nan

        self.results["RUNTIME"][curr_indices] = runtime
        self.results["CV_IDX"][curr_indices] = self.cv_idx
        self.results["Y_PRED"][curr_indices] = y_pred
        self.results["Y_TRUE"][curr_indices] = y_test
        self.results["SAMPLE_ID"][curr_indices] = y_test_ids

        for n in list(range(self.n_classes)):
            colname = "PROB_CLASS_" + str(n)
            if use_probabilities:
                self.results[colname][curr_indices] = probas[:, n]
            else:
                self.results[colname][curr_indices] = np.nan

        if self.contains_nan(y_pred):
            # All null
            self.results["AUPRC"][curr_indices] = np.nan
            self.results["AUROC"][curr_indices] = np.nan
            self.results["ACCURACY"][curr_indices] = np.nan
            self.results["F1"][curr_indices] = np.nan
            self.results["BALANCED_ACCURACY"][curr_indices] = np.nan
        elif not use_probabilities:
            # Just F1, Accuracy, Balanced Accuracy
            acc_score = accuracy_score(
                y_true=y_test, y_pred=y_pred
            )
            self.results["ACCURACY"][curr_indices] = acc_score

            balanced_acc = balanced_accuracy_score(
                y_true=y_test, y_pred=y_pred
            )
            self.results["BALANCED_ACCURACY"][curr_indices] = balanced_acc

            f1 = f1_score(y_test, y_pred)
            self.results["F1"][curr_indices] = f1

            # Others null
            self.results["AUPRC"][curr_indices] = np.nan
            self.results["AUROC"][curr_indices] = np.nan
        elif use_probabilities:
            # All metrics.
            probas = probas[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, probas)
            self.results["AUPRC"][curr_indices] = auc(recall, precision)
            self.results["AUROC"][curr_indices] = roc_auc_score(y_test, probas)

            acc_score = accuracy_score(
                y_true=y_test, y_pred=y_pred
            )
            self.results["ACCURACY"][curr_indices] = acc_score

            balanced_acc = balanced_accuracy_score(
                y_true=y_test, y_pred=y_pred
            )
            self.results["BALANCED_ACCURACY"][curr_indices] = balanced_acc

            f1 = f1_score(y_test, y_pred)
            self.results["F1"][curr_indices] = f1

        self.cv_idx += 1
        self.idx += nrows

        if acc_score > self.best_accuracy:
            self.best_accuracy = acc_score
            self.best_model = model


class RegressionTask(LearningTask):

    algorithms = {
        "KNeighborsRegressor": KNeighborsRegressor,
        "RandomForestRegressor": RandomForestRegressor,
        "GradientBoostingRegressor": GradientBoostingRegressor,
        "XGBRegressor": XGBRegressor,
        "AdaBoostRegressor": AdaBoostRegressor,
        "BaggingRegressor": BaggingRegressor,
        "LGBMRegressor": LGBMRegressor,
        "ExtraTreesRegressor": ExtraTreesRegressor,
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
        "LinearRegression": LinearRegression,
        "LinearSVR": LinearSVR,
        "RidgeRegressor": Ridge,
        "MLPRegressor": MLPRegressor,
        "LinearSVR": LinearSVR,
        "RadialSVR": RadialSVR,
        "SigmoidSVR": SigmoidSVR,
        "ElasticNet": ElasticNet,
        "Lasso": Lasso,
        "LGBMRegressor_GBDT": LGBMRegressorGBDT,
        "LGBMRegressor_RF": LGBMRegressorRF,
    }

    def __init__(
        self,
        table,
        metadata,
        algorithm,
        params,
        n_repeats,
        distance_matrix=None,
    ):
        super().__init__(
            table, metadata, algorithm, params, n_repeats, distance_matrix
        )

        kfold = RepeatedSortedStratifiedKFold(
            n_splits=5, n_repeats=self.n_repeats, random_state=2020
        )
        self.splits = kfold.split(X=self.X, y=self.y)

        self.results["MAE"] = np.full(
            self.table_size, fill_value=np.nan, dtype=float
        )
        self.results["RMSE"] = np.full(
            self.table_size, fill_value=np.nan, dtype=float
        )
        self.results["R2"] = np.full(
            self.table_size, fill_value=np.nan, dtype=float
        )

    def cv_fold(self, train_index, test_index):
        X_train, X_test = self.X[train_index], self.X[test_index]
        y_train, y_test = self.y[train_index], self.y[test_index]
        y_test_ids = self.metadata.index[test_index]

        # Start timing
        start = time.process_time()
        model = self.learner()
        model.set_params(**self.params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # End timimg
        end = time.process_time()

        runtime = end - start
        nrows = len(y_pred)
        curr_indices = list(range(self.idx, self.idx + nrows))

        self.results["RUNTIME"][curr_indices] = runtime
        self.results["CV_IDX"][curr_indices] = self.cv_idx
        self.results["Y_PRED"][curr_indices] = y_pred
        self.results["Y_TRUE"][curr_indices] = y_test
        self.results["SAMPLE_ID"][curr_indices] = y_test_ids

        if not self.contains_nan(y_pred):
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            self.results["RMSE"][curr_indices] = rmse
            r2 = r2_score(y_test, y_pred)
            self.results["R2"][curr_indices] = r2
            mae = mean_absolute_error(y_test, y_pred)
            self.results["MAE"][curr_indices] = mae
        else:
            self.results["RMSE"][curr_indices] = np.nan
            self.results["R2"][curr_indices] = np.nan

        self.cv_idx += 1
        self.idx += nrows
