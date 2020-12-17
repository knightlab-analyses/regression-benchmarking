import numpy as np
from sklearn.model_selection import ParameterGrid
from itertools import product


class ParameterGrids:

    lr_alphas = [10 ** (-1 * i) for i in range(10, 1, -1)] + \
                list(np.arange(0.05, 1, 0.05))
    l1_ratios = list(np.arange(0.05, 1, 0.05))
    random_state = [2020]
    n_jobs = [-1]
    silent = [1]
    n_estimators = [100, 1000, 5000]
    max_depth = [None, 3, 7, 10]

    mlp_hidden_layer_grids = []
    hidden_layer_sizes = [10, 50, 100, 200, 500]
    for r in range(1, 4):
        for comb in product(hidden_layer_sizes, repeat=r):
            mlp_hidden_layer_grids.append(comb)

    # Notes:
    # fit_intercept - set to True in all cases due to statistical properties
    # of the data (not necessarily centered)
    # n_estimators - higher n_etimators is known correlate with performnce but
    # begins to show diminishing returns at values >1000 for datasets of this
    # dimensionality
    full_grids = {
        "RandomForestClassifier": {
            "n_estimators": n_estimators,
            "criterion": ["gini", "entropy"],
            "max_features": (
                ["sqrt", "log2", None] + list(np.arange(0.2, 1, 0.2))
            ),
            "max_samples": [0.25, 0.5, 0.75, None],
            "max_depth": max_depth,
            "n_jobs": n_jobs,
            "random_state": random_state,
            "bootstrap": [True, False],
            "min_samples_split": list(np.arange(0.2, 1, 0.2)) + [2],
            "min_samples_leaf": list(np.arange(0.01, 0.5, 0.2)) + [1],
        },
        "RandomForestRegressor": {
            "n_estimators": n_estimators,
            "criterion": ["mse", "mae"],
            "max_features": (
                ["sqrt", "log2", None] + list(np.arange(0.2, 1, 0.2))
            ),
            "max_samples": [0.25, 0.5, 0.75, None],
            "max_depth": max_depth,
            "n_jobs": n_jobs,
            "random_state": random_state,
            "bootstrap": [True, False],
            "min_samples_split": list(np.arange(0.2, 1, 0.2)) + [2],
            "min_samples_leaf": list(np.arange(0.01, 0.5, 0.2)) + [1],
        },
        "GradientBoostingRegressor": {
            "loss": ["ls", "lad", "huber", "quantile"],
            "alpha": [1e-3, 1e-2, 1e-1, 0.5, 0.9],
            "learning_rate": [3e-1, 2e-1, 1e-1, 5e-2],
            "n_estimators": n_estimators,
            "criterion": ["friedman_mse", "mse" "mae"],
            "max_features": [None, "sqrt", "log2", 0.2, 0.4, 0.6, 0.8],
            "max_depth": max_depth,
            "random_state": random_state,
            "n_iter_no_change": [10]
        },
        "GradientBoostingClassifier": {
            "loss": ["deviance", "exponential"],
            "alpha": [1e-3, 1e-2, 1e-1, 0.5, 0.9],
            "learning_rate": [3e-1, 2e-1, 1e-1, 1e-2],
            "n_estimators": n_estimators,
            "criterion": ["friedman_mse", "mse" "mae"],
            "max_features": [None, "sqrt", "log2", 0.2, 0.4, 0.6, 0.8],
            "max_depth": max_depth,
            "random_state": random_state,
            "n_iter_no_change": [10]
        },
        "XGBRegressor": {
            "max_depth": max_depth,
            "learning_rate": [2e-1, 1e-2, 1e-3],
            "n_estimators": n_estimators,
            "objective": ["reg:squarederror"],
            "subsample": [0.5, 0.75, 1.0],
            "min_child_weight": [1, 5, 7],
            "colsample_bytree": [0.5, 0.75, 1.0],
            "booster": ["gbtree"],
            "gamma": [0.01, 0.5, 2],
            "reg_alpha": [0, 0.5, 1],
            "reg_lambda": [1e-1, 1, 5],
            "random_state": random_state,
            "silent": silent,
            "n_jobs": n_jobs,
        },
        "XGBClassifier": {
            "max_depth": max_depth,
            "learning_rate": [2e-1, 1e-2, 1e-3],
            "n_estimators": n_estimators,
            "objective": ["binary:logistic"],
            "subsample": [0.5, 0.75, 1.0],
            "min_child_weight": [1, 5, 7],
            "colsample_bytree": [0.5, 0.75, 1.0],
            "booster": ["gbtree"],
            "gamma": [0.01, 0.5, 2],
            "reg_alpha": [0, 0.5, 1],
            "reg_lambda": [1e-1, 1, 5],
            "random_state": random_state,
            "silent": silent,
            "n_jobs": n_jobs,
        },
        "LGBMRegressor_GBDT": {
            "num_leaves": [8, 31, 90, 900],
            "max_depth": max_depth,
            "learning_rate": [2e-1, 1e-2, 1e-3, 1e-4],
            "n_estimators": n_estimators,
            "min_child_weight": [0.001, 0.1, 1, 7],
            "subsample": [0.5, 0.75, 1.0],
            "colsample_bytree": [0.5, 0.75, 1.0],
            "reg_alpha": [0, 0.5, 1],
            "reg_lambda": [1e-1, 1, 5],
            "random_state": random_state,
            "silent": silent,
            "n_jobs": n_jobs,
        },
        "LGBMRegressor_RF": {
            "num_leaves": [8, 31, 90, 900],
            "max_depth": max_depth,
            "learning_rate": [2e-1, 1e-2, 1e-3, 1e-4],
            "n_estimators": n_estimators,
            "min_child_weight": [0.001, 0.1, 1, 7],
            "subsample": [0.5, 0.75, 1.0],
            "colsample_bytree": [0.5, 0.75, 1.0],
            "reg_alpha": [0, 0.5, 1],
            "reg_lambda": [1e-1, 1, 5],
            "random_state": random_state,
            "silent": silent,
            "n_jobs": n_jobs,
        },
        "LGBMClassifier_GBDT": {
            "num_leaves": [8, 31, 90, 900],
            "max_depth": max_depth,
            "learning_rate": [2e-1, 1e-2, 1e-3, 1e-4],
            "n_estimators": n_estimators,
            "min_child_weight": [0.001, 0.1, 1, 7],
            "subsample": [0.5, 0.75, 1.0],
            "colsample_bytree": [0.5, 0.75, 1.0],
            "reg_alpha": [0, 0.5, 1],
            "reg_lambda": [1e-1, 0, 1, 5],
            "random_state": random_state,
            "silent": silent,
            "n_jobs": n_jobs,
        },
        "LGBMClassifier_RF": {
            "num_leaves": [8, 31, 90, 900],
            "max_depth": max_depth,
            "learning_rate": [2e-1, 1e-2, 1e-3, 1e-4],
            "n_estimators": n_estimators,
            "min_child_weight": [0.001, 0.1, 1, 7],
            "subsample": [0.5, 0.75, 1.0],
            "colsample_bytree": [0.5, 0.75, 1.0],
            "reg_alpha": [0, 0.5, 1],
            "reg_lambda": [1e-1, 0, 1, 5],
            "random_state": random_state,
            "silent": silent,
            "n_jobs": n_jobs,
        },
        "HistGradientBoostingRegressor": {
            "loss": ["poisson", "least_squares", "least_absolute_deviation"],
            "learning_rate": [2e-1, 1e-2, 1e-3, 1e-4],
            "max_leaf_nodes": [8, 31, 90, 900],
            "max_depth": max_depth,
            "l2_regularization": [1e-1, 0, 1, 5],
            "early_stopping": [True, False],
            "random_state": random_state,
        },
        "HistGradientBoostingClassifier": {
            "learning_rate": [2e-1, 1e-2, 1e-3, 1e-4],
            "max_leaf_nodes": [8, 31, 90, 900],
            "max_depth": max_depth,
            "l2_regularization": [1e-1, 0, 1, 5],
            "early_stopping": [True, False],
            "random_state": random_state,
        },
        "AdaBoostRegressor": {
            "n_estimators": n_estimators,
            "learning_rate": [2e-1, 1e-2, 1e-3],
            "loss": ["linear", "square", "exponential"],
            "random_state": random_state,
        },
        "AdaBoostClassifier": {
            "n_estimators": n_estimators,
            "learning_rate": [2e-1, 1e-2, 1e-3],
            "algorithm": ["SAMME", "SAMME.R"],
            "random_state": random_state,
        },
        "ExtraTreesClassifier": {
            "n_estimators": n_estimators,
            "criterion": ["gini", "entropy"],
            "max_features": (
                ["sqrt", "log2", None] + list(np.arange(0.2, 1, 0.2))
            ),
            "max_samples": [0.25, 0.5, 0.75, None],
            "max_depth": max_depth,
            "n_jobs": n_jobs,
            "random_state": random_state,
            "bootstrap": [True],
            "min_samples_split": list(np.arange(0.2, 1, 0.2)) + [2],
            "min_samples_leaf": list(np.arange(0.01, 0.5, 0.2)) + [1],
        },
        "ExtraTreesRegressor": {
            "n_estimators": n_estimators,
            "criterion": ["mse", "mae"],
            "max_features": (
                ["sqrt", "log2", None] + list(np.arange(0.2, 1, 0.2))
            ),
            "max_samples": [0.25, 0.5, 0.75, None],
            "max_depth": max_depth,
            "n_jobs": n_jobs,
            "random_state": random_state,
            "bootstrap": [True],
            "min_samples_split": list(np.arange(0.2, 1, 0.2)) + [2],
            "min_samples_leaf": list(np.arange(0.01, 0.5, 0.2)) + [1],
        },
        "LinearSVC": {
            "penalty": ["l2"],
            "tol": [1e-4, 1e-3, 1e-2, 1e-1],
            "loss": ["hinge", "squared_hinge"],
            "max_iter": [1000, 5000],
            "random_state": random_state,
        },
        "LinearSVR": {
            "C": [1e-4, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e4],
            "epsilon": [1e-2, 1e-1, 0, 1],
            "loss": ["squared_epsilon_insensitive", "epsilon_insensitive"],
            "max_iter": [1000, 5000],
            "random_state": random_state,
        },
        "RadialSVR": {
            "C": [1e-4, 1e-3, 1e-2, 1e-1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7],
            "epsilon": [1e-5, 1e-4, 1e-3, 1e-2],
            "gamma": ["scale", "auto", 100, 10, 1, 1e-2, 1e-3, 1e-4, 1e-5],
        },
        "RadialSVC": {
            "C": [1e-4, 1e-3, 1e-2, 1e-1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7],
            "gamma": ["scale", "auto", 100, 10, 1, 1e-2, 1e-3, 1e-4, 1e-5],
        },
        "SigmoidSVR": {
            "C": [1e-4, 1e-3, 1e-2, 1e-1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7],
            "epsilon": [1e-5, 1e-4, 1e-3, 1e-2],
            "gamma": ["scale", "auto", 100, 10, 1, 1e-2, 1e-3, 1e-4, 1e-5],
            "coef0": [0, 1, 10, 100],
        },
        "SigmoidSVC": {
            "C": [1e-4, 1e-3, 1e-2, 1e-1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7],
            "gamma": ["scale", "auto", 100, 10, 1, 1e-2, 1e-3, 1e-4, 1e-5],
            "coef0": [0, 1, 10, 100],
        },
        "RidgeClassifier": [
            {
                "alpha": lr_alphas,
                "fit_intercept": [True],
                "normalize": [True, False],
                "tol": [1e-1, 1e-2, 1e-3],
                "solver": ["sparse_cg", "saga"],
                "random_state": random_state,
            },
            {
                "alpha": lr_alphas,
                "fit_intercept": [False],
                "normalize": [True, False],
                "tol": [1e-1, 1e-2, 1e-3],
                "solver": ["svd", "cholesky", "lsqr"],
                "random_state": random_state,
            },
        ],
        "RidgeRegressor": [
            {
                "alpha": lr_alphas,
                "fit_intercept": [True],
                "normalize": [True, False],
                "tol": [1e-1, 1e-2, 1e-3],
                "solver": ["sparse_cg", "saga"],
                "random_state": random_state,
            },
            {
                "alpha": lr_alphas,
                "fit_intercept": [False],
                "normalize": [True, False],
                "tol": [1e-1, 1e-2, 1e-3],
                "solver": ["svd", "cholesky", "lsqr"],
                "random_state": random_state,
            },
        ],
        "ElasticNet": {
            "alpha": lr_alphas,
            "l1_ratio": l1_ratios,
            "normalize": [True, False],
            "positive": [True, False],
            "random_state": random_state,
        },
        "Lasso": {
            "alpha": lr_alphas,
            "normalize": [True, False],
            "random_state": random_state,
        },
        "LinearRegression": {
            "fit_intercept": [True, False],
            "normalize": [True, False],
        },
        "LogisticRegression_Lasso": {
            # these are the only solvers that support l1 penalty
            "solver": ["saga", "liblinear"],
            "C": [1 / alpha for alpha in lr_alphas],
            "random_state": random_state,
        },
        "LogisticRegression_ElasticNet": {
            # this is the only solver that supports elasticnet penalty
            "solver": ["saga"],
            "C": [1 / alpha for alpha in lr_alphas],
            "l1_ratio": l1_ratios,
            "random_state": random_state,
        },
        "LogisticRegression": {
            "penalty": ["none"],
            "fit_intercept": [True, False],
            "random_state": random_state,
        },
        "KNeighborsRegressor": {
            "n_neighbors": list(range(1, 50)),
            "weights": ["uniform", "distance"],
            "metric": ["braycurtis", "jaccard"],
        },
        "KNeighborsClassifier": {
            "n_neighbors": list(range(1, 50)),
            "weights": ["uniform", "distance"],
            "metric": ["braycurtis", "jaccard"],
        },
        "MLPClassifier": {
            "hidden_layer_sizes": mlp_hidden_layer_grids,
            "solver": ["adam", "sgd", "lbfgs"],
            "activation": ["tanh", "relu", "logistic", "identity"],
            "learning_rate": ["constant", "adaptive", "invscaling"],
            "random_state": random_state,
        },
        "MLPRegressor": {
            "hidden_layer_sizes": mlp_hidden_layer_grids,
            "solver": ["adam", "sgd", "lbfgs"],
            "activation": ["tanh", "relu", "logistic", "identity"],
            "learning_rate": ["constant", "adaptive", "invscaling"],
            "random_state": random_state,
        },
    }

    reduced_ensemble_grids = {
        "RandomForestClassifier": {
            "n_estimators": [10, 100, 1000],
            "criterion": ["gini"],
            "max_features": ["sqrt", "log2", None, 0.4, 0.6],
            "max_samples": [0.25, 0.5, 0.75, None],
            "max_depth": [None, 10, 100],
            "n_jobs": n_jobs,
            "random_state": random_state,
            "bootstrap": [True],
        },
        "RandomForestRegressor": {
            "n_estimators": [10, 100, 1000],
            "criterion": ["mse"],
            "max_features": ["sqrt", "log2", None, 0.4, 0.6],
            "max_samples": [0.25, 0.5, 0.75, None],
            "max_depth": [None, 10, 100],
            "n_jobs": n_jobs,
            "random_state": random_state,
            "bootstrap": [True],
        },
        "GradientBoostingRegressor": {
            "loss": ["ls", "lad", "huber", "quantile"],
            "alpha": [1e-3, 1e-1, 0.5, 0.9],
            "learning_rate": [3e-1, 1e-1, 5e-2],
            "n_estimators": [1000, 5000],
            "criterion": ["mse"],
            "max_features": [None, "sqrt", "log2", 0.4, 0.6],
            "max_depth": [None, 10, 100],
            "random_state": random_state,
        },
        "GradientBoostingClassifier": {
            "loss": ["deviance", "exponential"],
            "learning_rate": [3e-1, 1e-1, 5e-2],
            "n_estimators": [1000, 5000],
            "criterion": ["mse"],
            "max_features": [None, "sqrt", "log2", 0.4, 0.6],
            "max_depth": [None, 10, 100],
            "random_state": random_state,
        },
        "XGBRegressor": {
            "max_depth": [None, 10, 100],
            "learning_rate": [3e-1, 2e-1, 1e-1, 5e-2],
            "n_estimators": [5000],
            "objective": ["reg:linear"],
            "booster": ["gbtree"],
            "gamma": [0],
            "reg_alpha": [1e-3, 1e-1, 1],
            "reg_lambda": [1e-3, 1e-1, 1],
            "random_state": random_state,
            "silent": [1],
            "n_jobs": n_jobs,
        },
        "XGBClassifier": {
            "max_depth": [None, 10, 100],
            "learning_rate": [3e-1, 2e-1, 1e-1, 5e-2],
            "n_estimators": [1000, 5000],
            "objective": ["reg:linear"],
            "booster": ["gbtree"],
            "gamma": [0],
            "reg_alpha": [1e-3, 1e-1, 1],
            "reg_lambda": [1e-3, 1e-1, 1],
            "random_state": random_state,
            "silent": [1],
            "n_jobs": n_jobs,
        },
        "ExtraTreesClassifier": {
            "n_estimators": [5000],
            "criterion": ["gini"],
            "max_features": [None, "sqrt", "log2", 0.4, 0.6],
            "max_samples": [0.25, 0.5, 0.75, None],
            "max_depth": [None],
            "n_jobs": n_jobs,
            "random_state": random_state,
            "bootstrap": [True],
        },
        "ExtraTreesRegressor": {
            "n_estimators": [5000],
            "criterion": ["mse"],
            "max_features": [None, "sqrt", "log2", 0.4, 0.6],
            "max_samples": [0.25, 0.5, 0.75, None],
            "max_depth": [None],
            "n_jobs": [-1],
            "random_state": random_state,
            "bootstrap": [True],
        },
    }

    def get_size(algorithm, reduced=False):
        if reduced:
            grid = ParameterGrids.get_reduced(algorithm)
        else:
            grid = ParameterGrids.get(algorithm)
        size = len(list(ParameterGrid(grid)))
        return size

    def get(algorithm):
        return ParameterGrids.full_grids[algorithm]

    def get_reduced(algorithm):
        # Updates the full parameter grids with reduced grids for just
        # the ensemble methods
        fullgrid_shallow_copy = ParameterGrids.full_grids.copy()
        fullgrid_shallow_copy.update(
            ParameterGrids.reduced_ensemble_grids
        )
        return fullgrid_shallow_copy[algorithm]
