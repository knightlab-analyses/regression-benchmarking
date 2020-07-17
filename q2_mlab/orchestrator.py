#!/usr/bin/env python 
import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import ParameterGrid
import click
from jinja2 import Environment, FileSystemLoader
import q2_mlab

LinearSVC_grid = {
    'penalty': ['l2'],
    'tol': [1e-4, 1e-3, 1e-2, 1e-1],
    'loss': ['hinge', 'squared_hinge'],
    'random_state': [2018]
}

LinearSVR_grid = {
    'C': [1e-4, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e4],
    'epsilon': [1e-2, 1e-1, 0, 1],
    'loss': ['squared_epsilon_insensitive', 'epsilon_insensitive'],
    'random_state': [2018]
}

RidgeClassifier_grid = {
    'alpha': [1e-15, 1e-10, 1e-8, 1e-4],
    'fit_intercept': [True],
    'normalize':  [True, False],
    'tol': [1e-1, 1e-2, 1e-3],
    'solver': ['svd', 'cholesky', 'lsqr',
               'sparse_cg', 'sag', 'saga'],
    'random_state': [2018]
}

RidgeRegressor_grid = {
    'alpha': [1e-15, 1e-10, 1e-8, 1e-4],
    'fit_intercept': [True],
    'normalize': [True, False],
    'tol': [1e-1, 1e-2, 1e-3],
    'solver': ['svd', 'cholesky', 'lsqr',
               'sparse_cg', 'sag', 'saga'],
    'random_state': [2018]
}

RandomForestClassifier_grid = {
    'n_estimators': [1000, 5000],
    'criterion': ['gini', 'entropy'],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None],
    'n_jobs': [-1],
    'random_state': [2018],
    'bootstrap': [True, False],
    'min_samples_split': list(np.arange(0.01, 1, 0.2)),
    'min_samples_leaf': list(np.arange(0.01, .5, 0.1)) + [1],
}

RandomForestRegressor_grid = {
    'n_estimators': [1000, 5000],
    'criterion': ['mse'],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None],
    'n_jobs': [-1],
    'random_state': [2018],
    'bootstrap': [True, False],
    'min_samples_split': list(np.arange(0.01, 1, 0.2)),
    'min_samples_leaf': list(np.arange(0.01, .5, 0.1)) + [1],
}

classifiers = set(q2_mlab.RegressionTask.algorithms.keys())
regressors = set(q2_mlab.ClassificationTask.algorithms.keys())
valid_algorithms = classifiers.union(regressors)

@click.command()
@click.argument('dataset')
@click.argument('preparation')
@click.argument('target')
@click.argument('algorithm',)
@click.option(
    '--repeats', '-r',
    default=3,
    help="Number of CV repeats",
)
@click.option(
    '--basedir', '-b',
    default="/projects/ibm_aihl/ML-benchmark/processed/",
    help="Directory to search for datasets in",
)
@click.option(
    '--ppn',
    default=1,
    help="Processors per node for job script",
)
@click.option(
    '--memory',
    default=32,
    help="GB of memory for job script",
)
@click.option(
    '--wall',
    default=10,
    help="Walltime in hours for job script",
)
@click.option(
    '--chunk_size',
    default=100,
    help="Number of params to run in one job for job script",
)
def cli(
    dataset,
    preparation,
    target,
    algorithm,
    repeats,
    basedir,
    ppn,
    memory,
    wall,
    chunk_size
):
    datatype = preparation
    base_dir = basedir
    algorithm_parameters = LinearSVC_grid
    PPN = ppn
    N_REPEATS = repeats
    GB_MEM = memory
    WALLTIME_HRS = wall
    CHUNK_SIZE = chunk_size
    JOB_NAME = "_".join([dataset, preparation, target, algorithm])

    TABLE_FP = os.path.join(
        base_dir, dataset,  datatype, target, "filtered_rarefied_table.qza"
    )
    METADATA_FP = os.path.join(
        base_dir, dataset, datatype, target, "filtered_metadata.qza"
    )
    RESULTS_DIR = os.path.join(
        base_dir, dataset, datatype, target, algorithm
    )

    if not os.path.isdir(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    barnacle_out_dir = os.path.join(base_dir, dataset, "barnacle_output/")
    if not os.path.isdir(barnacle_out_dir):
        os.makedirs(barnacle_out_dir)

    params_list = list(ParameterGrid(algorithm_parameters))
    params_list = [json.dumps(params_list[i]) for i in range(len(params_list))]

    PARAMS_FP = os.path.join(RESULTS_DIR, algorithm + "_parameters.txt")

    if not os.path.exists(PARAMS_FP):
        print("saving params")
        with open(PARAMS_FP, 'w') as f:
            i = 1
            for p in params_list:
                f.write(str(i)+"\t"+p+"\n")
                i += 1
    print("Number of parameters: " + str(len(params_list)))

    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template('array_job_template2.sh')

    output_from_parsed_template = template.render(
        JOB_NAME=JOB_NAME,
        STD_ERR_OUT=barnacle_out_dir,
        PPN=PPN,
        GB_MEM=GB_MEM,
        WALLTIME_HRS=WALLTIME_HRS,
        PARAMS_FP=PARAMS_FP,
        CHUNK_SIZE=CHUNK_SIZE,
        TABLE_FP=TABLE_FP,
        METADATA_FP=METADATA_FP,
        ALGORITHM=algorithm,
        N_REPEATS=N_REPEATS,
        RESULTS_DIR=RESULTS_DIR
    )

    output_script = os.path.join(
        base_dir,
        dataset,
        "_".join([datatype, target, algorithm]) + ".sh"
    )
    print(output_script)
    print(output_from_parsed_template)
    with open(output_script, "w") as fh:
        fh.write(output_from_parsed_template)

if __name__ == "__main__":
    cli()