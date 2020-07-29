#!/usr/bin/env python
import json
import math
import click
import random
from os import path, makedirs
from sklearn.model_selection import ParameterGrid
from jinja2 import Environment, FileSystemLoader
from q2_mlab import RegressionTask, ClassificationTask, ParameterGrids


@click.command()
@click.argument('dataset')
@click.argument('preparation')
@click.argument('target')
@click.argument('algorithm',)
@click.option(
    '--base_dir', '-b',
    default="./processed/",
    help="Directory to search for datasets in",
)
@click.option(
    '--repeats', '-r',
    default=3,
    help="Number of CV repeats",
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
    default=50,
    help="Walltime in hours for job script",
)
@click.option(
    '--chunk_size',
    default=100,
    help="Number of params to run in one job for job script",
)
@click.option(
    '--randomize/--no-randomize',
    default=True,
    help="Randomly shuffle the order of the hyperparameter list",
)
@click.option(
    '--reduced/--no-reduced',
    default=False,
    help="If a reduced parameter grid is available, run the reduced grid.",
)
def cli(
    dataset,
    preparation,
    target,
    algorithm,
    repeats,
    base_dir,
    ppn,
    memory,
    wall,
    chunk_size,
    randomize,
    reduced
):
    classifiers = set(RegressionTask.algorithms.keys())
    regressors = set(ClassificationTask.algorithms.keys())
    valid_algorithms = classifiers.union(regressors)
    if algorithm not in valid_algorithms:
        raise ValueError(
            "Unrecognized algorithm passed. Algorithms must be one of the "
            "following: \n" + str(valid_algorithms)
        )

    if reduced:
        try:
            algorithm_parameters = ParameterGrids.get_reduced(algorithm)
        except KeyError:
            print(
                f'{algorithm} does not have a reduced grid implemented grid '
                'in mlab.ParameterGrids'
            )
            raise

    else:
        try:
            algorithm_parameters = ParameterGrids.get(algorithm)
        except KeyError:
            print(
                f'{algorithm} does not have a grid implemented in '
                'mlab.ParameterGrids'
            )
            raise

    PPN = ppn
    N_REPEATS = repeats
    GB_MEM = memory
    WALLTIME_HRS = wall
    CHUNK_SIZE = chunk_size
    JOB_NAME = "_".join([dataset, preparation, target, algorithm])

    TABLE_FP = path.join(
        base_dir, dataset, preparation, target, "filtered_rarefied_table.qza"
    )
    if not path.exists(TABLE_FP):
        raise FileNotFoundError(
            "Table was not found at the expected path: "
            + TABLE_FP
        )

    METADATA_FP = path.join(
        base_dir, dataset, preparation, target, "filtered_metadata.qza"
    )
    if not path.exists(METADATA_FP):
        raise FileNotFoundError(
            "Metadata was not found at the expected path: "
            + TABLE_FP
        )

    RESULTS_DIR = path.join(
        base_dir, dataset, preparation, target, algorithm
    )
    if not path.isdir(RESULTS_DIR):
        makedirs(RESULTS_DIR)

    barnacle_out_dir = path.join(base_dir, dataset, "barnacle_output/")
    if not path.isdir(barnacle_out_dir):
        makedirs(barnacle_out_dir)

    params = list(ParameterGrid(algorithm_parameters))
    params_list = [json.dumps(param_dict) for param_dict in params]
    PARAMS_FP = path.join(RESULTS_DIR, algorithm + "_parameters.txt")

    random.seed(2021)
    if randomize:
            random.shuffle(params_list)

    with open(PARAMS_FP, 'w') as f:
        i = 1
        for p in params_list:
            f.write(str(i).zfill(4)+"\t"+p+"\n")
            i += 1

    mlab_dir = path.dirname(path.abspath(__file__))
    env = Environment(
        loader=FileSystemLoader(path.join(mlab_dir, 'templates'))
    )
    template = env.get_template('array_job_template.sh')

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

    output_script = path.join(
        base_dir,
        dataset,
        "_".join([preparation, target, algorithm]) + ".sh"
    )
    print(output_script)
    print(output_from_parsed_template)
    print("##########################")
    print("Saved to: " + output_script)
    print("Saved params to: "+PARAMS_FP)
    print("Number of parameters: " + str(len(params_list)))
    print(f"Max number of jobs with chunk size {CHUNK_SIZE}: " + str(math.floor(len(params_list)/chunk_size)+1))
    with open(output_script, "w") as fh:
        fh.write(output_from_parsed_template)


if __name__ == "__main__":
    cli()
