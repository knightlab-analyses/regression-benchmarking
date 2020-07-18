#!/usr/bin/env python
import json
import os
from sklearn.model_selection import ParameterGrid
import click
from jinja2 import Environment, FileSystemLoader
from q2_mlab import RegressionTask, ClassificationTask, ParameterGrids


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
    '--base_dir', '-b',
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
    base_dir,
    ppn,
    memory,
    wall,
    chunk_size
):
    classifiers = set(RegressionTask.algorithms.keys())
    regressors = set(ClassificationTask.algorithms.keys())
    valid_algorithms = classifiers.union(regressors)
    if algorithm not in valid_algorithms:
        raise ValueError(
            "Unrecognized algorithm passed. Algorithms must be one of the "
            "following: \n" + str(valid_algorithms)
        )

    try:
        algorithm_parameters = ParameterGrids[algorithm]
    except KeyError:
        print(
            f'{algorithm} does not have an implemented grid in '
            'mlab.ParameterGrids'
        )
        raise

    PPN = ppn
    N_REPEATS = repeats
    GB_MEM = memory
    WALLTIME_HRS = wall
    CHUNK_SIZE = chunk_size
    JOB_NAME = "_".join([dataset, preparation, target, algorithm])

    TABLE_FP = os.path.join(
        base_dir, dataset,  preparation, target, "filtered_rarefied_table.qza"
    )
    if not os.path.isdir(TABLE_FP):
        raise FileNotFoundError(
            "Table was not found at the expected path: "
            + TABLE_FP
        )

    METADATA_FP = os.path.join(
        base_dir, dataset, preparation, target, "filtered_metadata.qza"
    )
    if not os.path.isdir(METADATA_FP):
        raise FileNotFoundError(
            "Metadata was not found at the expected path: "
            + TABLE_FP
        )

    RESULTS_DIR = os.path.join(
        base_dir, dataset, preparation, target, algorithm
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
        "_".join([preparation, target, algorithm]) + ".sh"
    )
    print(output_script)
    print(output_from_parsed_template)
    with open(output_script, "w") as fh:
        fh.write(output_from_parsed_template)


if __name__ == "__main__":
    cli()
