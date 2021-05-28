#!/usr/bin/env python
import json
import math
import click
import random
from os import path, makedirs
from sklearn.model_selection import ParameterGrid
from jinja2 import Environment, FileSystemLoader
from q2_mlab import RegressionTask, ClassificationTask, ParameterGrids


def orchestrate_hyperparameter_search(
    dataset,
    preparation,
    target,
    algorithm,
    base_dir,
    repeats=3,
    ppn=1,
    memory=32,
    wall=50,
    chunk_size=100,
    randomize=True,
    reduced=False,
    force=False,
    dry=False,
    table_path=None,
    metadata_path=None,
):
    """
    Creates files and file structures necessary for a hyperparameter search
    in q2_mlab and returns paths to those files, which include the job script,
    list of hyperparameter sets, and some info about the search.
    Orchestrator does not enforce anything about the dataset, metadata, and
    target and assumes these are given from Preprocessing or similar steps.

            Parameters:
                    dataset (str): Name of dataset
                    preparation (str): Name of data type/preparation (e.g. 16S)
                    target (str): Name of the target variable in the metadata
                    algorithm (str): Valid algorithm included in q2_mlab
                    base_dir (str): The directory in which create the file structure.
                    table_path (str): Specify exact path to feature table, if it cannot be assumed. Default=None
                    metadata_path (str): Specify exact path to metadata, if it cannot be assumed. Default=None
                    repeats (int): Number of CV repeats. Default=3
                    ppn: Processors per node for job script. Default=1
                    memory: GB of memory for job script.Default=32
                    wall: Walltime in hours for job script. Default=50
                    chunk_size: DNumber of params to run in one job for job script. Default=100
                    randomize: Randomly shuffle the order of the hyperparameter list. Default=True
                    reduced: If a reduced parameter grid is available, run the reduced grid. Default=False
                    force: Overwrite existing results. Default=False
                    dry: Perform a dry run without writing files. Default=False

            Returns:
                    output_script (str): Filepath to the created job script
                    PARAMS_FP (str): Filepath to the created hyperparameter list
                    info_doc (str): Filepath to the created job info text file
    """
    classifiers = set(RegressionTask.algorithms.keys())
    regressors = set(ClassificationTask.algorithms.keys())
    valid_algorithms = classifiers.union(regressors)
    ALGORITHM = algorithm
    if ALGORITHM not in valid_algorithms:
        raise ValueError(
            "Unrecognized algorithm passed. Algorithms must be one of the "
            "following: \n" + str(valid_algorithms)
        )

    try:
        algorithm_parameters = ParameterGrids.get(ALGORITHM)
    except KeyError:
        print(
            f"{ALGORITHM} does not have a grid implemented in "
            "mlab.ParameterGrids"
        )
        raise

    PPN = ppn
    N_REPEATS = repeats
    GB_MEM = memory
    WALLTIME_HRS = wall
    CHUNK_SIZE = chunk_size
    JOB_NAME = "_".join([dataset, preparation, target, ALGORITHM])
    FORCE = str(force).lower()

    # Use user-specified path, otherwise assume path from Preprocessing
    if table_path:
        TABLE_FP = table_path
    else:
        TABLE_FP = path.join(
            base_dir,
            dataset,
            preparation,
            target,
            "filtered_rarefied_table.qza",
        )
    if not path.exists(TABLE_FP):
        raise FileNotFoundError(
            "Table was not found at the expected path: " + TABLE_FP
        )

    if metadata_path:
        METADATA_FP = metadata_path
    else:
        METADATA_FP = path.join(
            base_dir, dataset, preparation, target, "filtered_metadata.qza"
        )
    if not path.exists(METADATA_FP):
        raise FileNotFoundError(
            "Metadata was not found at the expected path: " + METADATA_FP
        )

    RESULTS_DIR = path.join(base_dir, dataset, preparation, target, ALGORITHM)
    if not path.isdir(RESULTS_DIR):
        makedirs(RESULTS_DIR)

    BARNACLE_OUT_DIR = path.join(base_dir, dataset, "barnacle_output/")
    if not path.isdir(BARNACLE_OUT_DIR):
        makedirs(BARNACLE_OUT_DIR)

    params = list(ParameterGrid(algorithm_parameters))
    params_list = [json.dumps(param_dict) for param_dict in params]
    PARAMS_FP = path.join(RESULTS_DIR, ALGORITHM + "_parameters.txt")
    N_PARAMS = len(params_list)
    N_CHUNKS = math.ceil(N_PARAMS / CHUNK_SIZE)
    REMAINDER = N_PARAMS % CHUNK_SIZE

    random.seed(2021)
    if randomize:
        random.shuffle(params_list)

    mlab_dir = path.dirname(path.abspath(__file__))
    env = Environment(
        loader=FileSystemLoader(path.join(mlab_dir, "templates"))
    )
    job_template = env.get_template("array_job_template.sh")
    info_template = env.get_template("info.txt")

    output_from_job_template = job_template.render(
        JOB_NAME=JOB_NAME,
        STD_ERR_OUT=BARNACLE_OUT_DIR,
        PPN=PPN,
        GB_MEM=GB_MEM,
        WALLTIME_HRS=WALLTIME_HRS,
        PARAMS_FP=PARAMS_FP,
        CHUNK_SIZE=CHUNK_SIZE,
        TABLE_FP=TABLE_FP,
        METADATA_FP=METADATA_FP,
        ALGORITHM=ALGORITHM,
        N_REPEATS=N_REPEATS,
        RESULTS_DIR=RESULTS_DIR,
        FORCE_OVERWRITE=FORCE,
    )

    output_from_info_template = info_template.render(
        PARAMS_FP=PARAMS_FP,
        CHUNK_SIZE=CHUNK_SIZE,
        N_PARAMS=N_PARAMS,
        REMAINDER=REMAINDER,
        N_CHUNKS=N_CHUNKS,
    )

    output_script = path.join(
        base_dir, dataset, "_".join([preparation, target, ALGORITHM]) + ".sh"
    )

    info_doc = path.join(
        base_dir,
        dataset,
        "_".join([preparation, target, ALGORITHM]) + "_info.txt",
    )

    if dry:
        print(output_from_job_template)
        print("##########################")
        print(f"Number of parameters: {len(params_list)}")
        print(f"Max number of jobs with chunk size {CHUNK_SIZE}: {N_CHUNKS}")
        print(f"Will save info to: {info_doc}")
        print(f"Will save params to: {PARAMS_FP}")
        print(f"Will save script to: {output_script}")
    else:
        with open(info_doc, "w") as fh:
            fh.write(output_from_info_template)
        with open(PARAMS_FP, "w") as fh:
            for i, p in enumerate(params_list, 1):
                fh.write(str(i).zfill(8) + "\t" + p + "\n")
        with open(output_script, "w") as fh:
            fh.write(output_from_job_template)

    return output_script, PARAMS_FP, info_doc


@click.command()
@click.argument("dataset")
@click.argument("preparation")
@click.argument("target")
@click.argument("algorithm")
@click.option(
    "--base_dir",
    "-b",
    help="Directory to search for datasets in",
    required=True,
)
@click.option(
    "--repeats",
    "-r",
    default=3,
    show_default=True,
    help="Number of CV repeats",
)
@click.option(
    "--ppn",
    default=1,
    show_default=True,
    help="Processors per node for job script",
)
@click.option(
    "--memory",
    default=32,
    show_default=True,
    help="GB of memory for job script",
)
@click.option(
    "--wall",
    default=50,
    show_default=True,
    help="Walltime in hours for job script",
)
@click.option(
    "--chunk_size",
    default=100,
    show_default=True,
    help="Number of params to run in one job for job script",
)
@click.option(
    "--randomize/--no-randomize",
    default=True,
    show_default=True,
    help="Randomly shuffle the order of the hyperparameter list",
)
@click.option(
    "--reduced/--no-reduced",
    default=False,
    show_default=True,
    help="If a reduced parameter grid is available, run the reduced grid.",
)
@click.option(
    "--force/--no-force",
    default=False,
    show_default=True,
    help="Overwrite existing results.",
)
@click.option(
    "--dry/--wet",
    default=False,
    show_default=True,
    help="Perform a dry run without writing files.",
)
@click.option(
    "--table-path",
    help="Path to feature table artifact from preprocess.",
)
@click.option(
    "--metadata-path",
    help="Path to target metadata artifact from preprocess.",
)
def cli(
    dataset,
    preparation,
    target,
    algorithm,
    base_dir,
    repeats,
    ppn,
    memory,
    wall,
    chunk_size,
    randomize,
    reduced,
    force,
    dry,
    table_path,
    metadata_path,
):
    orchestrate_hyperparameter_search(
        dataset=dataset,
        preparation=preparation,
        target=target,
        algorithm=algorithm,
        base_dir=base_dir,
        repeats=repeats,
        ppn=ppn,
        memory=memory,
        wall=wall,
        chunk_size=chunk_size,
        randomize=randomize,
        reduced=reduced,
        force=force,
        dry=dry,
        table_path=table_path,
        metadata_path=metadata_path,
    )


if __name__ == "__main__":
    cli()
