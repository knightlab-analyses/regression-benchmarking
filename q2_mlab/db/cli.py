import click
import json
from qiime2 import Artifact
from q2_mlab.db.maint import (
    create as db_create,
    add_from_qza
)


@click.group()
def mlab_db():
    pass


@mlab_db.command()
@click.option(
    '--db',
    help='Path to SQLite database file.',
    type=click.Path(exists=False),
)
@click.option(
    '--echo/--no-echo',
    default=False,
    help='Echo the database log statements',
)
def create(db, echo):
    db_create(db, echo=echo)


@mlab_db.command()
@click.option(
    '--db',
    help='Path to SQLite database file.',
    required=True,
    type=click.Path(exists=True),
)
@click.option(
    '--results-artifact',
    help='Path to Results artifact',
    required=True,
    type=click.Path(exists=True),
)
@click.option(
    '--algorithm',
    help='Name of the algorithm corresponding to these results',
    required=True,
    type=str,
)
@click.option(
    '--parameters',
    help="JSON-formatted string corresponding to the algorithm's parameters",
    required=True,
    type=str,
)
@click.option(
    '--dataset',
    help='Name of the dataset corresponding to these results',
    required=True,
    type=str,
)
@click.option(
    '--target',
    help='Name of the target corresponding to these results',
    required=True,
    type=str,
)
@click.option(
    '--level',
    help='Name corresponding to the sequencing type/level used for these '
         'results',
    required=True,
    type=str,
)
@click.option(
    '--echo/--no-echo',
    default=False,
    help='Echo the database log statements',
)
def add(db, echo, results_artifact, parameters, dataset, target, level,
        algorithm):
    artifact = Artifact.load(results_artifact)
    parameters = json.loads(parameters)
    add_from_qza(
        artifact=artifact,
        parameters=parameters,
        dataset=dataset,
        target=target,
        level=level,
        algorithm=algorithm,
        db_file=db,
        echo=echo,
    )
