import json
import pandas as pd
from sqlalchemy.engine import Engine
from sqlalchemy import create_engine as sql_create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from qiime2 import Artifact
from q2_mlab.db.schema import RegressionScore, Parameters, Base
from q2_mlab.db.mapping import remap_parameters
from q2_mlab.learningtask import RegressionTask
from typing import Optional, Callable


DROP_COLS = ['SAMPLE_ID', 'Y_PRED', 'Y_TRUE']


def format_db(db_file: Optional[str] = None) -> str:
    if db_file is not None:
        loc = '/' + db_file
    else:
        # this creates an in memory database
        loc = ''
    return f"sqlite://{loc}"


def create_engine(db_file: Optional[str] = None, echo=True) -> Engine:
    engine = sql_create_engine(format_db(db_file), echo=echo)
    return engine


def create(db_file: Optional[str] = None, echo=True) -> Engine:
    engine = create_engine(db_file, echo=echo)
    Base.metadata.create_all(engine)
    return engine


def add(engine: Engine, results: pd.DataFrame, parameters: dict,
        dataset: str, target: str, level: str, algorithm: str,
        artifact_uuid: str,
        ) -> None:
    Session = sessionmaker(bind=engine)
    session = Session()

    # check if parameters exists in db
    query = session.query(Parameters).filter_by(**parameters)
    params = query.first()

    # if no record exists with these parameters, add them to the table
    if params is None:
        params = Parameters(**parameters)
        session.add(params)
    session.flush()
    params_id = params.id

    # check if algorithm is valid
    valid_algorithms = RegressionTask.algorithms
    if algorithm not in valid_algorithms:
        raise ValueError(f"Invalid choice '{algorithm}' for algorithm."
                         f"Valid choices: {valid_algorithms}."
                         )

    time = datetime.now()
    for entry in results.iterrows():
        score = RegressionScore(datetime=time,
                                parameters_id=params_id,
                                dataset=dataset,
                                target=target,
                                level=level,
                                algorithm=algorithm,
                                artifact_uuid=artifact_uuid,
                                **entry[1],
                                )
        session.add(score)

    session.commit()


def add_from_qza(artifact: Artifact,
                 parameters: dict,
                 dataset: str,
                 target: str,
                 level: str,
                 algorithm: str,
                 db_file: Optional[str] = None,
                 echo: bool = True,
                 engine_creator: Callable = create_engine,
                 ):

    engine = engine_creator(db_file, echo=echo)

    results = artifact.view(pd.DataFrame)
    artifact_uuid = str(artifact.uuid)
    # perform filtering on results (single entry per cross validation)
    results.drop(DROP_COLS, axis=1, inplace=True)
    results.drop_duplicates(inplace=True)

    # remap multi-type arguments, e.g., 'max_features'
    parameters = remap_parameters(parameters)

    add(engine, results, parameters,
        dataset=dataset, target=target, level=level, algorithm=algorithm,
        artifact_uuid=artifact_uuid,
        )
    return engine
