import pandas as pd
from sqlalchemy.engine import Engine
from sqlalchemy import create_engine as sql_create_engine
from sqlalchemy.orm import sessionmaker as sql_sessionmaker
from datetime import datetime
from qiime2 import Artifact
from q2_mlab.db.schema import (
    Score,
    RegressionScore,
    ClassificationScore,
    Parameters,
    Base,
)
from q2_mlab.db.mapping import remap_parameters
from q2_mlab.learningtask import RegressionTask, ClassificationTask
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


def sessionmaker(db_file=None, engine=None, echo=False):
    if db_file is not None and engine is None:
        engine = create_engine(db_file, echo=echo)
    elif engine is not None and db_file is None:
        pass
    elif engine is not None and db_file is not None:
        raise ValueError(f"Parameters 'db_file' and 'engine' are mutually "
                         f"exclusive. Got: db_file={db_file}, engine={engine}")
    else:
        raise ValueError(f"Either 'db_file' or 'engine' is required for "
                         f"sessionmaker.")

    # otherwise assume it is a proper engine
    Session = sql_sessionmaker(bind=engine)
    return Session


def get_table_from_algorithm(algorithm: str) -> Score:
    # check if algorithm is valid, and of regression or classification
    # and assign Table to the corresponding Regression or Classification table
    valid_algorithms = set(RegressionTask.algorithms).union(
        set(ClassificationTask.algorithms)
    )
    if algorithm in RegressionTask.algorithms:
        Table = RegressionScore
    elif algorithm in ClassificationTask.algorithms:
        Table = ClassificationScore
    else:
        raise ValueError(
            f"Invalid choice '{algorithm}' for algorithm."
            f"Valid choices: {valid_algorithms}."
        )
    return Table


def uuid_is_unique(engine: Engine, artifact_uuid: str, algorithm: str) -> bool:

    Session = sessionmaker(engine=engine)
    session = Session()

    Table = get_table_from_algorithm(algorithm)

    query = session.query(Table).filter_by(
        artifact_uuid=artifact_uuid
    )
    matching_artifact = query.first()
    session.close()
    if matching_artifact is None:
        return True
    else:
        return False


def add(engine: Engine, results: pd.DataFrame, parameters: dict,
        dataset: str, target: str, level: str, algorithm: str,
        artifact_uuid: str,
        ) -> None:
    Session = sessionmaker(engine=engine)
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

    Table = get_table_from_algorithm(algorithm)

    time = datetime.now()
    for entry in results.iterrows():
        score = Table(
            datetime=time,
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
                 allow_duplicate_uuids: bool = True,
                 engine_creator: Callable = create_engine,
                 ) -> Engine:

    engine = engine_creator(db_file, echo=echo)

    results = artifact.view(pd.DataFrame)
    artifact_uuid = str(artifact.uuid)

    if not allow_duplicate_uuids:
        # check if this uuid is in the table
        uuid_unique = uuid_is_unique(engine, artifact_uuid, algorithm)
        if not uuid_unique:
            raise ValueError(
                f"Supplied artifact {artifact_uuid} is already "
                f"contained in the table."
            )

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
