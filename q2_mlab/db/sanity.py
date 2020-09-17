from enum import Enum
from functools import partial
import sqlalchemy
import pandas as pd
import logging
from q2_mlab.db.maint import sessionmaker
from q2_mlab.db.schema import RegressionScore, Parameters


def _set_params(obj, args, kwargs):
    obj.session = kwargs.pop('session', None)
    obj.db_file = kwargs.pop('db_file', None)
    obj.echo = kwargs.pop('echo', False)
    return args, kwargs


def _validate_params(obj):
    if obj.db_file is None and obj.session is None:
        raise ValueError("Requires keyword argument 'db_file' or 'session'")


def _make_session(obj, sessionmaker):
    if obj.session is None:
        Session = sessionmaker(obj.db_file, echo=obj.echo)
        obj.session = Session()
    elif obj.echo:
        raise UserWarning("Parameter 'echo' has no effect when used with "
                          "'session'")


class CreateSession:

    def __init__(self,
                 session_strategy=None,
                 params_strategy=None,
                 validation_strategy=None,
                 ):
        self.session = None
        self.db_file = None
        self.echo = None
        self.session_strategy = None
        self.params_strategy = None
        self.validation_strategy = None

        self.set_strategies(
            params_strategy, session_strategy, validation_strategy
        )

    def set_strategies(self, params_strategy, session_strategy,
                       validation_strategy):
        if session_strategy is None:
            session_strategy = partial(_make_session,
                                       sessionmaker=sessionmaker)
        if params_strategy is None:
            params_strategy = _set_params
        if validation_strategy is None:
            validation_strategy = _validate_params
        self.session_strategy = session_strategy
        self.params_strategy = params_strategy
        self.validation_strategy = validation_strategy

    def __call__(self, func):

        def func_wrapper(*args, **kwargs):
            args, kwargs = self.params_strategy(self, args, kwargs)
            self.validation_strategy(self)
            self.session_strategy(self)
            session = self.session
            return func(*args, session, **kwargs)

        return func_wrapper


@CreateSession()
def has_duplicate_parameters(session):
    """
    The `create_session` decorator guarantees that session is not None,
    and is a valid session, instead.
    """
    engine = session.get_bind()
    Parameters_columns = sqlalchemy.inspect(
        engine
    ).get_columns(
        Parameters.__tablename__
    )
    columns = [col['name'] for col in Parameters_columns
               if col['name'] != 'id']

    all_params = pd.read_sql_table(Parameters.__tablename__, con=engine)

    return any(all_params[columns].duplicated())


@CreateSession()
def has_duplicate_artifact_uuid(session):
    engine = session.get_bind()
    q = session.query(RegressionScore.artifact_uuid,
                      RegressionScore.datetime,
                      ).distinct()
    df = pd.read_sql(q.statement, con=engine)
    has_duplicates = any(df['artifact_uuid'].duplicated())
    return has_duplicates


@CreateSession()
def count_duplicate_artifact_uuid(session):
    engine = session.get_bind()
    q = session.query(RegressionScore.artifact_uuid,
                      RegressionScore.datetime,
                      ).distinct()
    df = pd.read_sql(q.statement, con=engine)
    count = df.groupby('artifact_uuid').count()['datetime'].value_counts()
    count.index.name = 'times_observed'
    count.name = 'uuid_count'
    count = pd.DataFrame(count).reset_index().T
    return count.to_dict().values()


class Violations(Enum):
    DUPLICATE_PARAMETERS = 1
    DUPLICATE_UUID = 2


class Check:

    def __init__(self, session):
        self.session = session

    def __call__(self):
        raise NotImplementedError('Abstract Method')

    def failure(self):
        return self()

    @staticmethod
    def violation():
        raise NotImplementedError('Abstract method')

    def failure_info(self):
        return None

    @property
    def name(self):
        return type(self).__name__


class NoDuplicatedParameterCheck(Check):
    def __call__(self):
        return has_duplicate_parameters(session=self.session)

    @staticmethod
    def violation():
        return Violations.DUPLICATE_PARAMETERS


class NoDuplicatedUUIDCheck(Check):
    def __call__(self):
        return has_duplicate_artifact_uuid(session=self.session)

    def failure_info(self):
        return count_duplicate_artifact_uuid(session=self.session)

    @staticmethod
    def violation():
        return Violations.DUPLICATE_UUID


DEFAULT_CHECKS = [
    NoDuplicatedParameterCheck,
    NoDuplicatedUUIDCheck,
]


class SanityChecker:

    def __init__(self, checks=None, additional_info=False):
        self.session = None
        self.violations = set()
        if checks is None:
            checks = DEFAULT_CHECKS
        self.checks = checks
        self.additional_info = additional_info

    @CreateSession()
    def start(self, session):
        self.session = session

    def run(self, db_file=None):
        if db_file is not None:
            self.start(db_file=db_file)
        for check in self.checks:
            checker = check(session=self.session)
            logging.info(f"Performing check: {checker.name}...")
            if checker.failure():
                logging.info(f"Failed {checker.name}.")
                self.violations.add(checker.violation())
                if self.additional_info:
                    self.get_extra_info(checker)
            else:
                logging.info(f"Successful {checker.name}.")

        logging.info("Finished Checks.")
        logging.info(f"Violations: {self.violations}")

    def get_extra_info(self, checker):
        info = checker.failure_info()
        if info is not None:
            logging.info(f"Additional info on failed {checker.name}: {info}")


if __name__ == "__main__":
    import sys
    db_file = sys.argv[1]
    logging.getLogger().setLevel(logging.INFO)
    checker = SanityChecker(additional_info=True)
    checker.run(db_file=db_file)
