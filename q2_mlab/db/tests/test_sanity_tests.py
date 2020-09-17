import unittest
from unittest.mock import MagicMock
from datetime import datetime, timedelta
import tempfile
from q2_mlab.db.maint import (
    create,
    sessionmaker as maint_sessionmaker,
)
from q2_mlab.db.schema import (
    Parameters,
    RegressionScore,
)
from q2_mlab.db.sanity import (
    CreateSession,
    _set_params,
    _validate_params,
    _make_session,
    has_duplicate_parameters,
    has_duplicate_artifact_uuid,
    count_duplicate_artifact_uuid,
)


class CreateSessionTestCase(unittest.TestCase):

    def test_new_CreateSession_with_mocks(self):
        def mock_set_params(obj, args, kwargs):
            obj.foo = args[0]
            obj.bar = kwargs.pop('bar')
            return (), kwargs

        def mock_validate_strategy(obj):
            return True

        def mock_make_session(obj):
            obj.session = {'foo': obj.foo, 'bar': obj.bar}

        @CreateSession(
            session_strategy=mock_make_session,
            params_strategy=mock_set_params,
            validation_strategy=mock_validate_strategy
        )
        def f(session):
            return session

        exp = {'foo': 7.25, 'bar': '2019'}
        obs = f(7.25, bar='2019')
        self.assertDictEqual(exp, obs)

    def test_set_params(self):
        obj = MagicMock()
        input_args = ('foo', 'bar', 'baz')
        input_kwargs = {
            'db_file': 'some_file',
            'echo': True,
            'other_kwarg': True,
        }
        args, kwargs = _set_params(obj, input_args, input_kwargs)

        self.assertIsNone(obj.session)
        self.assertEqual(obj.db_file, 'some_file')
        self.assertEqual(obj.echo, True)
        self.assertTupleEqual(input_args, args)
        self.assertDictEqual({'other_kwarg': True}, kwargs)

    def test_validate_params(self):
        obj = MagicMock()
        obj.db_file = None
        obj.session = None
        with self.assertRaises(ValueError):
            _validate_params(obj)

        obj.db_file = 'some file'
        obj.session = None
        _validate_params(obj)

    def test_make_session(self):

        def mock_sessionmaker(db_file, echo):
            class Session(dict):
                def __init__(self):
                    super().__init__()
                    self.update({'db_file': db_file, 'echo': echo})
            return Session

        obj = MagicMock()
        obj.session = None
        obj.db_file = 'db_filename'
        obj.echo = 8.25

        exp = {'db_file': 'db_filename', 'echo': 8.25}

        _make_session(obj, mock_sessionmaker)

        self.assertDictEqual(exp, obj.session)

    def test_create_session_integration_defaults(self):
        fh = tempfile.NamedTemporaryFile()
        fp = fh.name
        engine = create(db_file=fp, echo=False)
        Session = maint_sessionmaker(engine=engine)
        sess = Session()
        params = Parameters(alpha=1)
        sess.add(params)
        sess.commit()

        @CreateSession()
        def some_session_function(session):
            results = session.query(Parameters).all()
            return len(results)

        obs = some_session_function(db_file=fp, echo=False)
        self.assertEqual(1, obs)

        obs = some_session_function(session=sess, echo=False)
        self.assertEqual(1, obs)


class SanityCheckTestCases(unittest.TestCase):

    def setUp(self):
        self.db_fh = tempfile.NamedTemporaryFile()
        fp = self.db_fh.name
        engine = create(db_file=fp, echo=False)
        Session = maint_sessionmaker(engine=engine)
        self.session = Session()
        self.boilerplate = {
            'algorithm': 'TestRegressor',
            'dataset': 'TestDataset',
            'level': 'TestLevel',
            'target': 'TestTarget',
        }

    def tearDown(self):
        self.db_fh.close()

    def test_has_duplicate_parameters(self):
        params = Parameters(alpha=1)
        self.session.add(params)
        self.session.commit()

        finds_dup = has_duplicate_parameters(session=self.session)
        self.assertFalse(finds_dup)

        new_params = Parameters(alpha=1)
        self.session.add(new_params)
        self.session.commit()
        finds_dup = has_duplicate_parameters(session=self.session)
        self.assertTrue(finds_dup)

    def test_has_duplicate_artifact_uuid(self):
        time01 = datetime.now()
        score01 = RegressionScore(**self.boilerplate,
                                  artifact_uuid='id1',
                                  datetime=time01
                                  )
        self.session.add(score01)
        score02 = RegressionScore(**self.boilerplate,
                                  artifact_uuid='id1',
                                  datetime=time01
                                  )
        self.session.add(score02)
        score03 = RegressionScore(**self.boilerplate,
                                  artifact_uuid='id2',
                                  datetime=time01
                                  )
        self.session.add(score03)
        self.session.commit()
        finds_dup = has_duplicate_artifact_uuid(session=self.session)
        self.assertFalse(finds_dup)

        time02 = time01 + timedelta(seconds=1)
        score04 = RegressionScore(**self.boilerplate,
                                  artifact_uuid='id1',
                                  datetime=time02,
                                  )
        self.session.add(score04)
        self.session.commit()
        finds_dup = has_duplicate_artifact_uuid(session=self.session)
        self.assertTrue(finds_dup)

    def test_count_uuid(self):
        time01 = datetime.now()
        score01 = RegressionScore(**self.boilerplate,
                                  artifact_uuid='id1',
                                  datetime=time01
                                  )
        self.session.add(score01)
        score02 = RegressionScore(**self.boilerplate,
                                  artifact_uuid='id1',
                                  datetime=time01
                                  )
        self.session.add(score02)
        score03 = RegressionScore(**self.boilerplate,
                                  artifact_uuid='id2',
                                  datetime=time01
                                  )
        self.session.add(score03)
        score04 = RegressionScore(**self.boilerplate,
                                  artifact_uuid='id3',
                                  datetime=time01
                                  )
        self.session.add(score04)
        time02 = time01 + timedelta(seconds=1)
        score04 = RegressionScore(**self.boilerplate,
                                  artifact_uuid='id1',
                                  datetime=time02,
                                  )
        self.session.add(score04)
        self.session.commit()
        count_dup = count_duplicate_artifact_uuid(session=self.session)
        exp = [{'times_observed': 1, 'uuid_count': 2},
               {'times_observed': 2, 'uuid_count': 1},
               ]
        self.assertCountEqual(exp, count_dup)


if __name__ == '__main__':
    unittest.main()
