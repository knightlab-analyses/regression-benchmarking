from sqlalchemy import (
    Column,
    Float,
    String,
    Boolean,
    Integer,
    DateTime,
    ForeignKey,
)

from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()


class Score:
    id = Column(Integer, primary_key=True)
    algorithm = Column(String, nullable=False)
    dataset = Column(String, nullable=False)
    level = Column(String, nullable=False)  # 16S, metagenomics, etc.
    target = Column(String, nullable=False)
    artifact_uuid = Column(String, nullable=False)
    datetime = Column(DateTime)
    # could be cool to store the timedelta with Interval instead
    RUNTIME = Column(Float)
    CV_IDX = Column(Integer)


class RegressionScore(Base, Score):

    __tablename__ = 'regression'

    R2 = Column(Float)
    RMSE = Column(Float)
    MAE = Column(Float)

    parameters_id = Column(Integer, ForeignKey('parameters.id'))


class ClassificationScore(Base, Score):

    __tablename__ = 'classification'

    ACCURACY = Column(Float)
    AUPRC = Column(Float)
    AUROC = Column(Float)
    F1 = Column(Float)
    PROB_CLASS_0 = Column(Float)
    PROB_CLASS_1 = Column(Float)

    parameters_id = Column(Integer, ForeignKey('parameters.id'))


class Parameters(Base):
    __tablename__ = 'parameters'

    id = Column(Integer, primary_key=True)

    # algorithm parameters
    activation = Column(String)
    alpha = Column(Float)
    booster = Column(String)
    bootstrap = Column(Boolean)
    C = Column(Float)
    coef0 = Column(Integer)
    colsample_bytree = Column(Float)
    criterion = Column(String)
    early_stopping = Column(Boolean)
    epsilon = Column(Float)
    fit_intercept = Column(Boolean)
    gamma_STRING = Column(String)
    gamma_NUMBER = Column(Float)
    hidden_layer_sizes = Column(String)
    l1_ratio = Column(Float)
    l2_regularization = Column(Float)
    learning_rate_STRING = Column(String)
    learning_rate_NUMBER = Column(Float)
    loss = Column(String)
    max_depth = Column(Integer)
    max_features_STRING = Column(String)
    max_features_NUMBER = Column(Float)
    max_iter = Column(Integer)
    max_leaf_nodes = Column(Integer)
    max_samples = Column(Float)
    metric = Column(String)
    min_child_weight = Column(Float)
    min_samples_split = Column(Float)
    min_samples_leaf = Column(Float)
    n_jobs = Column(Integer)
    n_estimators = Column(Integer)
    n_iter_no_change = Column(Integer)
    n_neighbors = Column(Integer)
    normalize = Column(Boolean)
    num_leaves = Column(Integer)
    objective = Column(String)
    penalty = Column(String)
    positive = Column(Boolean)
    silent = Column(Integer)
    solver = Column(String)
    subsample = Column(Float)
    tol = Column(Float)
    random_state = Column(Integer)
    reg_alpha = Column(Float)
    reg_lambda = Column(Float)
    weights = Column(String)
