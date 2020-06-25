# ----------------------------------------------------------------------------
# Copyright (c) 2020, mlab development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from ._preprocess import preprocess
from ._benchmark import unit_benchmark
from ._version import get_versions
from .learningtask import LearningTask, ClassificationTask, RegressionTask
from ._type import Target, Results
from ._format import ResultsDirectoryFormat, ResultsFormat

__version__ = get_versions()["version"]

del get_versions

__all__ = [
    "preprocess",
    "unit_benchmark",
    "ResultsDirectoryFormat",
    "ResultsFormat",
    "Target",
    "Results",
    "LearningTask",
    "ClassificationTask",
    "RegressionTask",
]
