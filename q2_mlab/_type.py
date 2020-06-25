# ----------------------------------------------------------------------------
# Copyright (c) 2020, mlab development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from qiime2.plugin import SemanticType
from q2_types.sample_data import SampleData

Target = SemanticType('Target', variant_of=SampleData.field['type'])
Results = SemanticType('Results', variant_of=SampleData.field['type'])