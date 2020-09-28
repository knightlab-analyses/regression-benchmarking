import unittest

from q2_mlab.db.mapping import (
    remap_parameters,
    trivial_remapper,
    string_or_number_remapper,
    serialize_remapper,
)


class ParameterRemapTestCase(unittest.TestCase):
    def test_trivial_remapper(self):
        parameter = 'param'
        value = 24
        obs = trivial_remapper(parameter, value)
        self.assertDictEqual({'param': 24}, obs)

    def test_string_or_number_remapper(self):
        parameter = 'param'
        value = 24
        obs = string_or_number_remapper(parameter, value)
        self.assertDictEqual({'param_NUMBER': 24}, obs)
        value = '24'
        obs = string_or_number_remapper(parameter, value)
        self.assertDictEqual({'param_STRING': '24'}, obs)

    def test_serialize_mapper(self):
        parameter = 'param'
        value = [1, 3, 4]
        obs = serialize_remapper(parameter, value)
        exp = {"param": '[1, 3, 4]'}
        self.assertDictEqual(exp, obs)

    def test_remap_parameters(self):
        params = {
            'trivial': 24,
            'gamma': 0.01,
            'hidden_layer_sizes': [1, 3, 4],
        }

        obs = remap_parameters(params)
        exp = {
            'trivial': 24,
            'gamma_NUMBER': 0.01,
            'hidden_layer_sizes': '[1, 3, 4]',
        }
        self.assertDictEqual(exp, obs)


if __name__ == '__main__':
    unittest.main()
