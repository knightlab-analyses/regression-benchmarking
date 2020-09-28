import pandas as pd
import biom
import skbio
from qiime2 import Artifact, Metadata
from os import path
from q2_mlab._preprocess import clean_metadata
from qiime2.plugin.testing import TestPluginBase
from pandas.testing import assert_series_equal


class PreprocessingTests(TestPluginBase):
    package = 'q2_mlab'

    def setUp(self):
        super().setUp()
        self.preprocess = self.plugin.pipelines['preprocess']

        continuous_metadata = pd.DataFrame({'target': ['1.0', '2.0',
                                                       '3.0', '4.0'],
                                            'contain_nan': ['3.3', '3.5',
                                                            None, '3.9']},
                                           index=pd.Index(['A', 'B', 'C', 'D'],
                                           name='id'))
        self.continuous_metadata = continuous_metadata

        discrete_metadata = pd.DataFrame({'target': ['0', '1', '0', '1'],
                                          'target_int': [1, 0, 1, 0],
                                          'contain_nan': ['0', '1', None, '1'],
                                          'non_encoded': ['10', '2', '', 'b']},
                                         index=pd.Index(['A', 'B', 'C', 'D'],
                                         name='id'))
        self.discrete_metadata = discrete_metadata

        TEST_DIR = path.split(__file__)[0]
        md_path = path.join(TEST_DIR, 'data/sample-metadata-binary.tsv')
        table_path = path.join(TEST_DIR, 'data/table.qza')
        rooted_tree_path = path.join(TEST_DIR, 'data/rooted-tree.qza')
        unrooted_tree_path = path.join(TEST_DIR, 'data/unrooted-tree.qza')

        self.mp_sample_metadata = Metadata.load(md_path)
        self.mp_table = Artifact.load(table_path)
        self.mp_rooted_tree = Artifact.load(rooted_tree_path)
        self.mp_unrooted_tree = Artifact.load(unrooted_tree_path)

    def test_preprocess_output(self):

        results = self.preprocess(table=self.mp_table,
                                  metadata=self.mp_sample_metadata,
                                  phylogeny=self.mp_rooted_tree,
                                  sampling_depth=1000,
                                  min_frequency=10,
                                  target_variable="body-site",
                                  discrete=True,
                                  with_replacement=False,
                                  n_jobs=1)
        self.assertEqual(len(results), 9)
        self.assertTrue(str(results[0].type) == 'FeatureTable[Frequency]')
        self.assertTrue(str(results[1].type) == 'FeatureTable[Frequency]')
        self.assertTrue(str(results[2].type) == 'SampleData[Target]')
        self.assertTrue(str(results[3].type) == 'DistanceMatrix')
        self.assertTrue(str(results[4].type) == 'DistanceMatrix')
        self.assertTrue(str(results[5].type) == 'DistanceMatrix')
        self.assertTrue(str(results[6].type) == 'DistanceMatrix')

        phylo_dm_strings = ["DistanceMatrix % Properties('phylogenetic')",
                            "DistanceMatrix"]
        # QIIME2 2020.8 changes beta_phylogenetic to a Pipeline and
        # changes the resulting artifact type to "DistanceMatrix"
        self.assertTrue(str(results[7].type) in phylo_dm_strings)
        self.assertTrue(str(results[8].type) in phylo_dm_strings)

        self.assertEqual(
            results[0].view(biom.Table).shape[1],
            results[1].view(biom.Table).shape[1]
        )

    def test_preprocess_output_no_phylo(self):

        results = self.preprocess(table=self.mp_table,
                                  metadata=self.mp_sample_metadata,
                                  sampling_depth=1000,
                                  min_frequency=10,
                                  target_variable="body-site",
                                  discrete=True,
                                  with_replacement=False,
                                  n_jobs=1)
        self.assertEqual(len(results), 9)
        self.assertTrue(str(results[0].type) == 'FeatureTable[Frequency]')
        self.assertTrue(str(results[1].type) == 'FeatureTable[Frequency]')
        self.assertTrue(str(results[2].type) == 'SampleData[Target]')
        self.assertTrue(str(results[3].type) == 'DistanceMatrix')
        self.assertTrue(str(results[4].type) == 'DistanceMatrix')
        self.assertTrue(str(results[5].type) == 'DistanceMatrix')
        self.assertTrue(str(results[6].type) == 'DistanceMatrix')
        self.assertTrue(str(results[7].type) == 'DistanceMatrix')
        self.assertTrue(str(results[8].type) == 'DistanceMatrix')

        # Assert that empty phylogenetic distance matrices have shape (1,1)
        self.assertTupleEqual(
            results[7].view(skbio.DistanceMatrix).shape,
            (1, 1)
        )
        self.assertTupleEqual(
            results[8].view(skbio.DistanceMatrix).shape,
            (1, 1)
        )

        # Assert that the rarefied and unrarefied tables match in # samples
        self.assertEqual(
            results[0].view(biom.Table).shape[1],
            results[1].view(biom.Table).shape[1]
        )

    def test_clean_metadata_continuous(self):
        clean_df = clean_metadata(self.continuous_metadata,
                                  target_variable='target',
                                  discrete=False)
        self.assertTupleEqual(clean_df.shape, (4, 1))
        expected_series = pd.Series(data=[1.0, 2.0, 3.0, 4.0],
                                    index=pd.Index(['A', 'B', 'C', 'D'],
                                    name='id'),
                                    name='target')
        assert_series_equal(clean_df['target'], expected_series)

        clean_df2 = clean_metadata(self.continuous_metadata,
                                   target_variable='contain_nan',
                                   discrete=False)
        self.assertTupleEqual(clean_df2.shape, (3, 1))
        expected_series = pd.Series(data=[3.3, 3.5, 3.9],
                                    index=pd.Index(['A', 'B', 'D'],
                                    name='id'),
                                    name='contain_nan')
        assert_series_equal(clean_df2['contain_nan'], expected_series)

    def test_clean_metadata_discrete(self):
        clean_df = clean_metadata(self.discrete_metadata,
                                  target_variable='target_int',
                                  discrete=True)
        self.assertTupleEqual(clean_df.shape, (4, 1))
        expected_series = pd.Series(data=[1, 0, 1, 0],
                                    index=pd.Index(['A', 'B', 'C', 'D'],
                                    name='id'),
                                    name='target_int')
        assert_series_equal(clean_df['target_int'], expected_series)

        clean_df2 = clean_metadata(self.discrete_metadata,
                                   target_variable='target',
                                   discrete=True)
        self.assertTupleEqual(clean_df2.shape, (4, 1))
        expected_series = pd.Series(data=[0, 1, 0, 1],
                                    index=pd.Index(['A', 'B', 'C', 'D'],
                                    name='id'),
                                    name='target')
        assert_series_equal(clean_df2['target'], expected_series)

        clean_df3 = clean_metadata(self.discrete_metadata,
                                   target_variable='contain_nan',
                                   discrete=True)
        self.assertTupleEqual(clean_df3.shape, (3, 1))
        expected_series = pd.Series(data=[0.0, 1.0, 1.0],
                                    index=pd.Index(['A', 'B', 'D'],
                                    name='id'),
                                    name='contain_nan')
        assert_series_equal(clean_df3['contain_nan'], expected_series)

        with self.assertRaisesRegex(ValueError,
                                    'For classification, the only allowed '
                                    'values in the target column are 0 and 1'):
            clean_metadata(self.discrete_metadata,
                           target_variable='non_encoded',
                           discrete=True)
