import unittest
import os
import shutil
import subprocess
import pandas as pd
from q2_mlab import (
    orchestrate_hyperparameter_search,
    doctor_hyperparameter_search,
    sort_result_artifact_filenames,
    parse_info,
    get_results,
    filter_duplicate_parameter_results,
)


class DoctorTests(unittest.TestCase):
    @classmethod
    def setUpClass(self):

        self.TEST_DIR = os.path.split(__file__)[0]
        self.dataset_file = os.path.join(self.TEST_DIR, "data/table.qza")

        # The metadata doesn't matter here, as mlab will only accept
        # SampleData[Target] artifacts from preprocessing.
        self.metadata_file = os.path.join(
            self.TEST_DIR, "data/sample-metadata.tsv"
        )

        self.dataset = "dset_test"
        self.target = "reported-antibiotic-usage"
        self.prep = "16S"
        self.alg = "LinearSVR"  # Expect 112 parameters
        self.chunk_size = 20

        (
            self.script_fp,
            self.params_fp,
            self.run_info_fp,
        ) = orchestrate_hyperparameter_search(
            dataset=self.dataset,
            preparation=self.prep,
            target=self.target,
            algorithm=self.alg,
            base_dir=self.TEST_DIR,
            table_path=self.dataset_file,
            metadata_path=self.metadata_file,
            chunk_size=self.chunk_size,
            dry=False,
        )

        self.results_dir = os.path.join(
            self.TEST_DIR, self.dataset, self.prep, self.target, self.alg
        )
        # Populate results directory with older duplicate results (duplicate param_idx)
        self.older_duplicate_results = [
            "00000024_LinearSVR_chunk_888.qza",
            "00000025_LinearSVR_chunk_888.qza",
        ]
        for dupe in self.older_duplicate_results:
            open(os.path.join(self.results_dir, dupe), "a").close()

        # Populate results directory with empty results
        remove_indices = {1, 21, 41, 101}  # Chunk 1, 2, 3, 6
        param_indices = [i for i in range(1, 113) if i not in remove_indices]
        for param_idx in param_indices:
            chunk_num = (self.chunk_size // param_idx) + 1
            param_idx_str = str(param_idx).zfill(8)
            artifact_name = f"{param_idx_str}_{self.alg}_chunk_{chunk_num}.qza"
            artifact_path = os.path.join(self.results_dir, artifact_name)
            open(artifact_path, "a").close()

        # Add newer duplicate results
        self.newer_duplicate_results = [
            "00000026_LinearSVR_chunk_999.qza",
            "00000027_LinearSVR_chunk_999.qza",
        ]
        for dupe in self.newer_duplicate_results:
            open(os.path.join(self.results_dir, dupe), "a").close()

        # Move "inserted" results
        inserted_dir = os.path.join(self.results_dir, "inserted")
        if not os.path.isdir(inserted_dir):
            os.mkdir(inserted_dir)
        to_insert = [
            "00000110_LinearSVR_chunk_1.qza",
            "00000106_LinearSVR_chunk_1.qza",
        ]
        for artifact in to_insert:
            uninserted_path = os.path.join(self.results_dir, artifact)
            inserted_path = os.path.join(inserted_dir, artifact)
            os.rename(uninserted_path, inserted_path)

    @classmethod
    def tearDownClass(self):

        # Remove files we generated
        files_generated = [
            self.script_fp,
            self.params_fp,
            self.run_info_fp,
        ]
        for file in files_generated:
            if file and os.path.exists(file):
                os.remove(file)

        # Remove the barnacle output directory
        error_dir = os.path.join(
            self.TEST_DIR, self.dataset, "barnacle_output/"
        )
        os.rmdir(error_dir)

        # Remove the inserted directory
        inserted_dir = os.path.join(self.results_dir, "inserted")
        for file in os.listdir(inserted_dir):
            os.remove(os.path.join(inserted_dir, file))
        os.removedirs(inserted_dir)

        # Remove parameter subset lists, results
        for file in os.listdir(self.results_dir):
            os.remove(os.path.join(self.results_dir, file))
        os.removedirs(self.results_dir)

    def test_sort_result_artifact_filenames(self):
        artifact_filenames = [
            "00002_AlgorithmName_chunk_01.qza",
            "0001_AlgorithmName_chunk_01.qza",
            "000003_AlgorithmName_chunk_01.qza",
            "004_AlgorithmName_chunk_01.qza",
            "007_AlgorithmName_chunk_01.qza",
            "0006_AlgorithmName_chunk_01.qza",
            "5_AlgorithmName_chunk_01.qza",
        ]
        sorted_results = sort_result_artifact_filenames(artifact_filenames)

        expected_artifact_filenames = [
            "0001_AlgorithmName_chunk_01.qza",
            "00002_AlgorithmName_chunk_01.qza",
            "000003_AlgorithmName_chunk_01.qza",
            "004_AlgorithmName_chunk_01.qza",
            "5_AlgorithmName_chunk_01.qza",
            "0006_AlgorithmName_chunk_01.qza",
            "007_AlgorithmName_chunk_01.qza",
        ]
        self.assertListEqual(sorted_results, expected_artifact_filenames)

    def test_parse_info(self):
        info_dict = parse_info(self.run_info_fp)
        expected_dict = {
            "parameters_fp": self.params_fp,
            "parameter_space_size": 112,
            "chunk_size": 20,
            "remainder": 12,
            "n_chunks": 6,
        }
        self.assertDictEqual(info_dict, expected_dict)

        with open("temp_info.txt", "w") as f:
            f.write(
                "parameters_fp, parameter_space_size, chunk_size, remainder, "
                + "n_chunks\n"
                + "./sol/MG/bmi_v2/alg/alg_params.txt, 5580, 20, 0, 279"
            )
        info_dict = parse_info("temp_info.txt")
        os.remove("temp_info.txt")
        expected_dict = {
            "parameters_fp": "./sol/MG/bmi_v2/alg/alg_params.txt",
            "parameter_space_size": 5580,
            "chunk_size": 20,
            "remainder": 0,
            "n_chunks": 279,
        }
        self.assertDictEqual(info_dict, expected_dict)

    def test_getting_filtering_results(self):

        # Test getting uninserted results 
        results_list = get_results(self.results_dir)
        for fname in results_list:
            self.assertTrue(fname.endswith(".qza"))
        self.assertEqual(len(results_list), 110)

        # Test getting inserted results 
        inserted_dir = os.path.join(self.results_dir, "inserted")
        results_list = get_results(inserted_dir)
        for fname in results_list:
            self.assertTrue(fname.endswith(".qza"))
        self.assertEqual(len(results_list), 2)

        results_list = get_results(self.results_dir)
        self.assertEqual(len(results_list), 110)
        self.assertTrue(
            set(self.newer_duplicate_results).issubset(set(results_list))
        )

        # Without actually removing duplicated files:
        filtered_results_list = filter_duplicate_parameter_results(
            results_list, self.results_dir, delete=False
        )
        self.assertEqual(len(filtered_results_list), 106)

        # Assert that we kept the newer duplicates,  discarded the older
        self.assertTrue(
            set(self.newer_duplicate_results).issubset(
                set(filtered_results_list)
            )
        )
        self.assertFalse(
            set(self.older_duplicate_results).issubset(
                set(filtered_results_list)
            )
        )

        # With removing duplicated files:
        filtered_results_list = filter_duplicate_parameter_results(
            results_list, self.results_dir, delete=True
        )
        self.assertEqual(len(filtered_results_list), 106)
        # This should now not see any uninserted results.
        results_list = get_results(self.results_dir)
        self.assertEqual(len(results_list), 106)
        self.assertTrue(
            set(self.newer_duplicate_results).issubset(set(results_list))
        )
        self.assertFalse(
            set(self.older_duplicate_results).issubset(set(results_list))
        )

    def test_doctor_hyperparameter_search(self):
        cmd = doctor_hyperparameter_search(
            dataset=self.dataset,
            preparation=self.prep,
            target=self.target,
            algorithm=self.alg,
            base_dir=self.TEST_DIR,
            max_results=1000,
            delete_duplicates=False,
        )

        self.assertEqual(cmd, f"qsub -t 1,2,3,6 {self.script_fp}")

    def test_doctor_with_all_missing_results(self):

        this_alg = "ExtraTreesClassifier"
        (
            this_script_fp,
            this_params_fp,
            this_run_info_fp,
        ) = orchestrate_hyperparameter_search(
            dataset=self.dataset,
            preparation=self.prep,
            target=self.target,
            algorithm=this_alg,
            base_dir=self.TEST_DIR,
            table_path=self.dataset_file,
            metadata_path=self.metadata_file,
            chunk_size=self.chunk_size,
            dry=False,
        )
        cmd = doctor_hyperparameter_search(
            dataset=self.dataset,
            preparation=self.prep,
            target=self.target,
            algorithm=this_alg,
            base_dir=self.TEST_DIR,
            max_results=1000,
            delete_duplicates=False,
        )
        expected_chunks = ",".join([str(i) for i in range(1, 51)])
        self.assertEqual(cmd, f"qsub -t {expected_chunks} {this_script_fp}")

        # Remove files we generated
        files_generated = [
            this_script_fp,
            this_params_fp,
            this_run_info_fp,
        ]
        for file in files_generated:
            if file and os.path.exists(file):
                os.remove(file)
        this_results_dir = os.path.join(
            self.TEST_DIR, self.dataset, self.prep, self.target, this_alg
        )
        os.removedirs(this_results_dir)
