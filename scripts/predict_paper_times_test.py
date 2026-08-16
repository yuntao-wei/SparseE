import json
import math
import statistics
import tempfile
import unittest
from dataclasses import fields, replace
from pathlib import Path

import predict_paper_times as predictor


class PredictPaperTimesTest(unittest.TestCase):
    def setUp(self):
        self.fhe = predictor.FheParams()
        self.workloads = predictor.paper_workloads()
        self.calibration = predictor.load_software_calibration(
            predictor.DEFAULT_CALIBRATION_PATH
        )

    def test_paper_parameter_profile(self):
        self.assertEqual(
            self.fhe,
            predictor.FheParams(
                rlwe_dimension=4096,
                rlwe_modulus_bits=64,
                rgsw_modulus_bits=128,
                rgsw_special_prime_bits=64,
                rgsw_decomposition_digits=2,
            ),
        )

    def test_all_paper_workload_rows_are_present(self):
        labels = {(row.domain, row.dataset) for row in self.workloads}
        self.assertEqual(len(labels), 14)
        self.assertEqual(
            labels,
            {
                ("GNN", "CR"),
                ("GNN", "PB"),
                ("GNN", "OA"),
                ("GNN", "OC"),
                ("LLM", "L=32"),
                ("LLM", "L=1024"),
                ("LLM", "L=2048"),
                ("LLM", "L=4096"),
                ("3D CNN", "MN10"),
                ("3D CNN", "MN40"),
                ("3D CNN", "SB"),
                ("RecSys", "ML"),
                ("RecSys", "YP"),
                ("RecSys", "AE"),
            },
        )

    def test_direct_and_proxy_nnz_are_not_conflated(self):
        direct = [row for row in self.workloads if row.nnz_confidence == "direct"]
        proxy = [row for row in self.workloads if row.nnz_confidence == "proxy"]
        self.assertEqual({row.dataset for row in direct}, {"CR", "PB", "OA", "OC"})
        self.assertEqual(
            {row.dataset for row in proxy},
            {"MN10", "MN40", "SB", "ML", "YP", "AE"},
        )

    def test_calibration_uses_measured_software_samples(self):
        self.assertEqual(
            self.calibration.schema_version,
            predictor.SOFTWARE_CALIBRATION_SCHEMA_VERSION,
        )
        self.assertEqual(len(self.calibration.samples), 4)
        self.assertEqual(self.calibration.external_products_per_hswitch, 1)
        self.assertEqual(
            {
                (sample.effective_nnz, sample.selector_count)
                for sample in self.calibration.samples
            },
            {(13, 56), (7, 20), (12, 56), (11, 56)},
        )
        expected = statistics.median(
            sample.median_elapsed_s
            / predictor.calibration_sample_work_units(sample, self.calibration)
            for sample in self.calibration.samples
        )
        self.assertEqual(
            predictor.calibrate_seconds_per_unit(self.calibration), expected
        )

    def test_prediction_is_linear_in_normalized_software_work(self):
        cost = predictor.calibrate_seconds_per_unit(self.calibration)
        prediction = predictor.predict(self.workloads, self.fhe, cost)[0]
        doubled = predictor.predict(self.workloads, self.fhe, cost * 2)[0]

        self.assertEqual(prediction.predicted_software_s, cost * prediction.work_units)
        self.assertEqual(
            doubled.predicted_software_s, prediction.predicted_software_s * 2
        )
        self.assertAlmostEqual(
            prediction.software_delta_s,
            prediction.predicted_software_s
            - prediction.workload.paper_sparsee_software_s,
        )
        self.assertTrue(math.isfinite(prediction.software_delta_percent))

    def test_benes_selector_count_matches_padded_network(self):
        self.assertEqual(predictor.padded_network_size(13), 16)
        self.assertEqual(predictor.benes_layer_count(13), 7)
        self.assertEqual(predictor.benes_selector_count(13), 56)
        self.assertEqual(predictor.padded_network_size(10556), 16384)
        self.assertEqual(predictor.benes_layer_count(10556), 27)
        self.assertEqual(predictor.benes_selector_count(10556), 221184)
        self.assertEqual(predictor.benes_selector_count(1), 0)

    def test_prediction_models_one_external_product_per_hswitch(self):
        workload = next(row for row in self.workloads if row.dataset == "CR")
        one_ep = predictor.work_units(workload, self.fhe, 1)
        two_ep = predictor.work_units(workload, self.fhe, 2)

        self.assertEqual(predictor.DEFAULT_EXTERNAL_PRODUCTS_PER_HSWITCH, 1)
        self.assertLess(one_ep, two_ep)
        self.assertEqual(
            predictor.predict(
                [workload], self.fhe, seconds_per_work_unit=1.0
            )[0].work_units,
            one_ep,
        )

    def test_external_product_count_must_be_positive(self):
        workload = next(row for row in self.workloads if row.dataset == "CR")
        with self.assertRaises(ValueError):
            predictor.work_units(workload, self.fhe, 0)

    def test_every_fhe_parameter_affects_work_model(self):
        workload = next(row for row in self.workloads if row.dataset == "CR")
        baseline = predictor.work_units(workload, self.fhe)
        variants = [
            replace(self.fhe, rlwe_dimension=8192),
            replace(self.fhe, rlwe_modulus_bits=65),
            replace(self.fhe, rgsw_modulus_bits=129),
            replace(self.fhe, rgsw_special_prime_bits=65),
            replace(self.fhe, rgsw_decomposition_digits=3),
        ]
        for variant in variants:
            with self.subTest(variant=variant):
                self.assertNotEqual(predictor.work_units(workload, variant), baseline)

    def test_true_nnz_override_replaces_proxy_marker(self):
        overridden = predictor.apply_overrides(self.workloads, {"ML": 1_000_209})
        movie_lens = next(row for row in overridden if row.dataset == "ML")
        self.assertEqual(movie_lens.effective_nnz, 1_000_209)
        self.assertEqual(movie_lens.nnz_confidence, "override")
        self.assertEqual(movie_lens.nnz_source, "user JSON override")

    def test_prediction_schema_is_stable(self):
        workload_fields = {field.name for field in fields(predictor.Workload)}
        prediction_fields = {field.name for field in fields(predictor.Prediction)}
        self.assertEqual(
            workload_fields,
            {
                "domain",
                "dataset",
                "parameters",
                "effective_nnz",
                "nnz_source",
                "nnz_confidence",
                "paper_sparsee_software_s",
            },
        )
        self.assertEqual(
            prediction_fields,
            {
                "workload",
                "row_ciphertexts",
                "network_size",
                "log_factor",
                "benes_layers",
                "selector_count",
                "work_units",
                "predicted_software_s",
                "software_delta_s",
                "software_delta_percent",
            },
        )

        cost = predictor.calibrate_seconds_per_unit(self.calibration)
        predictions = predictor.predict(self.workloads, self.fhe, cost)
        payload = predictor.build_json_payload(
            predictions,
            self.fhe,
            self.calibration,
            cost,
            "test",
            1,
        )

        self.assertEqual(
            set(payload),
            {
                "target_fhe_params",
                "software_calibration",
                "coefficient_source",
                "seconds_per_work_unit",
                "external_products_per_hswitch",
                "predictions",
            },
        )

    def test_calibration_schema_version_is_validated(self):
        raw = json.loads(predictor.DEFAULT_CALIBRATION_PATH.read_text(encoding="utf-8"))
        for version in (None, True, 2):
            with self.subTest(version=version):
                with tempfile.TemporaryDirectory() as temp_dir:
                    candidate = dict(raw)
                    candidate["schema_version"] = version
                    path = Path(temp_dir) / "calibration.json"
                    path.write_text(json.dumps(candidate), encoding="utf-8")
                    with self.assertRaisesRegex(
                        ValueError, "unsupported software calibration schema version"
                    ):
                        predictor.load_software_calibration(path)


if __name__ == "__main__":
    unittest.main()
