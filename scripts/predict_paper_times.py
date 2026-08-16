#!/usr/bin/env python3
"""Linearly extrapolate SparseE software execution time.

The predictor uses three inputs only:

- measured Lattigo Server.Execute benchmark samples;
- workload sizes and SparseE software latencies reported by the paper;
- the paper target FHE parameters.

The paper latencies are comparison values and never calibrate the model. The
predicted time is linear in normalized protocol work: encrypted Benes switches
plus Apply multiplications.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, TextIO, Tuple, Union


DEFAULT_EXTERNAL_PRODUCTS_PER_HSWITCH = 1
SOFTWARE_CALIBRATION_SCHEMA_VERSION = 1
DEFAULT_CALIBRATION_PATH = Path(__file__).with_name("software_calibration.json")


@dataclass(frozen=True)
class FheParams:
    rlwe_dimension: int = 4096
    rlwe_modulus_bits: int = 64
    rgsw_modulus_bits: int = 128
    rgsw_special_prime_bits: int = 64
    rgsw_decomposition_digits: int = 2


@dataclass(frozen=True)
class Workload:
    domain: str
    dataset: str
    parameters: Mapping[str, Union[int, str]]
    effective_nnz: int
    nnz_source: str
    nnz_confidence: str
    paper_sparsee_software_s: float


@dataclass(frozen=True)
class SoftwareCalibrationSample:
    name: str
    effective_nnz: int
    selector_count: int
    row_ciphertexts: int
    elapsed_samples_s: Tuple[float, ...]

    @property
    def median_elapsed_s(self) -> float:
        return statistics.median(self.elapsed_samples_s)


@dataclass(frozen=True)
class SoftwareCalibration:
    schema_version: int
    environment: Mapping[str, str]
    command: str
    fhe: FheParams
    external_products_per_hswitch: int
    samples: Tuple[SoftwareCalibrationSample, ...]


@dataclass(frozen=True)
class Prediction:
    workload: Workload
    row_ciphertexts: int
    network_size: int
    log_factor: int
    benes_layers: int
    selector_count: int
    work_units: float
    predicted_software_s: float
    software_delta_s: float
    software_delta_percent: float


def paper_workloads() -> List[Workload]:
    """Return the 14 workload rows from the paper evaluation table.

    SparseE software latency is reported in units of 10^3 seconds. Values are
    converted to seconds here.
    """

    return [
        Workload(
            "GNN",
            "CR",
            {"name": "Cora", "vertices": 2708, "edges": 10556},
            10556,
            "paper table #Edges",
            "direct",
            0.1 * 1000.0,
        ),
        Workload(
            "GNN",
            "PB",
            {"name": "PubMed", "vertices": 19717, "edges": 88648},
            88648,
            "paper table #Edges",
            "direct",
            1.3 * 1000.0,
        ),
        Workload(
            "GNN",
            "OA",
            {"name": "ogbn-arxiv", "vertices": 169343, "edges": 1166243},
            1166243,
            "paper table #Edges",
            "direct",
            21.2 * 1000.0,
        ),
        Workload(
            "GNN",
            "OC",
            {"name": "ogbn-collab", "vertices": 235868, "edges": 1285465},
            1285465,
            "paper table #Edges",
            "direct",
            23.5 * 1000.0,
        ),
        Workload(
            "LLM",
            "L=32",
            {"model": "LLaMA-7B", "hidden_dim": 4096, "sequence_length": 32},
            32,
            "derived from one-hot sequence length: one nonzero per token",
            "derived",
            6.6 * 1000.0,
        ),
        Workload(
            "LLM",
            "L=1024",
            {"model": "LLaMA-7B", "hidden_dim": 4096, "sequence_length": 1024},
            1024,
            "derived from one-hot sequence length: one nonzero per token",
            "derived",
            6.6 * 1000.0,
        ),
        Workload(
            "LLM",
            "L=2048",
            {"model": "LLaMA-7B", "hidden_dim": 4096, "sequence_length": 2048},
            2048,
            "derived from one-hot sequence length: one nonzero per token",
            "derived",
            6.6 * 1000.0,
        ),
        Workload(
            "LLM",
            "L=4096",
            {"model": "LLaMA-7B", "hidden_dim": 4096, "sequence_length": 4096},
            4096,
            "derived from one-hot sequence length: one nonzero per token",
            "derived",
            6.6 * 1000.0,
        ),
        Workload(
            "3D CNN",
            "MN10",
            {"name": "ModelNet10", "clouds": 4899, "dim": 3},
            4899,
            "proxy: paper table reports #Clouds, not true sparse voxel nnz",
            "proxy",
            1.0 * 1000.0,
        ),
        Workload(
            "3D CNN",
            "MN40",
            {"name": "ModelNet40", "clouds": 12311, "dim": 3},
            12311,
            "proxy: paper table reports #Clouds, not true sparse voxel nnz",
            "proxy",
            3.2 * 1000.0,
        ),
        Workload(
            "3D CNN",
            "SB",
            {"name": "Stanford Bunny", "clouds": 10, "dim": 3},
            10,
            "proxy: paper table reports #Clouds, not true sparse voxel nnz",
            "proxy",
            7.5 * 1000.0,
        ),
        Workload(
            "RecSys",
            "ML",
            {
                "name": "MovieLens-1M",
                "embedding_dim": 128,
                "users": 6040,
                "items": 3706,
            },
            6040,
            "proxy: paper table reports #Users, not true user-item interaction nnz",
            "proxy",
            17.9 * 1000.0,
        ),
        Workload(
            "RecSys",
            "YP",
            {"name": "Yelp", "embedding_dim": 128, "users": 42936, "items": 8026},
            42936,
            "proxy: paper table reports #Users, not true user-item interaction nnz",
            "proxy",
            0.7 * 1000.0,
        ),
        Workload(
            "RecSys",
            "AE",
            {
                "name": "Amazon-Electronics-5",
                "embedding_dim": 128,
                "users": 36074,
                "items": 2020,
            },
            36074,
            "proxy: paper table reports #Users, not true user-item interaction nnz",
            "proxy",
            0.7 * 1000.0,
        ),
    ]


def _positive_int(raw: object, label: str) -> int:
    if not isinstance(raw, int) or isinstance(raw, bool) or raw <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return raw


def _nonnegative_int(raw: object, label: str) -> int:
    if not isinstance(raw, int) or isinstance(raw, bool) or raw < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return raw


def _positive_float_tuple(raw: object, label: str) -> Tuple[float, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{label} must be a non-empty array")
    values: List[float] = []
    for value in raw:
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{label} must contain positive numbers")
        values.append(float(value))
    return tuple(values)


def _parse_fhe_params(raw: object, label: str) -> FheParams:
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be an object")
    return FheParams(
        rlwe_dimension=_positive_int(
            raw.get("rlwe_dimension"), f"{label}.rlwe_dimension"
        ),
        rlwe_modulus_bits=_positive_int(
            raw.get("rlwe_modulus_bits"), f"{label}.rlwe_modulus_bits"
        ),
        rgsw_modulus_bits=_positive_int(
            raw.get("rgsw_modulus_bits"), f"{label}.rgsw_modulus_bits"
        ),
        rgsw_special_prime_bits=_positive_int(
            raw.get("rgsw_special_prime_bits"),
            f"{label}.rgsw_special_prime_bits",
        ),
        rgsw_decomposition_digits=_positive_int(
            raw.get("rgsw_decomposition_digits"),
            f"{label}.rgsw_decomposition_digits",
        ),
    )


def load_software_calibration(path: Union[str, Path]) -> SoftwareCalibration:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("software calibration must be a JSON object")

    schema_version = raw.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != SOFTWARE_CALIBRATION_SCHEMA_VERSION
    ):
        raise ValueError(
            f"unsupported software calibration schema version: {schema_version!r}"
        )

    environment_raw = raw.get("environment")
    if not isinstance(environment_raw, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in environment_raw.items()
    ):
        raise ValueError("environment must map strings to strings")
    command = raw.get("command")
    if not isinstance(command, str) or not command:
        raise ValueError("command must be a non-empty string")

    samples_raw = raw.get("samples")
    if not isinstance(samples_raw, list) or not samples_raw:
        raise ValueError("samples must be a non-empty array")
    samples: List[SoftwareCalibrationSample] = []
    for index, sample_raw in enumerate(samples_raw):
        label = f"samples[{index}]"
        if not isinstance(sample_raw, dict):
            raise ValueError(f"{label} must be an object")
        name = sample_raw.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label}.name must be a non-empty string")
        samples.append(
            SoftwareCalibrationSample(
                name=name,
                effective_nnz=_positive_int(
                    sample_raw.get("effective_nnz"), f"{label}.effective_nnz"
                ),
                selector_count=_nonnegative_int(
                    sample_raw.get("selector_count"), f"{label}.selector_count"
                ),
                row_ciphertexts=_positive_int(
                    sample_raw.get("row_ciphertexts"), f"{label}.row_ciphertexts"
                ),
                elapsed_samples_s=_positive_float_tuple(
                    sample_raw.get("elapsed_samples_s"), f"{label}.elapsed_samples_s"
                ),
            )
        )

    return SoftwareCalibration(
        schema_version=schema_version,
        environment=dict(environment_raw),
        command=command,
        fhe=_parse_fhe_params(raw.get("fhe_parameters"), "fhe_parameters"),
        external_products_per_hswitch=_positive_int(
            raw.get("external_products_per_hswitch"),
            "external_products_per_hswitch",
        ),
        samples=tuple(samples),
    )


def load_nnz_overrides(path: Optional[str]) -> Dict[str, int]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("nnz override file must be a JSON object")
    overrides: Dict[str, int] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise ValueError("nnz override keys must be strings")
        overrides[key] = _positive_int(value, f"nnz override {key!r}")
    return overrides


def apply_overrides(
    workloads: Sequence[Workload], overrides: Mapping[str, int]
) -> List[Workload]:
    result: List[Workload] = []
    for workload in workloads:
        if workload.dataset in overrides:
            result.append(
                replace(
                    workload,
                    effective_nnz=overrides[workload.dataset],
                    nnz_source="user JSON override",
                    nnz_confidence="override",
                )
            )
        else:
            result.append(workload)
    return result


def padded_network_size(effective_nnz: int) -> int:
    if effective_nnz <= 0:
        raise ValueError("effective_nnz must be positive")
    if effective_nnz == 1:
        return 1
    return 1 << (effective_nnz - 1).bit_length()


def ceil_log2_for_work(effective_nnz: int) -> int:
    return (padded_network_size(effective_nnz)).bit_length() - 1


def benes_layer_count(effective_nnz: int) -> int:
    """Return the layer count after next-power-of-two padding."""

    log_factor = ceil_log2_for_work(effective_nnz)
    return 0 if log_factor == 0 else 2 * log_factor - 1


def benes_selector_count(effective_nnz: int) -> int:
    network_size = padded_network_size(effective_nnz)
    if network_size == 1:
        return 0
    return (network_size // 2) * benes_layer_count(effective_nnz)


def row_ciphertext_count(workload: Workload, fhe: FheParams) -> int:
    width = workload.parameters.get("hidden_dim")
    if width is None:
        width = workload.parameters.get("embedding_dim")
    if width is None:
        width = workload.parameters.get("dim")
    if not isinstance(width, int):
        return 1
    return max(1, math.ceil(width / fhe.rlwe_dimension))


def ntt_complexity_factor(fhe: FheParams) -> float:
    """Normalize polynomial work to the paper N=4096 target."""

    return (fhe.rlwe_dimension * math.log2(fhe.rlwe_dimension)) / (4096 * 12)


def rgsw_external_product_factor(fhe: FheParams) -> float:
    """Normalize one RGSW EP to the paper Q/q'/l target."""

    return (
        (fhe.rgsw_modulus_bits / 128)
        * (fhe.rgsw_special_prime_bits / 64)
        * (fhe.rgsw_decomposition_digits / 2)
    )


def rlwe_apply_factor(fhe: FheParams) -> float:
    """Normalize one RLWE Apply multiplication to the paper q target."""

    return fhe.rlwe_modulus_bits / 64


def normalized_work_units(
    effective_nnz: int,
    selector_count: int,
    row_ciphertexts: int,
    fhe: FheParams,
    external_products_per_hswitch: int,
) -> float:
    if effective_nnz <= 0:
        raise ValueError("effective_nnz must be positive")
    if selector_count < 0:
        raise ValueError("selector_count must be nonnegative")
    if row_ciphertexts <= 0:
        raise ValueError("row_ciphertexts must be positive")
    if external_products_per_hswitch <= 0:
        raise ValueError("external_products_per_hswitch must be positive")

    gather_units = (
        selector_count
        * external_products_per_hswitch
        * rgsw_external_product_factor(fhe)
    )
    apply_units = effective_nnz * rlwe_apply_factor(fhe)
    return row_ciphertexts * ntt_complexity_factor(fhe) * (gather_units + apply_units)


def work_units(
    workload: Workload,
    fhe: FheParams,
    external_products_per_hswitch: int = DEFAULT_EXTERNAL_PRODUCTS_PER_HSWITCH,
) -> float:
    return normalized_work_units(
        workload.effective_nnz,
        benes_selector_count(workload.effective_nnz),
        row_ciphertext_count(workload, fhe),
        fhe,
        external_products_per_hswitch,
    )


def calibration_sample_work_units(
    sample: SoftwareCalibrationSample, calibration: SoftwareCalibration
) -> float:
    return normalized_work_units(
        sample.effective_nnz,
        sample.selector_count,
        sample.row_ciphertexts,
        calibration.fhe,
        calibration.external_products_per_hswitch,
    )


def calibrate_seconds_per_unit(calibration: SoftwareCalibration) -> float:
    """Return the median cost across independently measured benchmark samples."""

    costs = [
        sample.median_elapsed_s / calibration_sample_work_units(sample, calibration)
        for sample in calibration.samples
    ]
    return statistics.median(costs)


def percent_delta(predicted: float, paper: float) -> float:
    if paper == 0.0:
        return math.nan
    return (predicted - paper) / paper * 100.0


def predict(
    workloads: Sequence[Workload],
    fhe: FheParams,
    seconds_per_work_unit: float,
    external_products_per_hswitch: int = DEFAULT_EXTERNAL_PRODUCTS_PER_HSWITCH,
) -> List[Prediction]:
    if seconds_per_work_unit <= 0:
        raise ValueError("seconds_per_work_unit must be positive")
    predictions: List[Prediction] = []
    for workload in workloads:
        units = work_units(workload, fhe, external_products_per_hswitch)
        predicted = seconds_per_work_unit * units
        predictions.append(
            Prediction(
                workload=workload,
                row_ciphertexts=row_ciphertext_count(workload, fhe),
                network_size=padded_network_size(workload.effective_nnz),
                log_factor=ceil_log2_for_work(workload.effective_nnz),
                benes_layers=benes_layer_count(workload.effective_nnz),
                selector_count=benes_selector_count(workload.effective_nnz),
                work_units=units,
                predicted_software_s=predicted,
                software_delta_s=predicted - workload.paper_sparsee_software_s,
                software_delta_percent=percent_delta(
                    predicted, workload.paper_sparsee_software_s
                ),
            )
        )
    return predictions


def fmt_float(value: float, digits: int = 4) -> str:
    if math.isnan(value):
        return "nan"
    if value == 0:
        return "0"
    abs_value = abs(value)
    if abs_value >= 1000:
        return f"{value:.1f}"
    if abs_value >= 10:
        return f"{value:.2f}"
    if abs_value >= 1:
        return f"{value:.3f}"
    return f"{value:.{digits}f}"


def _environment_text(environment: Mapping[str, str]) -> str:
    return ", ".join(f"{key}={value}" for key, value in environment.items())


def print_markdown(
    predictions: Sequence[Prediction],
    fhe: FheParams,
    calibration: SoftwareCalibration,
    seconds_per_work_unit: float,
    coefficient_source: str,
    external_products_per_hswitch: int,
) -> None:
    print("# SparseE Software-Time Linear Extrapolation")
    print()
    print("## Target FHE Parameters")
    print()
    print(f"- RLWE dimension N: {fhe.rlwe_dimension}")
    print(f"- RLWE modulus q: {fhe.rlwe_modulus_bits} bits")
    print(f"- RGSW modulus Q: {fhe.rgsw_modulus_bits} bits")
    print(f"- RGSW special prime q': {fhe.rgsw_special_prime_bits} bits")
    print(f"- RGSW decomposition digits: {fhe.rgsw_decomposition_digits}")
    print(f"- External products per HSwitch: {external_products_per_hswitch}")
    print()
    print("## Software Calibration")
    print()
    print(f"- Environment: {_environment_text(calibration.environment)}")
    print(f"- Benchmark command: `{calibration.command}`")
    print(f"- Coefficient source: {coefficient_source}")
    print(f"- Linear coefficient: {seconds_per_work_unit:.9g} s/work-unit")
    print(
        "- Paper software latencies are comparison targets only and do not "
        "participate in calibration."
    )
    print()
    headers = [
        "Benchmark sample",
        "nnz/Apply",
        "Selectors/EP",
        "Median server ms",
        "Normalized work",
        "us/work",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for sample in calibration.samples:
        units = calibration_sample_work_units(sample, calibration)
        row = [
            sample.name,
            str(sample.effective_nnz),
            str(sample.selector_count * calibration.external_products_per_hswitch),
            f"{sample.median_elapsed_s * 1000:.3f}",
            f"{units:.3f}",
            f"{sample.median_elapsed_s / units * 1_000_000:.3f}",
        ]
        print("| " + " | ".join(row) + " |")
    print()
    print("## Linear Model")
    print()
    print(
        "selector_count = (padded_network_size / 2) * "
        "(2*ceil(log2(effective_nnz))-1)."
    )
    print(
        "work_units = row_ciphertexts * normalized_NlogN * "
        "(selector_count*EPs_per_HSwitch*RGSW_factor + "
        "effective_nnz*Apply_factor)."
    )
    print("predicted_software_seconds = coefficient * work_units.")
    print(
        "Scatter and ciphertext additions are included in measured Server.Execute "
        "time and are absorbed into the fitted coefficient."
    )
    print()
    print("## Interpretation Limits")
    print()
    print(
        "- Calibration samples cover 7-13 nnz and 20-56 selectors; "
        "paper-scale rows can "
        "contain tens of millions of selectors, so this is a long-range linear "
        "extrapolation rather than a full-scale run."
    )
    print(
        "- EP and Apply use normalized operation weights with one fitted "
        "coefficient rather than separate primitive coefficients."
    )
    print(
        "- Calibration ran with the executable 64/125/61-bit aggregate profile. "
        "Conversion to the paper 64/128/64-bit target assumes linear bit-width "
        "scaling."
    )
    print()
    print(
        "Rows marked proxy do not have a true paper-reported nnz. The table's "
        "workload-count column is used unless a JSON override is supplied."
    )
    print()
    print("## Software Comparison")
    print()
    headers = [
        "Domain",
        "Dataset",
        "nnz/effective",
        "nnz conf.",
        "Network",
        "Selectors",
        "Paper SW s",
        "Pred SW s",
        "Delta s",
        "Delta %",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for pred in predictions:
        workload = pred.workload
        row = [
            workload.domain,
            workload.dataset,
            str(workload.effective_nnz),
            workload.nnz_confidence,
            str(pred.network_size),
            str(pred.selector_count),
            fmt_float(workload.paper_sparsee_software_s),
            fmt_float(pred.predicted_software_s),
            fmt_float(pred.software_delta_s),
            fmt_float(pred.software_delta_percent, 2),
        ]
        print("| " + " | ".join(row) + " |")
    print()
    print("## nnz Sources")
    print()
    for pred in predictions:
        workload = pred.workload
        print(
            f"- {workload.domain}/{workload.dataset}: {workload.nnz_source}. "
            f"Parameters: {dict(workload.parameters)}"
        )


def print_csv(predictions: Sequence[Prediction], output: TextIO) -> None:
    writer = csv.writer(output)
    writer.writerow(
        [
            "domain",
            "dataset",
            "effective_nnz",
            "nnz_confidence",
            "nnz_source",
            "network_size",
            "log_factor",
            "benes_layers",
            "selector_count",
            "row_ciphertexts",
            "work_units",
            "paper_sparsee_software_s",
            "predicted_software_s",
            "software_delta_s",
            "software_delta_percent",
        ]
    )
    for pred in predictions:
        workload = pred.workload
        writer.writerow(
            [
                workload.domain,
                workload.dataset,
                workload.effective_nnz,
                workload.nnz_confidence,
                workload.nnz_source,
                pred.network_size,
                pred.log_factor,
                pred.benes_layers,
                pred.selector_count,
                pred.row_ciphertexts,
                pred.work_units,
                workload.paper_sparsee_software_s,
                pred.predicted_software_s,
                pred.software_delta_s,
                pred.software_delta_percent,
            ]
        )


def prediction_to_dict(pred: Prediction) -> Dict[str, object]:
    workload = pred.workload
    return {
        "domain": workload.domain,
        "dataset": workload.dataset,
        "parameters": dict(workload.parameters),
        "effective_nnz": workload.effective_nnz,
        "nnz_source": workload.nnz_source,
        "nnz_confidence": workload.nnz_confidence,
        "network_size": pred.network_size,
        "log_factor": pred.log_factor,
        "benes_layers": pred.benes_layers,
        "selector_count": pred.selector_count,
        "row_ciphertexts": pred.row_ciphertexts,
        "work_units": pred.work_units,
        "paper_sparsee_software_s": workload.paper_sparsee_software_s,
        "predicted_software_s": pred.predicted_software_s,
        "software_delta_s": pred.software_delta_s,
        "software_delta_percent": pred.software_delta_percent,
    }


def calibration_to_dict(calibration: SoftwareCalibration) -> Dict[str, object]:
    return {
        "schema_version": calibration.schema_version,
        "environment": dict(calibration.environment),
        "command": calibration.command,
        "fhe_parameters": calibration.fhe.__dict__,
        "external_products_per_hswitch": calibration.external_products_per_hswitch,
        "samples": [
            {
                "name": sample.name,
                "effective_nnz": sample.effective_nnz,
                "selector_count": sample.selector_count,
                "row_ciphertexts": sample.row_ciphertexts,
                "elapsed_samples_s": list(sample.elapsed_samples_s),
                "median_elapsed_s": sample.median_elapsed_s,
                "normalized_work_units": calibration_sample_work_units(
                    sample, calibration
                ),
            }
            for sample in calibration.samples
        ],
    }


def build_json_payload(
    predictions: Sequence[Prediction],
    fhe: FheParams,
    calibration: SoftwareCalibration,
    seconds_per_work_unit: float,
    coefficient_source: str,
    external_products_per_hswitch: int,
) -> Dict[str, object]:
    return {
        "target_fhe_params": fhe.__dict__,
        "software_calibration": calibration_to_dict(calibration),
        "coefficient_source": coefficient_source,
        "seconds_per_work_unit": seconds_per_work_unit,
        "external_products_per_hswitch": external_products_per_hswitch,
        "predictions": [prediction_to_dict(pred) for pred in predictions],
    }


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive finite number")
    return parsed


def power_of_two(value: str) -> int:
    parsed = positive_int(value)
    if parsed < 2 or parsed & (parsed - 1):
        raise argparse.ArgumentTypeError(
            "RLWE dimension must be a power of two greater than one"
        )
    return parsed


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        choices=("markdown", "csv", "json"),
        default="markdown",
        help="output format",
    )
    parser.add_argument(
        "--calibration-json",
        default=str(DEFAULT_CALIBRATION_PATH),
        help="software benchmark calibration JSON",
    )
    parser.add_argument(
        "--seconds-per-work-unit",
        type=positive_float,
        help="override the coefficient derived from the calibration benchmark",
    )
    parser.add_argument(
        "--nnz-override-json",
        help=(
            "optional JSON object mapping dataset labels such as MN10 or ML "
            "to true nnz"
        ),
    )
    parser.add_argument(
        "--strict-direct-nnz",
        action="store_true",
        help="exit nonzero if any workload uses a proxy nnz",
    )
    parser.add_argument("--rlwe-dimension", type=power_of_two, default=4096)
    parser.add_argument("--rlwe-modulus-bits", type=positive_int, default=64)
    parser.add_argument("--rgsw-modulus-bits", type=positive_int, default=128)
    parser.add_argument("--rgsw-special-prime-bits", type=positive_int, default=64)
    parser.add_argument("--rgsw-decomposition-digits", type=positive_int, default=2)
    parser.add_argument(
        "--external-products-per-hswitch",
        type=positive_int,
        default=DEFAULT_EXTERNAL_PRODUCTS_PER_HSWITCH,
        help="RGSW external products used by each HSwitch",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    fhe = FheParams(
        rlwe_dimension=args.rlwe_dimension,
        rlwe_modulus_bits=args.rlwe_modulus_bits,
        rgsw_modulus_bits=args.rgsw_modulus_bits,
        rgsw_special_prime_bits=args.rgsw_special_prime_bits,
        rgsw_decomposition_digits=args.rgsw_decomposition_digits,
    )
    calibration = load_software_calibration(args.calibration_json)
    workloads = apply_overrides(
        paper_workloads(), load_nnz_overrides(args.nnz_override_json)
    )

    proxy_rows = [
        workload for workload in workloads if workload.nnz_confidence == "proxy"
    ]
    if args.strict_direct_nnz and proxy_rows:
        names = ", ".join(
            f"{workload.domain}/{workload.dataset}" for workload in proxy_rows
        )
        print(f"error: proxy nnz rows present: {names}", file=sys.stderr)
        return 2

    if args.seconds_per_work_unit is None:
        seconds_per_work_unit = calibrate_seconds_per_unit(calibration)
        coefficient_source = "median of measured benchmark sample costs"
    else:
        seconds_per_work_unit = args.seconds_per_work_unit
        coefficient_source = "command-line override"

    predictions = predict(
        workloads,
        fhe,
        seconds_per_work_unit,
        args.external_products_per_hswitch,
    )

    if args.format == "markdown":
        print_markdown(
            predictions,
            fhe,
            calibration,
            seconds_per_work_unit,
            coefficient_source,
            args.external_products_per_hswitch,
        )
    elif args.format == "csv":
        print_csv(predictions, sys.stdout)
    else:
        json.dump(
            build_json_payload(
                predictions,
                fhe,
                calibration,
                seconds_per_work_unit,
                coefficient_source,
                args.external_products_per_hswitch,
            ),
            sys.stdout,
            indent=2,
        )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
