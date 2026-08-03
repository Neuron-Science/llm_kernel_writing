"""Verified streaming residual RMSNorm lab used by Section 4.2.

Run on one available NeuronCore:

    NEURON_RT_VISIBLE_CORES=0 /opt/nki-venv/bin/python rmsnorm_agent_lab.py \
        --kernel both
"""

import argparse
from dataclasses import dataclass

import numpy as np
from ml_dtypes import bfloat16
import nki
import nki.isa as nisa
import nki.language as nl


TOKENS = 1024
TILE_TOKENS = 128
HIDDEN = 4096
NUM_TILES = TOKENS // TILE_TOKENS
EPS = 1e-6
ATOL = 2**-7
RTOL = 1e-2
MAX_ABS = 2**-5
MAX_MEAN_ABS = 5e-5
MIN_COSINE = 0.9999


@nki.jit
def residual_rms_norm_kernel(x, residual, weight):
    """Compute a prefill-sized residual add followed by weighted RMSNorm."""
    assert x.shape == residual.shape == (TOKENS, HIDDEN)
    assert x.dtype == residual.dtype == nl.bfloat16
    assert weight.shape == (HIDDEN,)
    assert weight.dtype == nl.bfloat16

    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    weight_row = nl.load(weight.reshape((1, HIDDEN)))
    weight_tile = nl.broadcast_to(
        weight_row,
        shape=(TILE_TOKENS, HIDDEN),
    )

    for tile_idx in range(NUM_TILES):
        start = tile_idx * TILE_TOKENS
        stop = start + TILE_TOKENS
        x_tile = nl.load(x[start:stop, :])
        residual_tile = nl.load(residual[start:stop, :])
        summed = nl.add(x_tile, residual_tile, dtype=nl.float32)
        mean_square = nl.mean(
            nl.square(summed, dtype=nl.float32),
            axis=1,
            dtype=nl.float32,
            keepdims=True,
        )
        inverse_rms = nl.rsqrt(
            nl.add(mean_square, EPS, dtype=nl.float32),
            dtype=nl.float32,
        )
        inverse_rms = nl.broadcast_to(
            inverse_rms,
            shape=(TILE_TOKENS, HIDDEN),
        )
        normalized = nl.multiply(
            summed,
            inverse_rms,
            dtype=nl.float32,
        )
        y_tile = nl.multiply(normalized, weight_tile, dtype=x.dtype)
        nl.store(out[start:stop, :], value=y_tile)
    return out


@nki.jit
def residual_rms_norm_fused_kernel(x, residual, weight):
    """Use the profile-driven instruction-level candidate."""
    assert x.shape == residual.shape == (TOKENS, HIDDEN)
    assert x.dtype == residual.dtype == nl.bfloat16
    assert weight.shape == (HIDDEN,)
    assert weight.dtype == nl.bfloat16

    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    weight_row = nl.load(weight.reshape((1, HIDDEN)))
    weight_tile = nl.broadcast_to(
        weight_row,
        shape=(TILE_TOKENS, HIDDEN),
    )

    for tile_idx in range(NUM_TILES):
        start = tile_idx * TILE_TOKENS
        stop = start + TILE_TOKENS
        x_tile = nl.ndarray(
            (TILE_TOKENS, HIDDEN),
            dtype=x.dtype,
            buffer=nl.sbuf,
        )
        residual_tile = nl.ndarray(
            (TILE_TOKENS, HIDDEN),
            dtype=residual.dtype,
            buffer=nl.sbuf,
        )
        summed = nl.ndarray(
            (TILE_TOKENS, HIDDEN),
            dtype=nl.float32,
            buffer=nl.sbuf,
        )
        square_scratch = nl.ndarray(
            (TILE_TOKENS, HIDDEN),
            dtype=nl.float32,
            buffer=nl.sbuf,
        )
        sum_square = nl.ndarray(
            (TILE_TOKENS, 1),
            dtype=nl.float32,
            buffer=nl.sbuf,
        )

        nisa.dma_copy(dst=x_tile, src=x[start:stop, :])
        nisa.dma_copy(
            dst=residual_tile,
            src=residual[start:stop, :],
        )
        nisa.tensor_tensor(
            dst=summed,
            data1=x_tile,
            data2=residual_tile,
            op=nl.add,
        )
        nisa.activation(
            dst=square_scratch,
            op=nl.square,
            data=summed,
            reduce_op=nl.add,
            reduce_res=sum_square,
            reduce_cmd=nisa.reduce_cmd.reset_reduce,
        )
        nisa.activation(
            dst=sum_square,
            op=nl.rsqrt,
            data=sum_square,
            scale=1.0 / HIDDEN,
            bias=EPS,
        )
        nisa.tensor_scalar(
            dst=summed,
            data=summed,
            op0=nl.multiply,
            operand0=sum_square,
        )
        nisa.tensor_tensor(
            dst=x_tile,
            data1=summed,
            data2=weight_tile,
            op=nl.multiply,
        )
        nisa.dma_copy(dst=out[start:stop, :], src=x_tile)
    return out


@nki.jit
def residual_rms_norm_framework_kernel(x, residual, weight):
    """Reuse scratch storage for framework custom-op compilation."""
    assert x.shape == residual.shape == (TOKENS, HIDDEN)
    assert x.dtype == residual.dtype == nl.bfloat16
    assert weight.shape == (HIDDEN,)
    assert weight.dtype == nl.bfloat16

    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    weight_row = nl.load(weight.reshape((1, HIDDEN)))
    weight_tile = nl.broadcast_to(
        weight_row,
        shape=(TILE_TOKENS, HIDDEN),
    )
    x_tile = nl.ndarray(
        (TILE_TOKENS, HIDDEN),
        dtype=x.dtype,
        buffer=nl.sbuf,
    )
    residual_tile = nl.ndarray(
        (TILE_TOKENS, HIDDEN),
        dtype=residual.dtype,
        buffer=nl.sbuf,
    )
    summed = nl.ndarray(
        (TILE_TOKENS, HIDDEN),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    square_scratch = nl.ndarray(
        (TILE_TOKENS, HIDDEN),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    sum_square = nl.ndarray(
        (TILE_TOKENS, 1),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )

    for tile_idx in range(NUM_TILES):
        start = tile_idx * TILE_TOKENS
        stop = start + TILE_TOKENS
        nisa.dma_copy(dst=x_tile, src=x[start:stop, :])
        nisa.dma_copy(
            dst=residual_tile,
            src=residual[start:stop, :],
        )
        nisa.tensor_tensor(
            dst=summed,
            data1=x_tile,
            data2=residual_tile,
            op=nl.add,
        )
        nisa.activation(
            dst=square_scratch,
            op=nl.square,
            data=summed,
            reduce_op=nl.add,
            reduce_res=sum_square,
            reduce_cmd=nisa.reduce_cmd.reset_reduce,
        )
        nisa.activation(
            dst=sum_square,
            op=nl.rsqrt,
            data=sum_square,
            scale=1.0 / HIDDEN,
            bias=EPS,
        )
        nisa.tensor_scalar(
            dst=summed,
            data=summed,
            op0=nl.multiply,
            operand0=sum_square,
        )
        nisa.tensor_tensor(
            dst=x_tile,
            data1=summed,
            data2=weight_tile,
            op=nl.multiply,
        )
        nisa.dma_copy(dst=out[start:stop, :], src=x_tile)
    return out


def residual_rms_norm_reference(x, residual, weight):
    """NumPy reference with the fixed BF16 and FP32 dtype contract."""
    summed = x.astype(np.float32) + residual.astype(np.float32)
    mean_square = np.mean(
        np.square(summed),
        axis=1,
        dtype=np.float32,
        keepdims=True,
    )
    normalized = summed / np.sqrt(mean_square + np.float32(EPS))
    return (normalized * weight.astype(np.float32)).astype(bfloat16)


def cosine_similarity(actual, expected):
    """Return a defined score for zero vectors as well as nonzero vectors."""
    actual_f64 = actual.astype(np.float64).ravel()
    expected_f64 = expected.astype(np.float64).ravel()
    actual_norm = np.linalg.norm(actual_f64)
    expected_norm = np.linalg.norm(expected_f64)
    if actual_norm == 0.0 or expected_norm == 0.0:
        return 1.0 if actual_norm == expected_norm else 0.0
    return float(
        np.dot(actual_f64, expected_f64)
        / (actual_norm * expected_norm)
    )


@dataclass(frozen=True)
class ErrorMetrics:
    allclose: bool
    max_abs: float
    l1: float
    linf: float
    mean_abs: float
    cosine: float


def compare(actual, expected):
    """Check structural invariants and return complementary error metrics."""
    assert actual.shape == expected.shape == (TOKENS, HIDDEN)
    assert actual.dtype == expected.dtype == bfloat16
    assert actual.flags.c_contiguous and expected.flags.c_contiguous
    assert np.isfinite(actual.astype(np.float32)).all()
    assert np.isfinite(expected.astype(np.float32)).all()

    error = actual.astype(np.float32) - expected.astype(np.float32)
    abs_error = np.abs(error)
    return ErrorMetrics(
        allclose=bool(
            np.allclose(actual, expected, atol=ATOL, rtol=RTOL)
        ),
        max_abs=float(np.max(abs_error)),
        l1=float(np.sum(abs_error, dtype=np.float64)),
        linf=float(np.linalg.norm(error.ravel(), ord=np.inf)),
        mean_abs=float(np.mean(abs_error, dtype=np.float64)),
        cosine=cosine_similarity(actual, expected),
    )


def test_inputs():
    """Yield adversarial and seeded inputs without retaining every large case."""
    shape = (TOKENS, HIDDEN)
    unit_weight = np.ones((HIDDEN,), dtype=bfloat16)
    ramp_weight = np.linspace(
        0.5,
        1.5,
        HIDDEN,
        dtype=np.float32,
    ).astype(bfloat16)

    yield (
        "zeros",
        np.zeros(shape, dtype=bfloat16),
        np.zeros(shape, dtype=bfloat16),
        unit_weight,
    )
    yield (
        "constant",
        np.full(shape, 3.0, dtype=bfloat16),
        np.full(shape, -1.0, dtype=bfloat16),
        ramp_weight,
    )

    alternating = np.ones(shape, dtype=np.float32)
    alternating[:, 1::2] = -1.0
    yield (
        "alternating",
        alternating.astype(bfloat16),
        np.full(shape, 0.25, dtype=bfloat16),
        ramp_weight,
    )
    yield (
        "small",
        np.full(shape, 2**-12, dtype=bfloat16),
        np.full(shape, -(2**-13), dtype=bfloat16),
        ramp_weight,
    )
    yield (
        "large",
        np.full(shape, 2**8, dtype=bfloat16),
        np.full(shape, -(2**7), dtype=bfloat16),
        ramp_weight,
    )

    cancellation_rng = np.random.default_rng(11)
    cancellation_x = cancellation_rng.normal(size=shape).astype(bfloat16)
    yield (
        "cancellation",
        cancellation_x,
        (-cancellation_x.astype(np.float32)).astype(bfloat16),
        ramp_weight,
    )

    for seed in (7, 2026):
        rng = np.random.default_rng(seed)
        yield (
            f"random_seed_{seed}",
            rng.normal(size=shape).astype(bfloat16),
            rng.normal(size=shape).astype(bfloat16),
            rng.uniform(0.75, 1.25, size=(HIDDEN,)).astype(bfloat16),
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kernel",
        choices=("baseline", "fused", "framework", "both"),
        default="baseline",
        help="Kernel implementation to compile and validate.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    kernels = {
        "baseline": residual_rms_norm_kernel,
        "fused": residual_rms_norm_fused_kernel,
        "framework": residual_rms_norm_framework_kernel,
    }
    kernel_names = (
        ("baseline", "fused")
        if args.kernel == "both"
        else (args.kernel,)
    )

    for kernel_name in kernel_names:
        kernel = kernels[kernel_name]
        print(f"[{kernel_name}]")
        for case_name, x, residual, weight in test_inputs():
            assert x.flags.c_contiguous
            assert residual.flags.c_contiguous
            assert weight.flags.c_contiguous
            x_before = x.copy()
            residual_before = residual.copy()
            weight_before = weight.copy()

            actual = kernel(x, residual, weight)

            assert np.array_equal(x, x_before)
            assert np.array_equal(residual, residual_before)
            assert np.array_equal(weight, weight_before)
            assert not np.shares_memory(actual, x)
            assert not np.shares_memory(actual, residual)
            expected = residual_rms_norm_reference(
                x,
                residual,
                weight,
            )
            metrics = compare(actual, expected)
            print(
                f"{case_name:16s} allclose={metrics.allclose} "
                f"max_abs={metrics.max_abs:.6f} "
                f"l1={metrics.l1:.6f} "
                f"linf={metrics.linf:.6f} "
                f"mean_abs={metrics.mean_abs:.6e} "
                f"cosine={metrics.cosine:.10f}"
            )
            assert metrics.allclose
            assert metrics.max_abs == metrics.linf
            assert metrics.linf <= MAX_ABS
            assert metrics.mean_abs <= MAX_MEAN_ABS
            assert metrics.cosine >= MIN_COSINE


if __name__ == "__main__":
    main()
