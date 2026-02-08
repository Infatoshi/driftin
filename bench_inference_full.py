"""Comprehensive inference benchmark: variable batch size, optimization techniques.

Measures latency and throughput with proper CUDA event timing, warmup, and percentiles.
Tests: baseline, torch.compile (default + max-autotune), CUDA graphs, and combinations.

Usage: uv run python3 bench_inference_full.py
"""

import torch
import torch.nn as nn
import numpy as np
import time
import csv
import os
import json
import gc
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Tuple

from drifting_vs_diffusion.config import UNetConfig
from drifting_vs_diffusion.models.unet import UNet

# ── Setup ────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda")
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
N_WARMUP = 20
N_ITERS = 100
OUT_DIR = "outputs/inference_bench"

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


@dataclass
class BenchResult:
    method: str
    batch_size: int
    latencies_ms: List[float]  # per-run latencies (each run = 1 batch)

    @property
    def p50(self): return float(np.percentile(self.latencies_ms, 50))
    @property
    def p95(self): return float(np.percentile(self.latencies_ms, 95))
    @property
    def p99(self): return float(np.percentile(self.latencies_ms, 99))
    @property
    def mean(self): return float(np.mean(self.latencies_ms))
    @property
    def std(self): return float(np.std(self.latencies_ms))
    @property
    def throughput(self): return self.batch_size / (self.mean / 1000)  # img/s
    @property
    def per_image_ms(self): return self.mean / self.batch_size


def make_model():
    cfg = UNetConfig()
    model = UNet(
        in_ch=cfg.in_ch, out_ch=cfg.out_ch, base_ch=cfg.base_ch,
        ch_mult=cfg.ch_mult, num_res_blocks=cfg.num_res_blocks,
        attn_resolutions=cfg.attn_resolutions, dropout=0.0,
        num_heads=cfg.num_heads,
    )
    return model


def cuda_event_timing(model, batch_size, n_warmup, n_iters, dtype=torch.bfloat16,
                      use_cuda_graph=False):
    """Benchmark with CUDA events for accurate GPU timing.

    Returns list of per-iteration latencies in ms.
    """
    # Pre-allocate static input for CUDA graphs
    static_input = torch.randn(batch_size, 3, 32, 32, device=DEVICE)

    # Warmup
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        for _ in range(n_warmup):
            static_input.normal_()
            _ = model(static_input)
    torch.cuda.synchronize()

    # CUDA graph capture (if requested)
    graph = None
    static_output = None
    if use_cuda_graph:
        # Capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
                static_output = model(static_input)
        torch.cuda.synchronize()

    # Timed iterations using CUDA events
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]

    with torch.no_grad():
        for i in range(n_iters):
            if not use_cuda_graph:
                static_input.normal_()

            start_events[i].record()
            if use_cuda_graph:
                graph.replay()
            else:
                with torch.amp.autocast("cuda", dtype=dtype):
                    _ = model(static_input)
            end_events[i].record()

    torch.cuda.synchronize()
    latencies = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return latencies


def bench_method(name, model, batch_sizes, use_cuda_graph=False, dtype=torch.bfloat16):
    """Run benchmark across all batch sizes for a given method."""
    results = []
    for bs in batch_sizes:
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()

        try:
            latencies = cuda_event_timing(
                model, bs, N_WARMUP, N_ITERS,
                dtype=dtype, use_cuda_graph=use_cuda_graph,
            )
            peak_mem = torch.cuda.max_memory_allocated() / 1e9
            r = BenchResult(method=name, batch_size=bs, latencies_ms=latencies)
            print(f"  bs={bs:>3d} | {r.mean:>7.2f} ms (p50={r.p50:.2f}, p95={r.p95:.2f}, p99={r.p99:.2f}) | "
                  f"{r.per_image_ms:.2f} ms/img | {r.throughput:.0f} img/s | {peak_mem:.2f} GB")
            results.append(r)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  bs={bs:>3d} | OOM")
                torch.cuda.empty_cache()
                break
            raise
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"Batch sizes: {BATCH_SIZES}")
    print(f"Warmup: {N_WARMUP}, Iterations: {N_ITERS}")
    print(f"Timing: CUDA events (GPU-side, no CPU overhead)")
    print()

    all_results = []

    # ── 1. Baseline (eager, bf16) ────────────────────────────────────────
    print("=" * 70)
    print("[1/5] Baseline (eager mode, bf16)")
    print("=" * 70)
    model = make_model().to(DEVICE).eval()
    results = bench_method("eager_bf16", model, BATCH_SIZES)
    all_results.extend(results)
    del model
    torch.cuda.empty_cache()
    print()

    # ── 2. Baseline (eager, fp32) ────────────────────────────────────────
    print("=" * 70)
    print("[2/5] Baseline (eager mode, fp32)")
    print("=" * 70)
    model = make_model().to(DEVICE).eval()
    results = bench_method("eager_fp32", model, BATCH_SIZES, dtype=torch.float32)
    all_results.extend(results)
    del model
    torch.cuda.empty_cache()
    print()

    # ── 3. torch.compile (default) ───────────────────────────────────────
    print("=" * 70)
    print("[3/5] torch.compile (default mode, bf16)")
    print("=" * 70)
    model = make_model().to(DEVICE).eval()
    model = torch.compile(model)
    # Extra warmup for compile
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(5):
            _ = model(torch.randn(1, 3, 32, 32, device=DEVICE))
    torch.cuda.synchronize()
    results = bench_method("compile_default", model, BATCH_SIZES)
    all_results.extend(results)
    del model
    torch.cuda.empty_cache()
    print()

    # ── 4. torch.compile (max-autotune) ──────────────────────────────────
    print("=" * 70)
    print("[4/5] torch.compile (max-autotune, bf16)")
    print("=" * 70)
    model = make_model().to(DEVICE).eval()
    model = torch.compile(model, mode="max-autotune")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(5):
            _ = model(torch.randn(1, 3, 32, 32, device=DEVICE))
    torch.cuda.synchronize()
    results = bench_method("compile_max_autotune", model, BATCH_SIZES)
    all_results.extend(results)
    del model
    torch.cuda.empty_cache()
    print()

    # ── 5. CUDA Graphs (requires static shapes) ─────────────────────────
    print("=" * 70)
    print("[5/5] CUDA Graphs + bf16 (static input, no randomization per iter)")
    print("=" * 70)
    model = make_model().to(DEVICE).eval()
    results = bench_method("cuda_graphs_bf16", model, BATCH_SIZES, use_cuda_graph=True)
    all_results.extend(results)
    del model
    torch.cuda.empty_cache()
    print()

    # ── Summary table ────────────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY: Per-image latency (ms) by method and batch size")
    print("=" * 70)

    methods = []
    seen = set()
    for r in all_results:
        if r.method not in seen:
            methods.append(r.method)
            seen.add(r.method)

    # Header
    header = f"{'Method':<25s}"
    for bs in BATCH_SIZES:
        header += f" | bs={bs:>3d}"
    print(header)
    print("-" * len(header))

    for method in methods:
        row = f"{method:<25s}"
        for bs in BATCH_SIZES:
            match = [r for r in all_results if r.method == method and r.batch_size == bs]
            if match:
                row += f" | {match[0].per_image_ms:>5.2f}"
            else:
                row += f" |   OOM"
        print(row)

    print()
    print("=" * 70)
    print("SUMMARY: Throughput (img/s) by method and batch size")
    print("=" * 70)

    header = f"{'Method':<25s}"
    for bs in BATCH_SIZES:
        header += f" | bs={bs:>4d}"
    print(header)
    print("-" * len(header))

    for method in methods:
        row = f"{method:<25s}"
        for bs in BATCH_SIZES:
            match = [r for r in all_results if r.method == method and r.batch_size == bs]
            if match:
                row += f" | {match[0].throughput:>5.0f}"
            else:
                row += f" |   OOM"
        print(row)

    # ── Save CSV ─────────────────────────────────────────────────────────
    csv_path = os.path.join(OUT_DIR, "inference_bench.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "batch_size", "mean_ms", "p50_ms", "p95_ms", "p99_ms",
                     "std_ms", "per_image_ms", "throughput_imgs"])
        for r in all_results:
            w.writerow([r.method, r.batch_size, f"{r.mean:.3f}", f"{r.p50:.3f}",
                        f"{r.p95:.3f}", f"{r.p99:.3f}", f"{r.std:.3f}",
                        f"{r.per_image_ms:.3f}", f"{r.throughput:.1f}"])
    print(f"\nSaved results to {csv_path}")

    # ── Save JSON for programmatic use ───────────────────────────────────
    json_path = os.path.join(OUT_DIR, "inference_bench.json")
    json_data = {
        "gpu": gpu_name,
        "gpu_mem_gb": round(gpu_mem, 1),
        "n_warmup": N_WARMUP,
        "n_iters": N_ITERS,
        "results": [
            {
                "method": r.method,
                "batch_size": r.batch_size,
                "mean_ms": round(r.mean, 3),
                "p50_ms": round(r.p50, 3),
                "p95_ms": round(r.p95, 3),
                "p99_ms": round(r.p99, 3),
                "per_image_ms": round(r.per_image_ms, 3),
                "throughput_imgs": round(r.throughput, 1),
            }
            for r in all_results
        ],
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved results to {json_path}")


if __name__ == "__main__":
    main()
