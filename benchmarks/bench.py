"""Unified GPU matmul benchmark.

Replaces the old bench_gpu.py / bench_gpu2.py pair. No CSV — just timing
runs and a summary table at the end.

Usage:
  python benchmarks/bench.py --impls cublas_bf16 b1_tc5 --sizes 2048 4096 8192
  python benchmarks/bench.py --impls cublas_bf16 --shapes 64x16384x65536 8192
  python benchmarks/bench.py --impls cublas_bf16          # defaults to SIZES

Selects implementations from the IMPLEMENTATIONS registry below plus any
matmul_cuda_*.py auto-discovered under mymatmul/gpu/{cuda_core,tensor_core}.
"""

import argparse
import importlib
import inspect
import os

import torch
import triton.testing

# ── Registry ─────────────────────────────────────────────────────────────────
# name -> (dotpath, max_size_or_None, dtype_default_float32)
# max_size: skip sizes > max_size for this impl (leaves a blank cell).

IMPLEMENTATIONS = {
    "triton_fp32simt_autotuned": ("mymatmul.gpu.matmul_triton.triton_fp32simt_autotuned", None),
    "torch_matmul":               ("mymatmul.gpu.matmul_torch.matmul_torch",   None),
    "cublas_fp32_notf32":         ("mymatmul.gpu.matmul_torch.matmul_torch_fp32_notf32", None),
    # ── BF16 references ──
    "triton_bf16_autotuned": ("mymatmul.gpu.matmul_triton.triton_bf16_autotuned", None, torch.bfloat16),
    "cublas_bf16":           ("mymatmul.gpu.matmul_torch.matmul_torch_bf16",      None, torch.bfloat16),
    "cutlass_bf16":          ("mymatmul.gpu.cutlass.matmul_cutlass.matmul_cutlass_bf16", None, torch.bfloat16),
    # ── Ada tensor-core series ──
    "tc3":           ("mymatmul.gpu.tensor_core.matmul_cuda_tc3.matmul_tc3",                       None, torch.bfloat16),
    "tc4":           ("mymatmul.gpu.tensor_core.matmul_cuda_tc4.matmul_tc4",                       None, torch.bfloat16),
    "tc5":           ("mymatmul.gpu.tensor_core.matmul_cuda_tc5.matmul_tc5",                       None, torch.bfloat16),
    "tc5rp":         ("mymatmul.gpu.tensor_core.matmul_cuda_tc5rp.matmul_tc5rp",                   None, torch.bfloat16),
    "tc5jit":        ("mymatmul.gpu.tensor_core.matmul_cuda_tc5jit.matmul_tc5jit",                 None, torch.bfloat16),
    "tc5jit_lb":     ("mymatmul.gpu.tensor_core.matmul_cuda_tc5jit_lb.matmul_tc5jit_lb",           None, torch.bfloat16),
    "tc5swz":        ("mymatmul.gpu.tensor_core.matmul_cuda_tc5swz.matmul_tc5swz",                 None, torch.bfloat16),
    "tc5swz_lb":     ("mymatmul.gpu.tensor_core.matmul_cuda_tc5swz_lb.matmul_tc5swz_lb",           None, torch.bfloat16),
    "tc5l2":         ("mymatmul.gpu.tensor_core.matmul_cuda_tc5l2.matmul_tc5l2",                   None, torch.bfloat16),
    "tc5_reg":       ("mymatmul.gpu.tensor_core.matmul_cuda_tc5_reg.matmul_tc5_reg",               None, torch.bfloat16),
    "tc5_regpruned": ("mymatmul.gpu.tensor_core.matmul_cuda_tc5_regpruned.matmul_tc5_regpruned",   None, torch.bfloat16),
    "tc6":           ("mymatmul.gpu.tensor_core.matmul_cuda_tc6.matmul_tc6",                       None, torch.bfloat16),
    "tc6_lb":        ("mymatmul.gpu.tensor_core.matmul_cuda_tc6_lb.matmul_tc6_lb",                 None, torch.bfloat16),
    "tc6_x4b":       ("mymatmul.gpu.tensor_core.matmul_cuda_tc6_x4b.matmul_tc6_x4b",               None, torch.bfloat16),
    "tc7":           ("mymatmul.gpu.tensor_core.matmul_cuda_tc7.matmul_tc7",                       None, torch.bfloat16),
    "tc7_lb":        ("mymatmul.gpu.tensor_core.matmul_cuda_tc7_lb.matmul_tc7_lb",                 None, torch.bfloat16),
    "tc8_4096":      ("mymatmul.gpu.tensor_core.matmul_cuda_tc8_4096.matmul_tc8_4096",             None, torch.bfloat16),
    "tc8_4096_ptx":  ("mymatmul.gpu.tensor_core.matmul_cuda_tc8_4096_ptx.matmul_tc8_4096_ptx",     None, torch.bfloat16),
    "tc8_gemini":    ("mymatmul.gpu.tensor_core.matmul_cuda_tc8_gemini.matmul_tc8_gemini",         None, torch.bfloat16),
    "tc8g":          ("mymatmul.gpu.tensor_core.matmul_cuda_tc8g.matmul_tc8g",                     None, torch.bfloat16),
    # ── Hopper series ──
    "h1_ms":              ("mymatmul.gpu.hopper.matmul_h1_ms.matmul_h1_ms",                                 None, torch.bfloat16),
    "h2_s1":              ("mymatmul.gpu.hopper.matmul_h2_s1.matmul_h2_s1",                                 None, torch.bfloat16),
    "h2_s2":              ("mymatmul.gpu.hopper.matmul_h2_s2.matmul_h2_s2",                                 None, torch.bfloat16),
    "h2_s3":              ("mymatmul.gpu.hopper.matmul_h2_s3.matmul_h2_s3",                                 None, torch.bfloat16),
    "h2_s4":              ("mymatmul.gpu.hopper.matmul_h2_s4.matmul_h2_s4",                                 None, torch.bfloat16),
    "h2_s5":              ("mymatmul.gpu.hopper.matmul_h2_s5.matmul_h2_s5",                                 None, torch.bfloat16),
    "h2_s6":              ("mymatmul.gpu.hopper.matmul_h2_s6.matmul_h2_s6",                                 None, torch.bfloat16),
    "h2_s7":              ("mymatmul.gpu.hopper.matmul_h2_s7.matmul_h2_s7",                                 None, torch.bfloat16),
    "h2_s7_runptr":       ("mymatmul.gpu.hopper.matmul_h2_s7_runptr.matmul_h2_s7_runptr",                   None, torch.bfloat16),
    "h2_s8":              ("mymatmul.gpu.hopper.matmul_h2_s8.matmul_h2_s8",                                 None, torch.bfloat16),
    "h2_s8_smem_wb":      ("mymatmul.gpu.hopper.matmul_h2_s8_smem_wb.matmul_h2_s8_smem_wb",                 None, torch.bfloat16),
    "h2_s8_smem_wb_swz":  ("mymatmul.gpu.hopper.matmul_h2_s8_smem_wb_swz.matmul_h2_s8_smem_wb_swz",         None, torch.bfloat16),
    "h3":                 ("mymatmul.gpu.hopper.matmul_h3.matmul_h3",                                       None, torch.bfloat16),
    "h4":                 ("mymatmul.gpu.hopper.matmul_h4.matmul_h4",                                       None, torch.bfloat16),
    "triton_ptx":         ("mymatmul.gpu.hopper.matmul_triton_ptx.matmul_triton_ptx",                       None, torch.bfloat16),
    # ── Blackwell series ──
    "b1_tc5":  ("mymatmul.gpu.blackwell.matmul_b1_tc5.matmul_b1_tc5",   None, torch.bfloat16),
    "b2_ms":   ("mymatmul.gpu.blackwell.matmul_b2_ms.matmul_b2_ms",     None, torch.bfloat16),
    "b3_tc05": ("mymatmul.gpu.blackwell.matmul_b3_tc05.matmul_b3_tc05", None, torch.bfloat16),
}

SIZES = [128, 256, 512, 1024, 2048, 4096, 8192]

# ── Auto-discovery for matmul_cuda_*.py kernels ──────────────────────────────

_GPU_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "mymatmul", "gpu")
_DISCOVERED: dict | None = None


def _auto_discover_impls() -> dict:
    import glob
    result = {}
    for subdir, pkg in [("cuda_core", "mymatmul.gpu.cuda_core"),
                        ("tensor_core", "mymatmul.gpu.tensor_core")]:
        for path in sorted(glob.glob(os.path.join(_GPU_DIR, subdir, "matmul_cuda_*.py"))):
            base = os.path.basename(path)[len("matmul_cuda_"):-3]
            dotpath = f"{pkg}.matmul_cuda_{base}.matmul_{base}"
            try:
                mod = importlib.import_module(f"{pkg}.matmul_cuda_{base}")
                if not hasattr(mod, f"matmul_{base}"):
                    continue
                dtype    = getattr(mod, "DTYPE",    torch.float32)
                max_size = getattr(mod, "MAX_SIZE", None)
            except Exception:
                continue
            result[base] = (dotpath, max_size, dtype)
    return result


def _all_impls() -> dict:
    global _DISCOVERED
    if _DISCOVERED is None:
        _DISCOVERED = _auto_discover_impls()
    merged = dict(_DISCOVERED)
    for k, v in IMPLEMENTATIONS.items():
        merged[k] = (*v, torch.float32) if len(v) == 2 else v
    return merged


def _find_prefix_entry(name: str):
    return _all_impls().get(name.split("_")[0])


def get_impl_dtype(name: str) -> torch.dtype:
    entry = _all_impls().get(name) or _find_prefix_entry(name)
    return entry[2] if entry and len(entry) > 2 else torch.float32


def load_fn(name_or_dotpath: str):
    """Load a matmul function by impl name or dotpath, with prefix-match fallback
    for specific-kernel-config names like 'tc4_bm128_bn64_bk64_nw4'."""
    if '.' in name_or_dotpath:
        module_path, fn_name = name_or_dotpath.rsplit(".", 1)
        return getattr(importlib.import_module(module_path), fn_name)

    entry = _all_impls().get(name_or_dotpath)
    if entry is not None:
        module_path, fn_name = entry[0].rsplit(".", 1)
        return getattr(importlib.import_module(module_path), fn_name)

    prefix_entry = _find_prefix_entry(name_or_dotpath)
    if prefix_entry is None:
        raise KeyError(f"Unknown impl: '{name_or_dotpath}'")

    module_path = prefix_entry[0].rsplit(".", 1)[0]
    mod = importlib.import_module(module_path)
    kname = f"matmul_cuda_{name_or_dotpath}"
    cfg = next((c for c in mod._CONFIGS if mod._kname(*c) == kname), None)
    if cfg is None:
        raise KeyError(f"Kernel '{kname}' not found in {module_path}._CONFIGS")

    block = mod._block(cfg[3]) if len(inspect.signature(mod._block).parameters) > 0 else mod._block()
    smem_nparams = len(inspect.signature(mod._smem).parameters)
    smem = mod._smem(*(cfg[:3] + cfg[4:])[:smem_nparams])
    bm, bn = cfg[0], cfg[1]

    def _direct(A, B):
        from mymatmul.gpu._pycuda_loader import launch_matmul, get_module
        get_module(mod._EXT)
        return launch_matmul(mod._EXT, kname, A, B, block,
                             mod._grid(A.shape[0], B.shape[1], bm, bn),
                             smem_bytes=smem)
    _direct.__name__ = kname
    return _direct


def validate_fn(fn, A_gpu, B_gpu, rtol=None, atol=None):
    if A_gpu.dtype == torch.bfloat16:
        # Reference in FP32: cuBLAS BF16 itself switches to BF16 accumulators
        # at large K, making it unreliable as a reference.
        expected = torch.mm(A_gpu.float(), B_gpu.float())
        if rtol is None: rtol = 1e-2
        if atol is None: atol = max(1.0, A_gpu.shape[1] ** 0.5 / 32)
    else:
        expected = torch.matmul(A_gpu.float(), B_gpu.float())
        if rtol is None: rtol = 1e-2
        if atol is None: atol = 1e-1
    result = fn(A_gpu, B_gpu).float()

    if not torch.allclose(result, expected, rtol=rtol, atol=atol):
        diff = (result - expected).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        raise AssertionError(f"max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}")
    return True


# ── Bench loop ────────────────────────────────────────────────────────────────

WARMUP_MS = 200
REP_MS    = 2000


def tflops(M, N, K, ms):
    return 2 * M * N * K / (ms / 1e3) / 1e12


def _get_tuned_config(name, M, N, K) -> str | None:
    """Return the autotuned config string for (impl, shape), if the impl exposes one.

    Convention: autotuner modules store `_best: dict[(M, N, K), tuple]`. We look
    that up after the kernel has run (so the autotune entry is populated).
    """
    entry = _all_impls().get(name)
    if entry is None: return None
    module_path = entry[0].rsplit(".", 1)[0]
    try:
        mod = importlib.import_module(module_path)
    except Exception:
        return None
    best = getattr(mod, "_best", None)
    if not isinstance(best, dict): return None
    cfg = best.get((M, N, K))
    if cfg is None: return None
    if isinstance(cfg, tuple):
        fields = getattr(mod, "_BEST_FIELDS", None)
        if fields and len(fields) == len(cfg):
            return " ".join(f"{k}={v}" for k, v in zip(fields, cfg))
        return "(" + ",".join(str(x) for x in cfg) + ")"
    return str(cfg)


def run(impl_names, shapes):
    """Returns results[impl][shape] = (tflops_med, ms_med, ms_min, config_or_None)."""
    all_i = _all_impls()
    results: dict = {name: {} for name in impl_names}

    for name in impl_names:
        entry    = all_i.get(name, (None, None, torch.float32))
        max_size = entry[1]
        dtype    = entry[2] if len(entry) > 2 else torch.float32
        try:
            fn = load_fn(name)
        except Exception as e:
            print(f"\n[{name}] load FAILED: {e}")
            for shape in shapes: results[name][shape] = None
            continue

        print(f"\n[{name}]")
        for shape in shapes:
            M, K, N = shape
            tag = f"{M}x{K}x{N}"
            if max_size is not None and max(M, K, N) > max_size:
                print(f"  {tag}: skipped (max_size={max_size})")
                results[name][shape] = None
                continue
            A = torch.randn(M, K, dtype=dtype, device='cuda')
            B = torch.randn(K, N, dtype=dtype, device='cuda')
            try:
                validate_fn(fn, A, B)
            except AssertionError as e:
                print(f"  {tag}: ✗ validation FAILED: {e}")
                results[name][shape] = None
                continue
            try:
                ms_med, ms_min, _ = triton.testing.do_bench(
                    lambda: fn(A, B), warmup=WARMUP_MS, rep=REP_MS,
                    quantiles=(0.5, 0.0, 1.0))
            except Exception as e:
                print(f"  {tag}: ✗ bench FAILED: {e}")
                results[name][shape] = None
                continue
            tf = tflops(M, N, K, ms_med)
            tf_best = tflops(M, N, K, ms_min)
            cfg = _get_tuned_config(name, M, N, K)
            cfg_str = f"  cfg={cfg}" if cfg else ""
            print(f"  {tag}: {tf:7.1f} TFLOPS  (median {ms_med:.3f} ms, best {ms_min:.3f} → {tf_best:7.1f} TFLOPS){cfg_str}")
            results[name][shape] = (tf, ms_med, ms_min, cfg)

    return results


def print_summary(impl_names, shapes, results):
    print("\n" + "=" * 78)
    print(f"Summary (TFLOPS, median) on {torch.cuda.get_device_name(0)}")
    print("=" * 78)

    shape_labels = [f"{M}x{K}x{N}" if not (M == K == N) else str(M)
                    for (M, K, N) in shapes]
    name_w = max(len(n) for n in impl_names) + 2
    col_w  = max(8, max(len(s) for s in shape_labels) + 2)

    header = " " * name_w + "".join(f"{s:>{col_w}}" for s in shape_labels)
    print(header)
    print("-" * len(header))
    for name in impl_names:
        row_cells = []
        for shape in shapes:
            r = results[name].get(shape)
            row_cells.append(f"{r[0]:>{col_w}.1f}" if r else f"{'—':>{col_w}}")
        print(f"{name:<{name_w}}" + "".join(row_cells))

    # Selected autotuned configs (one line per (impl, shape) that had one).
    cfg_lines = [(name, shape_labels[i], r[3])
                 for name in impl_names
                 for i, shape in enumerate(shapes)
                 if (r := results[name].get(shape)) and r[3]]
    if cfg_lines:
        print("\nSelected configs (autotuned):")
        nw = max(len(n) for n, _, _ in cfg_lines)
        sw = max(len(s) for _, s, _ in cfg_lines)
        for name, slabel, cfg in cfg_lines:
            print(f"  {name:<{nw}}  {slabel:<{sw}}  {cfg}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_shape(s):
    parts = s.split("x")
    if len(parts) == 1:
        n = int(parts[0])
        return (n, n, n)
    if len(parts) == 3:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    raise argparse.ArgumentTypeError(f"shape must be N or MxKxN, got {s!r}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--impls", nargs="+", required=True,
                   help="Implementations to benchmark (see IMPLEMENTATIONS in this file).")
    p.add_argument("--sizes", nargs="+", type=int, default=None,
                   help="Square sizes (shorthand for --shapes NxNxN)")
    p.add_argument("--shapes", nargs="+", type=_parse_shape, default=None,
                   help="Shapes as MxKxN (e.g. 64x16384x65536) or plain N for square")
    args = p.parse_args()

    if args.shapes is not None:
        shapes = args.shapes
    elif args.sizes is not None:
        shapes = [(s, s, s) for s in args.sizes]
    else:
        shapes = [(s, s, s) for s in SIZES]

    print(f"[bench] device: {torch.cuda.get_device_name(0)}  "
          f"(cap {torch.cuda.get_device_capability(0)})")
    print(f"[bench] torch {torch.__version__}  cuda {torch.version.cuda}")

    results = run(args.impls, shapes)
    print_summary(args.impls, shapes, results)


if __name__ == "__main__":
    main()
