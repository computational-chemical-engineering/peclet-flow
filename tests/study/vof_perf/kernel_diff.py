#!/usr/bin/env python3
"""Per-step kernel launches and times from two profiles of the same run at different lengths
(the VoF step-performance harness, WO-0 -- doc/vof_step_performance_design.md sec. 8 G-PERF and
the transfer gate).

A profile covers the whole process (build, warm-up, timed steps). Two runs that differ ONLY in the
number of timed steps (e.g. prof.py 10 --warm 3 and prof.py 30 --warm 3) differ by exactly those
steps, so (B - A) / steps is the per-step cost with setup and warm-up cancelled.

    # GPU: nsys, then its csv summaries
    nsys profile --trace=cuda -o n10 python prof.py 10 --warm 3 --pcg --flux device   (and n30)
    nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_size_sum --format csv --output n10 n10.nsys-rep
    python kernel_diff.py nsys n10 n30 --steps 20 [--top 25]
    # host: the Kokkos simple kernel timer (kokkos-tools profiling/simple-kernel-timer)
    KOKKOS_TOOLS_LIBS=.../kp_kernel_timer.so python prof.py 10 ... > k10.log 2>&1   (and k30)
    python kernel_diff.py kp k10.log k30.log --steps 20 [--top 25]
    (the 2024-09 timer prints its table to stdout at finalize and writes no .dat; an older one
    writes <host>-<pid>.dat, which kp_reader turns into the same text)

nsys mode names a kernel by the function that launched it (Kokkos labels are not visible under
--trace=cuda); kp mode by its Kokkos label. nsys mode also prints the D->H / H->D memcpy count per
step and a size histogram, from the cuda_gpu_mem_size_sum reports and the .sqlite exports.
"""
import csv
import os
import re
import sqlite3
import sys

A = sys.argv
MODE, PA, PB = A[1], A[2], A[3]
STEPS = int(A[A.index("--steps") + 1]) if "--steps" in A else 20
TOP = int(A[A.index("--top") + 1]) if "--top" in A else 25


def short_nsys(name):
    """'void Kokkos::Impl::cuda_parallel_launch_...<Kokkos::Impl::ParallelFor<peclet::flow::X::f(...)
    ::[lambda ... (instance 2)] ...' -> 'X::f#2' (the launching function, lambda instance)."""
    m = re.search(r"Parallel(?:For|Reduce|ScanWithTotal|Scan)<(.*)", name)
    body = m.group(1) if m else name
    body = re.sub(r"^(Kokkos::Impl::Combined\w+<)+", "", body)
    body = re.sub(r"^void ", "", body)
    m = re.match(r"([\w:]+(?:<[^()]*?>)?(?:::[\w]+)*)", body)
    fn = m.group(1) if m else body[:60]
    fn = re.sub(r"<.*>", "", fn).replace("peclet::flow::", "")
    inst = re.search(r"\(instance (\d+)\)", body)
    kind = "R:" if "ParallelReduce" in name else ("S:" if "ParallelScan" in name else "")
    return kind + fn + (f"#{inst.group(1)}" if inst and inst.group(1) != "1" else "")


def load_nsys(prefix):
    out = {}
    for r in csv.DictReader(open(prefix + "_cuda_gpu_kern_sum.csv")):
        k = short_nsys(r["Name"])
        n, t = out.get(k, (0, 0.0))
        out[k] = (n + int(r["Instances"]), t + float(r["Total Time (ns)"]) * 1e-9)
    return out


def load_kp(path):
    """kernel-timer text: '- <label>' then ' (<type>)  <total s> <calls> <avg> ...'."""
    out, label = {}, None
    for line in open(path):
        s = line.strip()
        if s.startswith("- "):
            label = s[2:].strip()
        elif label and s.startswith("("):
            f = s.split(")", 1)[1].split()
            out[label] = (int(f[1]), float(f[0]))
            label = None
    return out


def memcpy_hist(prefix):
    db = prefix + ".sqlite"
    if not os.path.exists(db):
        return {}
    con = sqlite3.connect(db)
    names = dict(con.execute("SELECT id, label FROM ENUM_CUDA_MEMCPY_OPER"))
    h = {}
    for kind, size in con.execute("SELECT copyKind, bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY"):
        b = "<=8B" if size <= 8 else "<=1KiB" if size <= 1024 else "<=64KiB" if size <= 65536 \
            else "<=1MiB" if size <= 1 << 20 else ">1MiB"
        key = (names.get(kind, str(kind)), b)
        h[key] = h.get(key, 0) + 1
    return h


a, b = (load_nsys(PA), load_nsys(PB)) if MODE == "nsys" else (load_kp(PA), load_kp(PB))
rows = []
for k in set(a) | set(b):
    na, ta = a.get(k, (0, 0.0))
    nb, tb = b.get(k, (0, 0.0))
    rows.append((k, (nb - na) / STEPS, 1000.0 * (tb - ta) / STEPS))
rows.sort(key=lambda r: -r[2])
tot_n = sum(r[1] for r in rows)
tot_t = sum(r[2] for r in rows)
print(f"per step over {STEPS} steps ({MODE}): {tot_n:.1f} launches, {tot_t:.3f} ms kernel time")
print(f"{'launches/step':>14} {'ms/step':>9}  kernel")
for k, n, t in rows[:TOP]:
    print(f"{n:14.2f} {t:9.3f}  {k}")
if MODE == "nsys":
    ha, hb = memcpy_hist(PA), memcpy_hist(PB)
    if ha or hb:
        print("memcpy per step (kind, size bin): count")
        for key in sorted(set(ha) | set(hb)):
            d = (hb.get(key, 0) - ha.get(key, 0)) / STEPS
            if d:
                print(f"   {key[0]:28s} {key[1]:>8s}  {d:8.2f}")
