# generate RMI code and estimate cost

import os
import shutil
import subprocess
import sys
from pathlib import Path

DATASET = "/mnt/home/zwshi/learned-index/CAM/src/rmi/dataset/books_10M_uint64_unique_fixed"
QUERY = "/mnt/data/Dataset/public/SOSD/books_10M_uint64_unique.query.bin"
N = 10_000_000
IPP = 512
STRATEGY = "all_in_once"
MEMORY_MIBS = [8, 16, 32, 64]
DATASET_TAG = "books_10M"
RMI_REPO = "src/rmi"
WORKDIR = Path("src/rmi/rmi_eval")
GEN = WORKDIR / "generated"
RESULTS = WORKDIR / "results"
LOGDIR = Path("build/log/rmi")
NPZ_DIR = LOGDIR / "npz"
GEN.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)
LOGDIR.mkdir(parents=True, exist_ok=True)
NPZ_DIR.mkdir(parents=True, exist_ok=True)

PYTHON_BIN = Path(os.environ.get("PYTHON_BIN", os.path.expanduser("~/miniconda3/bin/python")))
if not PYTHON_BIN.exists():
    PYTHON_BIN = Path(sys.executable)

configs = [
    ("linear_spline,linear", 64),
    ("linear_spline,linear", 128),
    ("linear_spline,linear", 256),
    ("linear_spline,linear", 512),
    ("linear_spline,linear", 1024),
    ("linear_spline,linear", 2048),
    ("linear_spline,linear", 4096),
    ("linear_spline,linear", 8192),
    ("linear_spline,linear", 16384),
    ("linear_spline,linear", 32768),
    ("linear_spline,linear", 65536),
    ("linear_spline,linear", 131072),
    ("linear_spline,linear", 262144),
    ("linear_spline,linear", 524288),
    ("linear_spline,linear", 1048576),
    ("linear_spline,linear", 2097152),
]

def make_wrapper(gen_dir: Path, ns: str):
    wrapper = gen_dir / "rmi_wrapper.h"
    wrapper.write_text(f'''#pragma once
#include "{ns}.h"
namespace rmi_ns = {ns};
''')


for memory_mib in MEMORY_MIBS:
    summary_log = LOGDIR / f"{DATASET_TAG}_M{memory_mib}_rmi_optimalBF_summary.log"
    if summary_log.exists():
        summary_log.unlink()


for models, branch in configs:
    ns = f"books_rmi_{models.replace(',', '_').replace('-', '_')}_{branch}"
    print("building", ns)

    # 1) generate RMI code
    subprocess.run(
        [
            "cargo", "run", "--release", "--",
            DATASET, ns, models, str(branch)
        ],
        cwd=RMI_REPO,
        check=True
    )
    
    make_wrapper(GEN, ns)
    
    # 2) move generated files
    for ext in [".h", ".cpp", "_data.h"]:
        src = Path(RMI_REPO) / f"{ns}{ext}"
        dst = GEN / f"{ns}{ext}"
        shutil.move(src, dst)

    # 3) build analysis binary
    analyze_cpp = WORKDIR / "rmi_collector.cpp"
    binary = WORKDIR / f"rmi_collector"

    subprocess.run(
        [
            "g++", "-O3", "-std=c++17",
            str(analyze_cpp),
            str(GEN / f"{ns}.cpp"),
            "-I", str(GEN),
            "-D", f"RMI_HEADER=\\\"{ns}.h\\\"",
            "-o", str(binary),
        ],
        check=True
    )

    # 4) run analysis exp
    out_file = RESULTS / f"{ns}.csv"

    subprocess.run(
        [str(binary), DATASET, RMI_REPO+"/rmi_data", QUERY, out_file],
        check=True
    )

    # 5) estimate BF under each memory budget
    for memory_mib in MEMORY_MIBS:
        summary_log = LOGDIR / f"{DATASET_TAG}_M{memory_mib}_rmi_optimalBF_summary.log"
        npz_out = NPZ_DIR / f"{ns}_M{memory_mib}_optimalBF.npz"

        subprocess.run(
            [
                str(PYTHON_BIN), "utils/optimalBF.py",
                str(out_file), str(N),
                "--ipp", str(IPP),
                "--strategy", STRATEGY,
                "--memory-mib", str(memory_mib),
                "--header-mode", "branch_factor",
                "--log-path", str(summary_log),
                "--out", str(npz_out),
                "--eps-transform", "cap",
                "--eps-transform-q", "0.9",
                "--mode", "global"
            ],
            check=True,
        )
