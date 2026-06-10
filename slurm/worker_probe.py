import os
import socket
import subprocess
import resource
from pathlib import Path


def read_first_existing(paths):
    for p in paths:
        path = Path(p)
        if path.exists():
            try:
                return str(path), path.read_text().strip()
            except Exception as e:
                return str(path), f"READ_ERROR: {e}"
    return None, None


def bytes_to_human(x):
    try:
        x = int(x)
    except Exception:
        return str(x)

    if x < 0 or x > 10**18:
        return "unlimited/unknown"

    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    val = float(x)
    for u in units:
        if val < 1024:
            return f"{val:.2f} {u}"
        val /= 1024
    return f"{val:.2f} PiB"


def get_cpu_affinity():
    try:
        affinity = os.sched_getaffinity(0)
        return sorted(affinity), len(affinity)
    except Exception as e:
        return f"ERROR: {e}", None


def get_cgroup_memory_limit():
    # cgroup v2 common path
    candidates = [
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ]

    path, value = read_first_existing(candidates)

    if value is None:
        return None, None, None

    if value == "max":
        return path, value, "unlimited"

    return path, value, bytes_to_human(value)


def get_cgroup_cpu_quota():
    # cgroup v2: cpu.max has format: "<quota> <period>", or "max <period>"
    path = Path("/sys/fs/cgroup/cpu.max")
    if path.exists():
        try:
            value = path.read_text().strip()
            parts = value.split()
            if len(parts) == 2:
                quota, period = parts
                if quota == "max":
                    return str(path), value, "unlimited"
                quota_f = float(quota)
                period_f = float(period)
                cpus = quota_f / period_f
                return str(path), value, f"{cpus:.2f} CPU quota"
            return str(path), value, "unknown format"
        except Exception as e:
            return str(path), f"READ_ERROR: {e}", None

    # cgroup v1
    quota_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota_path.exists() and period_path.exists():
        try:
            quota = int(quota_path.read_text().strip())
            period = int(period_path.read_text().strip())
            if quota < 0:
                return f"{quota_path}, {period_path}", f"{quota}, {period}", "unlimited"
            return f"{quota_path}, {period_path}", f"{quota}, {period}", f"{quota / period:.2f} CPU quota"
        except Exception as e:
            return f"{quota_path}, {period_path}", f"READ_ERROR: {e}", None

    return None, None, None


def run_cmd(cmd):
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception as e:
        return f"ERROR: {e}"


def main():
    hostname = socket.gethostname()

    procid = os.environ.get("SLURM_PROCID", "NA")
    localid = os.environ.get("SLURM_LOCALID", "NA")
    nodeid = os.environ.get("SLURM_NODEID", "NA")

    cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK", "NA")
    ntasks = os.environ.get("SLURM_NTASKS", "NA")
    ntasks_per_node = os.environ.get("SLURM_TASKS_PER_NODE", "NA")
    job_cpus_per_node = os.environ.get("SLURM_JOB_CPUS_PER_NODE", "NA")

    omp = os.environ.get("OMP_NUM_THREADS", "NA")
    mkl = os.environ.get("MKL_NUM_THREADS", "NA")
    openblas = os.environ.get("OPENBLAS_NUM_THREADS", "NA")

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "NA")

    affinity_list, affinity_count = get_cpu_affinity()

    mem_path, mem_raw, mem_human = get_cgroup_memory_limit()
    cpu_path, cpu_raw, cpu_quota = get_cgroup_cpu_quota()

    soft_as, hard_as = resource.getrlimit(resource.RLIMIT_AS)
    soft_rss, hard_rss = resource.getrlimit(resource.RLIMIT_RSS)

    print("=" * 80, flush=True)
    print(f"hostname                 : {hostname}", flush=True)
    print(f"SLURM_PROCID             : {procid}", flush=True)
    print(f"SLURM_LOCALID            : {localid}", flush=True)
    print(f"SLURM_NODEID             : {nodeid}", flush=True)
    print(f"SLURM_NTASKS             : {ntasks}", flush=True)
    print(f"SLURM_TASKS_PER_NODE     : {ntasks_per_node}", flush=True)
    print(f"SLURM_JOB_CPUS_PER_NODE  : {job_cpus_per_node}", flush=True)
    print(f"SLURM_CPUS_PER_TASK      : {cpus_per_task}", flush=True)
    print(f"os.cpu_count()           : {os.cpu_count()}", flush=True)
    print(f"CPU affinity count       : {affinity_count}", flush=True)
    print(f"CPU affinity list        : {affinity_list}", flush=True)
    print(f"OMP_NUM_THREADS          : {omp}", flush=True)
    print(f"MKL_NUM_THREADS          : {mkl}", flush=True)
    print(f"OPENBLAS_NUM_THREADS     : {openblas}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES     : {cuda_visible}", flush=True)
    print(f"cgroup memory path       : {mem_path}", flush=True)
    print(f"cgroup memory raw        : {mem_raw}", flush=True)
    print(f"cgroup memory human      : {mem_human}", flush=True)
    print(f"cgroup cpu path          : {cpu_path}", flush=True)
    print(f"cgroup cpu raw           : {cpu_raw}", flush=True)
    print(f"cgroup cpu quota         : {cpu_quota}", flush=True)
    print(f"RLIMIT_AS soft/hard      : {soft_as} / {hard_as}", flush=True)
    print(f"RLIMIT_RSS soft/hard     : {soft_rss} / {hard_rss}", flush=True)

    # Optional commands, useful on many clusters
    print(f"nproc                    : {run_cmd('nproc')}", flush=True)
    print(f"free -h                  : {run_cmd('free -h | head -n 2')}", flush=True)
    print("=" * 80, flush=True)
if __name__ == "__main__":
    main()