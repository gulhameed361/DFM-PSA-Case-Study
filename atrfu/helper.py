import os
import platform

from pyomo.common.dependencies import numpy as np


def system_report():
    """Return a multi-line string describing the machine the run executed on
    (for reproducibility / benchmarking provenance). Uses only the stdlib plus
    psutil if present (for RAM and physical-core count); degrades gracefully.
    """
    lines = []
    lines.append(f"  OS              : {platform.platform()}")
    lines.append(f"  Processor       : {platform.processor() or platform.machine()}")
    lines.append(f"  Architecture    : {platform.machine()}")
    lines.append(f"  Python          : {platform.python_version()}")

    logical = os.cpu_count()
    physical = None
    ram_gb = None
    try:
        import psutil
        physical = psutil.cpu_count(logical=False)
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        pass
    cores = f"{logical} logical" + (f", {physical} physical" if physical else "")
    lines.append(f"  CPU cores       : {cores}")
    if ram_gb is not None:
        lines.append(f"  RAM             : {ram_gb:.1f} GB")
    # Thread/parallelism hints actually in effect for the run
    threads = os.environ.get('OMP_NUM_THREADS')
    if threads:
        lines.append(f"  OMP_NUM_THREADS : {threads}")
    # Key library versions (affect numerics / reproducibility)
    try:
        import pyomo
        lines.append(f"  numpy / pyomo   : {np.__version__} / {pyomo.version.version}")
    except Exception:
        lines.append(f"  numpy           : {np.__version__}")
    return "\n".join(lines)


def cloneXYZ(x, y, z):
    """
    This function is to create a hard copy of vector x, y, z.
    """
    x0 = np.array(x)
    y0 = np.array(y)
    z0 = np.array(z)
    return x0, y0, z0

def packXYZ(x, y, z):
    """
    This function concatenate x, y, x to one vector and return it.
    """
    t = np.concatenate([x, y, z])
    return t



def minIgnoreNone(a,b):
    if a is None:
        return b
    if b is None:
        return a
    if a<b:
        return a
    return b

def maxIgnoreNone(a,b):
    if a is None:
        return b
    if b is None:
        return a
    if a<b:
        return b
    return a
