"""Keep numpy's BLAS single-threaded unless the user says otherwise; import before numpy.

The solver's matrices are small (a few hundred rows): extra BLAS threads do not make it faster, but
OpenBLAS threads busy-wait, so on a 12-core laptop every core shows 100% while the agent runs. With two
numpy callers at once (the background solve and the live fight estimates) the thread pools also
compete. Set OPENBLAS_NUM_THREADS / OMP_NUM_THREADS / MKL_NUM_THREADS yourself to override.
"""
import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
