# CatalystFlow Windows x86_64 offline runtime

`catalystflow-windows-x86_64.tar.gz` is the native Windows environment used
by `runtime/run_reproduction_windows.ps1`. It is a `conda-pack` archive: the
launcher verifies its SHA-256, extracts it on first use, runs `conda-unpack`,
performs the project smoke test, then executes the formal 2 datasets × 20
seeds × 40 rounds evaluation without network access or a Conda installation.

Validated package versions are Python 3.11.15, NumPy 1.26.4, SciPy 1.11.4,
pandas 2.1.4, scikit-learn 1.3.2, and RDKit 2026.03.4. The archive pins the
Windows OpenBLAS build (`libopenblas 0.3.25`, pthreads) rather than the
solver-selected MKL 2026.1.0 build, which failed native linear-algebra calls
on the validation host.

Validation completed on Windows 10/11 x86_64: 25 unit tests passed, the
NumPy/SciPy/RDKit smoke test passed, and all 40 formal trajectories matched
the Linux baseline in query index, selected condition, observed yield, and
best-so-far value. See `environment-lock.json` for the archive digest and
locked critical package builds.
