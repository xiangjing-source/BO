#!/usr/bin/env python3
"""Small cross-platform runtime preflight without reading any outcome table."""

from __future__ import annotations

import json
import platform
import sys

import numpy
import pandas
import rdkit
import scipy
import sklearn


def main() -> None:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"Expected Python 3.11, found {platform.python_version()}")
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
        "rdkit": rdkit.__version__,
    }
    print(json.dumps(versions, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
