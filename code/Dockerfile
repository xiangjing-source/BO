# Build from the submission directory:
# docker build -t bo-reaction -f code/Dockerfile .
# Keep the official data and generated output inside the image workspace, using
# the same self-contained path contract as the submission launchers.
FROM python:3.11-slim

LABEL maintainer="LightChem Team"
LABEL description="Leakage-safe Bayesian optimization for reaction conditions"

WORKDIR /workspace
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    scipy==1.11.4 \
    scikit-learn==1.3.2 \
    pandas==2.1.4 \
    rdkit==2026.3.4

COPY code /workspace/code
COPY data /workspace/data
WORKDIR /workspace
ENTRYPOINT ["python", "code/main/run_selected.py"]
