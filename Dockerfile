FROM mambaorg/micromamba:2.3.3
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
COPY --chown=$MAMBA_USER:$MAMBA_USER deploy/fix_gromacs_completion.py /tmp/fix_gromacs_completion.py
RUN micromamba install -y -n base -f /tmp/environment.yml && /opt/conda/bin/python /tmp/fix_gromacs_completion.py && micromamba clean --all --yes
WORKDIR /app
COPY --chown=$MAMBA_USER:$MAMBA_USER . /app
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN pip install --no-deps . && mkdir -p /app/data /app/mpl-cache
ENV SMILES_DATA_DIR=/app/data MPLCONFIGDIR=/app/mpl-cache OMP_NUM_THREADS=1
EXPOSE 8000
CMD ["uvicorn", "smiles_service.api:app", "--host", "0.0.0.0", "--port", "8000"]
