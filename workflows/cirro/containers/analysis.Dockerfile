# syntax=docker/dockerfile:1.7
FROM ghcr.io/dcjones/proseg@sha256:7daa31a81107ed8b7a65f14b61bec0e71c8f05b620ece1ffd7fe0b72bd83cfdb AS proseg

FROM mambaorg/micromamba:2.3.3
USER root
ENV PATH=/opt/conda/bin:${PATH} \
    PYTHONPATH=/opt/segbench/src \
    SEGBENCH_ENV_CONFIG=/opt/segbench/workflows/cirro/configs/environments.container.yaml \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git procps time libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN micromamba install -y -n base -c conda-forge \
      python=3.11 r-base=4.4 r-seurat=5.3.0 r-arrow=21.0.0 \
      r-optparse r-jsonlite r-dplyr r-remotes r-matrix \
    && micromamba clean --all --yes

RUN python -m pip install \
      numpy==1.26.4 pandas==2.3.3 scipy==1.17.1 pyarrow==25.0.1 \
      anndata==0.12.19 scanpy==1.11.5 h5py==3.16.0 scikit-learn==1.9.0 \
      psutil==7.2.2 matplotlib==3.11.1 seaborn==0.13.2 pyyaml==6.0.3 \
      fastparquet==2026.5.0 bin2cell==0.3.4 stardist==0.9.2 csbdeep==0.8.2 \
      tensorflow-cpu==2.16.2

RUN Rscript -e 'options(repos=c(CRAN="https://cloud.r-project.org")); \
  remotes::install_github("dmcable/spacexr@9f5dc33c8060f946c6072a138b70e189636e1435", dependencies=NA, upgrade="never"); \
  remotes::install_github("bdsc-tds/SPLIT@e880e39c03a7036e11c9867701c76fdb26acc14c", dependencies=NA, upgrade="never"); \
  remotes::install_github("kharchenkolab/cellAdmix@8cd0fdfef59f40ef7b3e77e0a03c92722a210767", dependencies=NA, upgrade="never")'

COPY --from=proseg /usr/local/bin/proseg* /usr/local/bin/
COPY . /opt/segbench

# Cache both named StarDist models in the image. Bin2Cell production jobs do
# not fetch models or install packages at runtime.
RUN python -c 'from stardist.models import StarDist2D; StarDist2D.from_pretrained("2D_versatile_he"); StarDist2D.from_pretrained("2D_versatile_fluo")' \
    && mkdir -p /opt/container-manifest \
    && python -m pip freeze > /opt/container-manifest/python-packages.txt \
    && Rscript -e 'x<-installed.packages(); write.table(x[,c("Package","Version")],"/opt/container-manifest/r-packages.tsv",sep="\t",quote=FALSE,row.names=FALSE)'

WORKDIR /work
CMD ["python", "-m", "segbench", "--version"]
