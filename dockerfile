# FROM ubuntu:20.04
FROM continuumio/miniconda3
WORKDIR /home/projects/hparam_project/
COPY . .

# RUN conda env create -f environment.yml
# RUN echo "source activate $(head -1 environment.yml | cut -d ' ' -f2)" > ~/.bashrc
# ENV PATH /opt/conda/envs/$(head -1 environment.yml | cut -d ' ' -f2)/bin:$PATH
# RUN pip install --editable .
# RUN conda install pytorch torchvision torchaudio cpuonly -c pytorch
# https://medium.com/@chadlagore/conda-environments-with-docker-82cdc9d25754
