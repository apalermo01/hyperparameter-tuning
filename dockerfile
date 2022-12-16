# FROM ubuntu:20.04
FROM continuumio/miniconda3
WORKDIR /home/projects/hparam_project/
COPY . .
RUN apt update
RUN apt install vim
RUN conda env create -f environment.yml
RUN echo "source activate $(head -1 environment.yml | cut -d ' ' -f2)" > ~/.bashrc
ENV PATH /opt/conda/envs/$(head -1 environment.yml | cut -d ' ' -f2)/bin:$PATH
RUN pip install --editable .
# https://medium.com/@chadlagore/conda-environments-with-docker-82cdc9d25754