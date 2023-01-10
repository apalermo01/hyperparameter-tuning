FROM python:3.10-slim-buster
# FROM continuumio/miniconda3
# kubernetes
# FROM rayproject/ray:2.2.0
WORKDIR /home/projects/hparam_project/
COPY . .
RUN apt update
# RUN apt -y install vim pip
RUN pip install -r requirements.txt
RUN pip install --editable .
#RUN apt update
#RUN apt -y install vim
#RUN conda env create -f environment.yml
#RUN echo "source activate $(head -1 environment.yml | cut -d ' ' -f2)" > ~/.bashrc
#ENV PATH /opt/conda/envs/$(head -1 environment.yml | cut -d ' ' -f2)/bin:$PATH
#RUN pip install --editable .
# https://medium.com/@chadlagore/conda-environments-with-docker-82cdc9d25754

