FROM ubuntu:20.04
FROM python:3.8
FROM continuumio/miniconda
RUN apt-get --allow-releaseinfo-change update
WORKDIR /home/
COPY . .
RUN . /root/.bashrc && \
	conda init bash && \
	conda env create -f environment.yml && \
	conda activate hparam_project && \
	pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116 && \
	pip install --editable . 

