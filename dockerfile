FROM condaforge/mambaforge:4.9.2-5
WORKDIR /home/
COPY . .
RUN apt-get --allow-releaseinfo-change update
RUN apt-get install vim tmux pip -y
RUN \
	pip install --upgrade pip && \
        .  /root/.bashrc && \
    conda update conda && \
	conda init bash && \
	conda env create -f environment.yml && \
	conda activate hparam_project && \
	conda clean -a && \
	conda install pytorch=1.13.0 torchvision=0.14 torchaudio cpuonly -c pytorch && \
	pip install --editable . 


