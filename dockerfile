FROM condaforge/mambaforge:4.9.2-5
WORKDIR /home/hparam_project/
COPY . .
RUN apt-get --allow-releaseinfo-change update
# RUN apt-get install vim
RUN \
	pip install -U pip && \
	pip install --upgrade pip && \
        .  /root/.bashrc && \
	conda init bash && \
	conda env create -f environment.yml && \
	conda activate hparam_project && \
	conda clean -a && \
	pip install torch torchvision torchaudio --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu117 && \
	pip install --editable . 

