FROM python:3.10-slim-buster
WORKDIR ./
COPY requirements.txt .
RUN apt update
RUN pip install -r requirements.txt
COPY . .
RUN pip install --editable .
