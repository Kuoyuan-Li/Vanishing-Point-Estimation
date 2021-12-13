# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

COPY . /app

RUN apt-get update

RUN apt-get install ffmpeg libsm6 libxext6  -y

CMD python3 vanish_point.py