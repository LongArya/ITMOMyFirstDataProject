FROM python:3.10

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y libopencv-dev python3-opencv

WORKDIR /MyFirstDataProject
COPY . /MyFirstDataProject

RUN pip install -r /MyFirstDataProject/requirements.txt

