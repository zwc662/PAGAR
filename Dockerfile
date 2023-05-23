# syntax=docker/dockerfile:1

FROM python:3.6.9

WORKDIR /

COPY . .
RUN apk update && apk add python3-dev \
                        gcc \
                        libc-dev
RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PWD}/mujoco210/bin/
RUN export MUJOCO_PY_MUJOCO_PATH=${PWD}/mujoco210 
RUN pip3 install -r requirements.txt
EXPOSE      6006
ENTRYPOINT ["bash run.sh"]
 