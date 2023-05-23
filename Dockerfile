# syntax=docker/dockerfile:1

FROM python:3.6.9

WORKDIR /

COPY . .
 
RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PWD}/mujoco210/bin/
RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PWD}/mujoco210/bin/" >> ~/.bashrc
RUN export MUJOCO_PY_MUJOCO_PATH=${PWD}/mujoco210 
RUN echo "export MUJOCO_PY_MUJOCO_PATH=${PWD}/mujoco210" >> ~/.bashrc
RUN pip3 install -r requirements.txt
RUN ["chmod", "+x", "/run.sh"]
EXPOSE      6006
ENTRYPOINT ["/run.sh"]
CMD ["mujoco", "Hopper-v2", "pgail"]
 