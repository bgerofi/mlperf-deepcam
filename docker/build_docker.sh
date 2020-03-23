#!/bin/bash

#login
#docker login -u tkurth gitlab-master.nvidia.com:5005

#we need to step out to expand the build context
cd ..

#nvidia-docker build -t tkurth/pytorch-bias_gan:latest .
nvidia-docker build -t gitlab-master.nvidia.com:5005/tkurth/mlperf-deepcam:debug -f docker/Dockerfile .
docker push gitlab-master.nvidia.com:5005/tkurth/mlperf-deepcam:debug

#tag for NERSC registry
docker tag gitlab-master.nvidia.com:5005/tkurth/mlperf-deepcam:debug registry.services.nersc.gov/tkurth/mlperf-deepcam:debug
docker push registry.services.nersc.gov/tkurth/mlperf-deepcam:debug
