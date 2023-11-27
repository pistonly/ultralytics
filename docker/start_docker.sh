#!/bin/bash
docker run -it --rm --ipc=host --gpus all -v /home:/home -v /mnt:/mnt -v /dataset:/dataset -v /share:/share ultralytics/ultralytics:latest
