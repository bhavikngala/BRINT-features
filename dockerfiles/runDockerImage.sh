#!/usr/bin/env bash
wd=`pwd`

#docker run -it -e DISPLAY=192.168.0.9:0 -v "$wd/..":/root/vslam opencv:v1.0 bash
docker run -it -v "$wd/..":/root/vslam opencv:v1.0 bash
