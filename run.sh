#!/bin/bash

# add ip to XQuartz acl
ip=$(python -c 'import socket; print(socket.gethostbyname(socket.gethostname()))')
xhost + $ip
trap 'xhost - $ip' EXIT

# forward to container
docker run -it --rm -p 8888:8888 -p 6006:6006 -v=$PWD:$PWD -w=$PWD -e DISPLAY=$ip:0 --name=tensorflow-cpu tensorflow-cpu /bin/bash
