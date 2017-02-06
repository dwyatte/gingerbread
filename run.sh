#!/bin/bash

# start the socket, trap pid for exit
socat TCP-LISTEN:6000,reuseaddr,fork UNIX-CLIENT:\"$DISPLAY\" &
pid=$!
trap 'kill $pid' EXIT

# get local ip and forward to container
ip=$(python -c 'import socket; print(socket.gethostbyname(socket.gethostname()))')
docker run -it --rm -p 8888:8888 -p 6006:6006 -v=$PWD:$PWD -w=$PWD -e DISPLAY=$ip:0 --name=tensorflow-cpu tensorflow-cpu /bin/bash
