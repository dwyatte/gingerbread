# gingerbread
Tensorflow experiments

### Requirements
If you are running Mac OS < 10.11, tensorflow must be installed manually from wheels hosted by Google:

`pip install https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.0.0-py2-none-any.whl`

When importing tensorflow, you may get an error along the lines of `"Versioning for this project requires either an
sdist tarball, or access to an upstream git repository. Are you sure that git is installed?"` See here for details:
https://github.com/tensorflow/tensorflow/issues/6411

### Docker

##### Build
Assuming you have docker installed, you can build the image from this repo:
```
docker build -f Dockerfile -t tensorflow .
```

##### Run
To run the container and get a shell, publishing jupyter/tensorboard on their default ports to the host and mounting
this repo:
```
docker run -it --rm -p 8888:8888 -p 6006:6006 -v=$PWD:$PWD -w=$PWD --name=tensorflow tensorflow /bin/bash
```

Note: If you want to run the jupyter notebook server, it needs to be run on 0.0.0.0 to expose it to the host
(`jupyter notebook --ip=0.0.0.0`)

On Mac OS 10.x, it should be possible to use tensorflow from the command line and plot through matplotlib's Tk backend 
by configuring the relevant X11 security preferences and forwarding your display 
(see: https://github.com/docker/docker/issues/8710).
```
#!/bin/bash

# add ip to X11 acl
ip=$(python -c 'import socket; print(socket.gethostbyname(socket.gethostname()))')
xhost + $ip
trap 'xhost - $ip' EXIT

# forward to container
docker run -it --rm -p 8888:8888 -p 6006:6006 -v=$PWD:$PWD -w=$PWD -e DISPLAY=$ip:0 --name=tensorflow tensorflow /bin/bash

```
