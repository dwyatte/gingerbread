# Build
Assuming you have docker installed, you can build the image from this repo:
```
docker build -f Dockerfile.cpu -t tensorflow-cpu .
```

# Run
To run the container and get a shell, publishing jupyter/tensorboard on their default ports to the host and mounting this repo:
```
docker run -it --rm -p 8888:8888 -p 6006:6006 -v=$PWD:$PWD -w=$PWD --name=tensorflow-cpu tensorflow-cpu /bin/bash
```

Note: If you want to run the jupyter notebook server, it needs to be run on 0.0.0.0 to expose it to the host (`jupyter notebook --ip=0.0.0.0)

It should be possible to use tensorflow from the command line and plotting through matplotlib's Tk backend by forwarding your display (even on Mac OS 10.x, see: https://github.com/docker/docker/issues/8710)
