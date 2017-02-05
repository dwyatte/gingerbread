# Build
docker build -f Dockerfile.cpu -t tensorflow-cpu .

# Run
docker run -it --rm --volume=/Users:/Users --name=tensorflow-cpu tensorflow-cpu /bin/bash