#!/bin/bash

if [ -z "$1" -a -z "$2" ]; then
    echo "Usage: ./setup_container.sh CONTAINER_NAME PORT"
    exit
fi

# Run container
docker run -it --rm --name=$1 --ipc=host --gpus=all -p $2:22 \
    -v `pwd -P`:/home/user1/$(basename `pwd`) \
    yoshinakam/pytorch:1.7.1-cuda10.2-py3.7 /bin/bash --login
