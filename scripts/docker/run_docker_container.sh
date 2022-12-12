#!/bin/bash
# runs the docker container, creating a bind mound at
# the file path specified 

if [ -d "$1" ]; then
    mkdir $1
fi

docker run \
    -d \
    --name hparam \
    --mount type=bind,\
            source=$1,\
            destination=/home/hparam_results
    hparam_project \
    tail -f /dev/null