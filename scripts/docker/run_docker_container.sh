#!/bin/bash
# runs the docker container, creating a bind mound at
# the file path specified 

read -p "directory on host machine for bind mount: " hostdir
if [ -d "$hostdir" ]; then
    mkdir $hostdir
fi

docker run \
    -d \
    --name hparam \
    --mount type=bind,\
            source=$hostdir,\
            destination=/home/hparam_results
    hparam_project \
    tail -f /dev/null