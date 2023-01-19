#!/bin/bash

set -e

# run:
# chmod +x docker_run.sh
# ./docker_run.sh

export NAME="ls6-stud-registry.informatik.uni-wuerzburg.de/studwangsadirdja-spirex:0.0.1"

echo "Building the container..."
fastbuildah bud -t ${NAME} --format docker -f Dockerfile .
echo "Login to container registry. Username: stud, Password: studregistry"
fastbuildah login ls6-stud-registry.informatik.uni-wuerzburg.de
echo "Pushing container to registry..."
fastbuildah push ${NAME}