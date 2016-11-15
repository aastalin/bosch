#!/usr/bin/env sh
set -e

export PYTHONPATH=/home/ubuntu/caffe/lib:/home/ubuntu/caffe/python
./build/tools/caffe train --solver=examples/bosch/solver.prototxt $@
