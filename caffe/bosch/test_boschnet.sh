#!/usr/bin/env sh
set -e

export PYTHONPATH=/home/ubuntu/caffe/lib:/home/ubuntu/caffe/python
./build/tools/caffe test -model examples/bosch/inference.prototxt -weights examples/bosch/inference.caffemodel -iterations 79
