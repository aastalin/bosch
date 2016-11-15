# Kaggle Bosch Competition
> There are two sample code for kaggle competitions:
> 
> 
> Bosch Production Line Performance: 
>  
> <https://www.kaggle.com/c/bosch-production-line-performance>
> 
---
##0. Separate data to positive and negative
Folder: script

$ python dataHandle.py


1. Simple script for a tiny one-layer model

Folder: script


Train:
 
$ python simpleNN.py


Test:
 
$ python predictNN.py


2. caffe sample with self-defined data layer

Folder: caffe

$ cp -r bosch $CAFFE_ROOT/examples/

$ cp -r lib $CAFFE_ROOT/


Train:

$ ./examples/bosch/train_boschnet.sh


Test:

$ ./examples/bosch/test_boschnet.sh




Feel free to download and make it your use-case : )
