# Kaggle Bosch Competition
There are two sample code for kaggle competitions:
</br>
</br>
Bosch Production Line Performance: 
</br> 
<https://www.kaggle.com/c/bosch-production-line-performance>

---
##0. Separate data to positive and negative
**Folder:** script
```C
$ python dataHandle.py
```
</br>
##1. Simple script for a tiny one-layer model
**Folder:** script
</br>
**Train:**
```
$ python simpleNN.py
```
**Test:**
```
$ python predictNN.py
```
</br>
##2. caffe sample with self-defined data layer
**Folder:** caffe
</br>
```
$ cp -r bosch $CAFFE_ROOT/examples/
$ cp -r lib $CAFFE_ROOT/
```
**Train:**
```
$ ./examples/bosch/train_boschnet.sh
```
**Test:**
```
$ ./examples/bosch/test_boschnet.sh
```
</br>
</br>
**Feel free to download and make it your use-case : )**
