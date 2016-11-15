# --------------------------------------------------------
# Bosch data layer
# Written by Aasta Lin
# --------------------------------------------------------
import caffe
import numpy as np
import pickle
import csv

class BoschDataLayer(caffe.Layer):
    def _openFile(self):
        self.ptr_pos = open('data/bosch/train_numeric_pos.csv',"r")
        self.ptr_neg = open('data/bosch/train_numeric_neg.csv',"r")
        self.reader_pos = csv.reader(self.ptr_pos)
        self.reader_neg = csv.reader(self.ptr_neg)
        header = next(self.reader_pos) #ingore header
        header = next(self.reader_neg) #ingore header

        self.date_pos = open('data/bosch/train_date_pos.csv',"r")
        self.date_neg = open('data/bosch/train_date_neg.csv',"r")
        self.rdate_pos = csv.reader(self.date_pos)
        self.rdate_neg = csv.reader(self.date_neg)
        header = next(self.rdate_pos) #ingore header
        header = next(self.rdate_neg) #ingore header

    def _readLine(self, ptr, reader):
        try:
            line = next(reader)
        except StopIteration:
            ptr.seek(0)
            header = next(reader)
            return self._readLine(ptr,reader)
        # read data and write missing to 999
        raw = line[1:(len(line))]
        for i in range(len(raw)):
            if not raw[i]:
                raw[i] = 0
        # convert to np array
        tmp = np.array(raw, dtype='Float32')
        raw = tmp.astype(np.float)
        # feature
        feature = np.zeros((len(raw)-1))
        for i in range(len(raw)-1): feature[i] = raw[i]
        # response
        response = raw[len(raw)-1]
        return feature, response

    def _readDate(self, ptr, reader):
        try:
            line = next(reader)
        except StopIteration:
            ptr.seek(0)
            header = next(reader)
            return self._readDate(ptr,reader)
        # read data and write missing to 999
        raw = line[1:(len(line))]
        for i in range(len(raw)):
            if not raw[i]:
                raw[i] = 0
        # convert to np array
        tmp = np.array(raw, dtype='Float32')
        raw = tmp.astype(np.float)
        # feature
        date = np.zeros((len(raw)))
        for i in range(len(raw)): date[i] = raw[i]
        return date

    def setup(self, bottom, top):
        self._openFile()
        top[0].reshape(128, 968, 1, 1)
        top[1].reshape(128, 1156, 1, 1)
        top[2].reshape(128, 1, 1, 1)

    def forward(self, bottom, top):
        neg_size = 96
        pos_size = 32
        # prepare inputs: batch_size/per time
        for i in range(neg_size):
            features, labels = self._readLine(self.ptr_neg, self.reader_neg)
            dates = self._readDate(self.date_neg, self.rdate_neg)
            top[0].data[i,...,0,0] = features
            top[1].data[i,...,0,0] = dates
            top[2].data[i,0,0,0] = labels
        for i in range(pos_size):
            j = i+neg_size
            features, labels = self._readLine(self.ptr_pos, self.reader_pos)
            dates = self._readDate(self.date_pos, self.rdate_pos)
            top[0].data[j,...,0,0] = features
            top[1].data[j,...,0,0] = dates
            top[2].data[j,0,0,0] = labels

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

class BoschInferenceInput(caffe.Layer):
    def _openFile(self):
        self.ptr1 = open('data/bosch/test_numeric.csv',"r")
        self.ptr2 = open('data/bosch/test_date.csv',"r")
        self.reader1 = csv.reader(self.ptr1)
        self.reader2 = csv.reader(self.ptr2)
        header = next(self.reader1) #ingore header
        header = next(self.reader2) #ingore header

    def _readLine(self, ptr, reader):
        try:
            line = next(reader)
        except StopIteration:
            return 0, 0
        # read data and write missing to 999
        raw = line[0:(len(line))]
        for i in range(len(raw)):
            if not raw[i]:
                raw[i] = 999
        # convert to np array
        tmp = np.array(raw, dtype='Float32')
        raw = tmp.astype(np.float)
        # feature
        feature = np.zeros((len(raw)-1))
        for i in range(len(raw)-1): feature[i] = raw[i+1]
        # index
        index = raw[0]
        return feature, index

    def _readDate(self, ptr, reader):
        line = next(reader)
        # read data and write missing to 999
        raw = line[1:(len(line))]
        for i in range(len(raw)):
            if not raw[i]:
                raw[i] = -999
        # convert to np array
        tmp = np.array(raw, dtype='Float32')
        raw = tmp.astype(np.float)
        # feature
        date = np.zeros((len(raw)))
        for i in range(len(raw)): date[i] = raw[i]
        return date

    def setup(self, bottom, top):
        self._openFile()
        top[0].reshape(128, 968, 1, 1)
        top[1].reshape(128, 1156, 1, 1)
        top[2].reshape(128, 1, 1, 1)

    def forward(self, bottom, top):
        # prepare inputs: batch_size/per time
        for i in range(128):
            features, indexes = self._readLine(self.ptr1, self.reader1)
            if indexes > 0:
                dates = self._readDate(self.ptr2, self.reader2)
                top[0].data[i,...,0,0] = features
                top[1].data[i,...,0,0] = dates
                top[2].data[i,0,0,0] = indexes
            else:
                top[2].data[i,0,0,0] = 0

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

class BoschInferenceOutput(caffe.Layer):
    def _openFile(self):
        self.wptr = open('data/bosch/test_response.csv',"w")
        self.writer  = csv.writer(self.wptr)
        self.writer.writerow(["Id","Response"]) #header

    def setup(self, bottom, top):
        self._openFile()

    def forward(self, bottom, top):
        for i in range(128):
            if int(bottom[1].data[i,0,0,0]) > 0:
                value = np.argmax(bottom[0].data[i,...])
                self.writer.writerow([int(bottom[1].data[i,0,0,0]),value])
            else:
                break;

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
