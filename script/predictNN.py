"""
Simple NN for Bosch competitation at Kaggle
Written by Aasta Lin
BSD License
"""
import numpy as np
import pickle
import csv

# data I/O
rptr = open('test_numeric.csv',"r")
ptr = open('test_date.csv',"r")
wptr = open('test_response.csv',"w")
reader = csv.reader(rptr)
eader = csv.reader(ptr)
writer = csv.writer(wptr)
header = next(reader) #header
header = next(eader) #header
writer.writerow(["Id","Response"]) #header

# hyperparameters
feature_size = 968+1156
output_size = 2
batch_size = 128
neg_size = 96
pos_size = 32
hidden_size = 128 # size of hidden layer of neurons

# load model
with open('weight.pickle') as f:
  Wxh, Why, bh, by = pickle.load(f)

def readDate(ptr, reader):
  try:
    line = next(reader)
  except StopIteration:
    ptr.seek(0)
    header = next(reader)
    return readDate(ptr,reader)
  # read data and write missing to 999
  raw = line[1:(len(line))]
  for i in range(len(raw)):
    if not raw[i]:
      raw[i] = 0
  # convert to np array
  tmp = np.array(raw, dtype='Float32')
  raw = tmp.astype(np.float)
  # feature
  date = np.zeros((len(raw),1))
  for i in range(len(raw)): date[i] = raw[i]
  return date

def forward(inputs):
  """
  inputs is np float array
  returns prediction
  """
  # forward pass
  hs = np.tanh(np.dot(Wxh, inputs) + bh) # hidden state
  ys = np.dot(Why,hs) + by # unnormalized log probabilities
  ps = np.exp(ys) / np.sum(np.exp(ys)) # probabilities
  return np.argmax(ps)

for line in reader:
  # read data and write missing to 999
  raw = line[1:(len(line))]
  for i in range(len(raw)):
    if not raw[i]:
      raw[i] = 0

  # convert to np array
  tmp = np.array(raw, dtype='Float32')
  raw = tmp.astype(np.float)

  # feature
  feature = np.zeros((len(raw),1))
  for i in range(len(raw)): feature[i] = raw[i]

  date = readDate(ptr, eader)
  inputs = np.concatenate((feature, date), axis=0)

  # predict
  value = forward(inputs)
  writer.writerow([line[0],value])
