"""
Simple NN for Bosch competitation at Kaggle
Written by Aasta Lin
BSD License
"""
import numpy as np
import pickle
import csv

# data I/O
ptr_nume_pos = open('train_numeric_pos.csv',"r")
ptr_nume_neg = open('train_numeric_neg.csv',"r")
ptr_date_pos = open('train_date_pos.csv',"r")
ptr_date_neg = open('train_date_neg.csv',"r")
reader_nume_pos = csv.reader(ptr_nume_pos)
reader_nume_neg = csv.reader(ptr_nume_neg)
reader_date_pos = csv.reader(ptr_date_pos)
reader_date_neg = csv.reader(ptr_date_neg)
header = next(reader_nume_pos) #ingore header
header = next(reader_nume_neg) #ingore header
header = next(reader_date_pos) #ingore header
header = next(reader_date_neg) #ingore header

# hyperparameters
feature_size = 968+1156
output_size = 2
batch_size = 128
neg_size = 96
pos_size = 32
hidden_size = 128 # size of hidden layer of neurons
learning_rate = 0.5

"""
# model parameters
Wxh = np.random.randn(hidden_size, feature_size)*0.01 # input to hidden
Why = np.random.randn(output_size, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((output_size, 1)) # output bias
"""

# load model
with open('weight.pickle') as f:
  Wxh, Why, bh, by = pickle.load(f)

def readLine(ptr,reader):
  try:
    line = next(reader)
  except StopIteration:
    ptr.seek(0)
    header = next(reader)
    return readLine(ptr,reader)
  # read data and write missing to 999
  raw = line[1:(len(line))]
  for i in range(len(raw)):
    if not raw[i]:
      raw[i] = 0
  # convert to np array
  tmp = np.array(raw, dtype='Float32')
  raw = tmp.astype(np.float)
  # feature
  feature = np.zeros((len(raw)-1,1))
  for i in range(len(raw)-1): feature[i] = raw[i]
  # response
  response = raw[len(raw)-1]
  return feature, response

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

def testFun(inputs, targets):
  pos = 0
  neg = 0
  acc = 0
  # forward pass
  for t in xrange(len(inputs)):
    hs = np.tanh(np.dot(Wxh, inputs[t]) + bh) # hidden state
    ys = np.dot(Why,hs) + by # unnormalized log probabilities
    ps = np.exp(ys) / np.sum(np.exp(ys)) # probabilities
    ans = np.argmax(ps)
    if ans==1:
      pos = pos + 1
    if ans==0:
      neg = neg + 1
    if ans==int(targets[t]):
      acc = acc + 1
  print 'pos %d, neg %d, accuracy: %f' % (pos, neg, (float(acc)/batch_size))

def lossFun(inputs, targets):
  """
  inputs,targets are both np float array
  returns the loss, gradients on model parameters, and last hidden state
  """
  hs, ys, ps, ts = {}, {}, {}, {}
  loss = 0
  # forward pass
  for t in xrange(len(inputs)):
    ts[t] = np.zeros((output_size,1)) # encode in 1-of-k representation
    ts[t][int(targets[t])] = 1
    hs[t] = np.tanh(np.dot(Wxh, inputs[t]) + bh) # hidden state
    ys[t] = np.dot(Why,hs[t]) + by # unnormalized log probabilities
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities
    loss += -np.log(np.dot(ts[t].T,ps[t])) # softmax (cross-entropy loss)
  # backward pass: compute gradients going backwards
  dWxh, dWhy = np.zeros_like(Wxh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  for t in reversed(xrange(len(inputs))):
    dy = np.copy(ps[t])
    dy[int(targets[t])] -= 1 # backprop into y
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(Why.T, dy) # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, inputs[t].T)
  for dparam in [dWxh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip gradients
  return loss, dWxh, dWhy, dbh, dby

n = 0
mWxh,  mWhy = np.zeros_like(Wxh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/feature_size)*batch_size # loss at iteration 0

while True:
  # prepare inputs: batch_size/per time
  inputs, targets = {}, {}
  for i in range(neg_size):
    numeric, targets[i] = readLine(ptr_nume_neg, reader_nume_neg)
    date = readDate(ptr_date_neg, reader_date_neg)
    inputs[i] = np.concatenate((numeric, date), axis=0)
  for i in range(pos_size):
    numeric, targets[i+neg_size] = readLine(ptr_nume_pos, reader_nume_pos)
    date = readDate(ptr_date_pos, reader_date_pos)
    inputs[i+neg_size] = np.concatenate((numeric, date), axis=0)

  # quite predit
  if n % 10 == 0:
    testFun(inputs, targets)

  # forward seq_length characters through the net and fetch gradient
  loss, dWxh, dWhy, dbh, dby = lossFun(inputs, targets)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 10 == 0:
    print 'iter %d, loss: %f' % (n, smooth_loss) # print progress

  # dump weight
  if n == 10000:
    with open('weight_%d.pickle' % n, 'w') as f:
      pickle.dump([Wxh, Why, bh, by], f)
    break;

  # perform parameter update with Adagrad
  for param, dparam, mem in zip([Wxh, Why, bh, by], 
                                [dWxh, dWhy, dbh, dby], 
                                [mWxh, mWhy, mbh, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  n += 1 # iteration counter
