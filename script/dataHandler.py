"""
Data handler for Bosch competitation at Kaggle
Written by Aasta Lin
BSD License
"""
import numpy as np
import csv

# data I/O
rptr1 = open('train_numeric.csv',"r")
rptr2 = open('train_date.csv',"r")
wptr1_pos = open('train_numeric_pos.csv',"w")
wptr1_neg = open('train_numeric_neg.csv',"w")
wptr2_pos = open('train_date_pos.csv',"w")
wptr2_neg = open('train_date_neg.csv',"w")

reader1 = csv.reader(rptr1)
reader2 = csv.reader(rptr2)
pwriter1 = csv.writer(wptr1_pos)
nwriter1 = csv.writer(wptr1_neg)
pwriter2 = csv.writer(wptr2_pos)
nwriter2 = csv.writer(wptr2_neg)

# parse header
header1 = next(reader1)
header2 = next(reader2)
pwriter1.writerow(header1)
nwriter1.writerow(header1)
pwriter2.writerow(header2)
nwriter2.writerow(header2)

# parse file
for line1 in reader1:
  line2 = next(reader2)
  if line1[len(line1)-1]=='1':
    pwriter1.writerow(line1)
    pwriter2.writerow(line2)
  else:
    nwriter1.writerow(line1)
    nwriter2.writerow(line2)
