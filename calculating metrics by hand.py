# building accuracy, recall, precision, and f1 from scratch

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# set random seed
seed = 8222025
np.random.seed(seed)


# first generate two arrays
y_pred = np.random.choice([True, False], size = 10000)
y_true = np.random.choice([True, False], size = 10000)

# all scores are tested against scikit-learn calculations
# calculate accuracy
'''
pseudo code:
accuracy = (# of true positives + # of true negatives) / total predictions
i.e. how good are you at getting predictions right
'''
df = pd.DataFrame({"y_pred": y_pred,
                   "y_true": y_true})

tp = df[(df["y_pred"] == True) & (df["y_true"] == True)].shape[0]
tn = df[(df["y_pred"] == False) & (df["y_true"] == False)].shape[0]
total = df.shape[0]

accuracy = (tp + tn) / total
accuracy
accuracy_score(y_true, y_pred)



# calculate recall
'''
psuedo code:
recall = true positives / (true positives + false negatives)
i.e. how good are you at capturing positive class
'''
fn = df[(df["y_pred"] == False) & (df["y_true"] == True)].shape[0]

recall = tp / (tp + fn)
recall
recall_score(y_true, y_pred)


#calculate precision
'''
psedo code:
precision = true positives / (true positives + false positives)
i.e. how reliable are your positive predictions?
'''

fp = df[(df["y_true"] == False) & (df["y_pred"] == True)].shape[0]
precision = tp / (tp + fp)
precision
precision_score(y_true, y_pred)


# calculate f1
'''
pseudocode:
f1 = 2 * (recall * precision) / (recall + precision)
i.e. the harmonic mean between precision and recall
use this when classes are imbalanced and you want to 
'''

f1 = 2 * (recall * precision) / (recall + precision)
f1
f1_score(y_true, y_pred)
