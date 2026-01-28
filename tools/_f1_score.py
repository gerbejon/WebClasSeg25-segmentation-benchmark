from collections import Counter

from datasets import load_dataset
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd

def f1_score_nodes(y_true, y_pred):
    y_true, y_pred = list(y_true.values()), list(y_pred.values())
    labels = [i for i in np.unique(y_true + y_pred)]
    f1_values = [float(f1) for f1 in f1_score(y_true=y_true, y_pred=y_pred, average=None, labels=labels)]
    f1_dict = dict(zip(labels, f1_values))
    return f1_dict

def f1_score_mask(mask, img_id, segm=None, ds=None):
    if ds == None:
        ds = load_dataset(f'gerbejon/WebClasSeg25-visual-{segm}')

    for row in ds['test']:
        if row['page_id'] == img_id:
            mask_true = np.array(row['annotation'])

            y_pred = [int(i) for i in mask.flatten()]
            y_true = [int(i) for i in mask_true.flatten()]

            labels = [int(i) for i in np.unique(y_pred + y_true)]
            f1_values = [float(f1) for f1 in f1_score(y_true=y_true, y_pred=y_pred, average=None, labels=list(labels))]
            # pair labels with their scores
            f1_dict = dict(zip(labels, f1_values))
            # f1_dict['img_id'] = img_id
            pixel_true = dict(Counter(y_true))
            pixel_pred = dict(Counter(y_pred))
            yield f1_dict