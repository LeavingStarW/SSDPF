import cv2
import json
import pickle
import numpy as np
import os
import sklearn.metrics as sk_metrics
import torch
import torch.nn.functional as F
import util

from args import TestArgParser
from data_loader import CTDataLoader
from collections import defaultdict

from PIL import Image
from saver import ModelSaver
from tqdm import tqdm

###
import pandas as pd
import numpy as npa

import warnings
warnings.filterwarnings("ignore")


class CTPE:
    def __init__(self, name, is_positive, parser, num_slice, first_appear, avg_bbox, last_appear):
        self.study_num = name
        self.is_positive = is_positive
        self.phase = parser
        self.num_slice = num_slice
        self.first_appear = first_appear
        self.bbox = avg_bbox
        self.last_appear = last_appear

    def __len__(self):
        return self.num_slice


###
def test(args,table):
    model, ckpt_info = ModelSaver.load_model(args.ckpt_path, args.gpu_ids)
    model = model.to(args.device)
    model.eval()
    
    data_loader = CTDataLoader(args, phase='test', is_training=False)
    study2slices = defaultdict(list)
    study2probs = defaultdict(list)
    study2labels = {}




    # Get model outputs, log to TensorBoard, write masks to disk window-by-window

    with tqdm(total=len(data_loader.dataset), unit=' windows') as progress_bar:
        for i, (inputs, targets_dict) in enumerate(data_loader):
            
            #w
            ids = [item for item in targets_dict['study_num']]
            tab=[]
            table = pd.read_csv(args.data_tab_path)
            for i in range(len(targets_dict['study_num'])):
                data = table[table['NewPatientID'] == ids[i]].iloc[0, 1:8].astype(np.float32)
                tab.append(torch.tensor(data, dtype=torch.float32))
            tab = torch.stack(tab).squeeze(1)

            ###
            '''ids = [int(item) for item in targets_dict['study_num']]
            table_data=[]
            for i in range(len(targets_dict['study_num'])):
                table_data.append(torch.tensor(np.array(table[table['idx']==ids[i]].iloc[:,4:]),dtype=torch.float32))
            table_data = torch.stack(table_data).squeeze(1)'''

 
            with torch.no_grad():

                #w 数据
                img =inputs.to(args.device)
                tab =tab.to(args.device)
                label =targets_dict['is_abnormal'].to(args.device)


                #w forward 
                output = model(tab)
                cls_logits = output['out']
                cls_probs = F.sigmoid(cls_logits)



            #w
            max_probs = cls_probs.to('cpu').numpy()
            for study_num, slice_idx, prob in \
                    zip(targets_dict['study_num'], targets_dict['slice_idx'], list(max_probs)):
                
                
                # Convert to standard python data types
                '''study_num = int(study_num)
                slice_idx = int(slice_idx)'''
                study_num = str(study_num)
                slice_idx = int(slice_idx)

                # Save series num for aggregation
                study2slices[study_num].append(slice_idx)
                study2probs[study_num].append(prob.item())

                series = data_loader.get_series(study_num)
                if study_num not in study2labels:
                    study2labels[study_num] = int(series.is_positive)

            progress_bar.update(img.size(0))
    
    # Combine masks

    max_probs = []
    labels = []
    predictions = {}
    print("Get max probability")
    for study_num in tqdm(study2slices):

        # Sort by slice index and get max probability
        slice_list, prob_list = (list(t) for t in zip(*sorted(zip(study2slices[study_num], study2probs[study_num]),
                                                              key=lambda slice_and_prob: slice_and_prob[0])))
        study2slices[study_num] = slice_list
        study2probs[study_num] = prob_list
        max_prob = max(prob_list)
        max_probs.append(max_prob)
        label = study2labels[study_num]
        labels.append(label)
        predictions[study_num] = {'label':label, 'pred':max_prob}

    #Save predictions to file, indexed by study number
    print("Save to pickle")
    with open('{}/{}.pickle'.format(args.results_preds_dir, args.name),"wb") as fp:
        pickle.dump(predictions,fp)
        
    # Write features for XGBoost
    save_for_xgb(args.results_preds_dir, study2probs, study2labels)
    # Write the slice indices used for the features
    print("Write slice indices")
    with open(os.path.join(args.results_preds_dir, 'xgb', 'series2slices.json'), 'w') as json_fh:
        json.dump(study2slices, json_fh, sort_keys=True, indent=4)

    # Compute AUROC and AUPRC using max aggregation, write to files
    max_probs, labels = np.array(max_probs), np.array(labels)
    metrics = {
        'test' + '_' + 'AUPRC': sk_metrics.average_precision_score(labels, max_probs),
        'test' + '_' + 'AUROC': sk_metrics.roc_auc_score(labels, max_probs),
    }
    print("Write metrics")
    with open(os.path.join(args.results_preds_dir, 'metrics.txt'), 'w') as metrics_fh:
        for k, v in metrics.items():
            metrics_fh.write('{}: {:.5f}\n'.format(k, v))

    curves = {
        'test' + '_' + 'PRC': sk_metrics.precision_recall_curve(labels, max_probs),
        'test' + '_' + 'ROC': sk_metrics.roc_curve(labels, max_probs)
    }
    for name, curve in curves.items():
        curve_np = util.get_plot(name, curve)
        curve_img = Image.fromarray(curve_np)
        curve_img.save(os.path.join(args.results_preds_dir, '{}.png'.format(name)))


def save_for_xgb(results_preds_dir, series2probs, series2labels):
    """Write window-level and series-level features to train an XGBoost classifier.
    Args:
        results_preds_dir: Path to results directory for writing outputs.
        series2probs: Dict mapping series numbers to probabilities.
        series2labels: Dict mapping series numbers to labels.
    """

    # Convert to numpy
    xgb_inputs = np.zeros([len(series2probs), max(len(p) for p in series2probs.values())])
    xgb_labels = np.zeros(len(series2labels))
    for i, (series_num, probs) in enumerate(series2probs.items()):
        xgb_inputs[i, :len(probs)] = np.array(probs).ravel()
        xgb_labels[i] = series2labels[series_num]

    # Write to disk
    os.makedirs(os.path.join(results_preds_dir, 'xgb'), exist_ok=True)
    xgb_inputs_path = os.path.join(results_preds_dir, 'xgb', 'inputs.npy')
    xgb_labels_path = os.path.join(results_preds_dir, 'xgb', 'labels.npy')
    np.save(xgb_inputs_path, xgb_inputs)
    np.save(xgb_labels_path, xgb_labels)


if __name__ == '__main__':
    ###
    table_data = pd.read_csv('/data2/wangchangmiao/wzp/data/Lung/G_first_last_nor.csv')

    util.set_spawn_enabled()
    parser = TestArgParser()
    args_ = parser.parse_args()
    ###
    test(args_,table_data)
