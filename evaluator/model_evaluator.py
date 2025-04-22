import numpy as np
import random
import sklearn.metrics as sk_metrics
import torch
import torch.nn.functional as F


from tqdm import tqdm
from .output_aggregator import OutputAggregator

###
import pandas as pd
import numpy as np
import torch.nn as nn
import sys
import os


import util



import warnings
warnings.filterwarnings("ignore")


def save_data_to_csv(path_csv, idx, label, data, num_dim, start_idx = None):
    path = path_csv
    
    if start_idx is None:
        new_data = pd.DataFrame({'idx':[idx], 'label':[label], 'count':[1]})
        data_columns = [f"data{i}" for i in range(num_dim)]
        new_data = pd.concat([new_data, pd.DataFrame(data, columns = data_columns)], axis = 1)

        if not os.path.exists(path):
            new_data.to_csv(path_csv, index = False)
        else:
            existing_data = pd.read_csv(path)
            if idx in existing_data['idx'].values:
                existing_idx = existing_data['idx'] == idx
                existing_count = existing_data.loc[existing_idx, 'count'].values[0]
                new_count = existing_count + 1
                existing_data.loc[existing_idx, 'count'] = new_count
                existing_data.loc[existing_idx, 'label'] = label

                for col in data_columns:
                    existing_data.loc[existing_idx, col] = (existing_data.loc[existing_idx, col] * existing_count + new_data[col].values[0]) / new_count
            else:
                existing_data = pd.concat([existing_data, new_data], ignore_index = True)
            existing_data.to_csv(path_csv, index = False)
    else:
        new_data = pd.DataFrame({'idx':[idx], 'start_idx':[start_idx], 'label':[label], 'count':[1]})
        data_columns = [f"data{i}" for i in range(num_dim)]
        new_data = pd.concat([new_data, pd.DataFrame(data, columns = data_columns)], axis = 1)

        if not os.path.exists(path):
            new_data.to_csv(path_csv, index = False)
        else:
            existing_data = pd.read_csv(path)
            if idx in existing_data['idx'].values and start_idx in existing_data['start_idx'].values:
                existing_idx = existing_data['idx'] == idx
                existing_idx = existing_data['start_idx'] == start_idx
                existing_count = existing_data.loc[existing_idx, 'count'].values[0]
                new_count = existing_count + 1
                existing_data.loc[existing_idx, 'count'] = new_count
                existing_data.loc[existing_idx, 'label'] = label

                for col in data_columns:
                    existing_data.loc[existing_idx, col] = (existing_data.loc[existing_idx, col] * existing_count + new_data[col].values[0]) / new_count
            else:
                existing_data = pd.concat([existing_data, new_data], ignore_index = True)
            existing_data.to_csv(path_csv, index = False)




class ModelEvaluator():
    def __init__(self, val_loader, agg_method, **kwargs):

        
        self.val_loader = val_loader #w 标记为val的数据子集
        self.aggregator = OutputAggregator(agg_method) #w 聚合输出，取最大值或者均值




        #w 设置损失函数
        self.loss_bf = util.optim_util.get_loss_fn('bf')

        

        #w 其他数据路径
        
        self.data_tab_path = kwargs.get('data_tab_path')
        
        
        

    def evaluate(self, model, device, loss_record):


        #w 读取其他数据
        tab = pd.read_csv(self.data_tab_path) 




        metrics={}




  
        data_loader = self.val_loader[0]


        model.eval()
        phase_metrics, loss_record = self._eval_phase(model, data_loader, device, tab, loss_record)
        metrics.update(phase_metrics)
        
            
        model.train()




      
        return metrics, loss_record




    def _eval_phase(self, model, data_loader, device, table, loss_record):
        


        
        





     
        bs=data_loader.bs

        
        #w
        records = {'keys':[], 'probs':[]}
        
        num_evaluated = 0 #w 已验证的样本数，初始值为0可以保证所有样本被验证
        num_examples = len(data_loader.dataset) #w 总样本数


        #w 辅助变量
        val_total_samples = 0
        out = None
        for_tsne = False


        with tqdm(total = num_examples, unit = ' example') as progress_bar:
          


            for img, target_dict in data_loader:
                if num_evaluated >= num_examples:
                    break
                
                #w 加载其他数据
                #w lung
                ids = [item for item in target_dict['study_num']]
                tab = []
                for i in range(len(target_dict['study_num'])):
                    data = table[table['NewPatientID'] == ids[i]].iloc[0, 1:8].astype(np.float32)
                    tab.append(torch.tensor(data, dtype=torch.float32))
                tab=torch.stack(tab).squeeze(1)


                #w PE
                '''ids = [int(item) for item in target_dict['study_num']]
                tab = []
                for idx in ids:
                    data = table[table['idx'] == idx].iloc[:, 4:].astype(np.float32)
                    tab.append(torch.tensor(np.array(data), dtype = torch.float32))
                tab = torch.stack(tab).squeeze(1)'''




                with torch.no_grad():
                    img = img.to(device)
                    tab = tab.to(device)
                    label = target_dict['is_abnormal'].to(device)

                    label = label.float().unsqueeze(1)
                    


                    #w Lung，你需要确保损失值都是用均值聚合后的
                    output = model(img, tab)
                    loss = output['all_loss']



                    #w update loss_record
                    for k, v in output.items():
                        if k.endswith('_loss') and v != -1:
                            len_sample = img.shape[0]  
                            val_total_samples += len_sample
                            if 'val_' + k not in loss_record:
                                loss_record['val_' + k] = v.item() * len_sample
                            else:
                                loss_record['val_' + k] += v.item() * len_sample
                        elif k.endswith('_loss'):
                            len_sample = img.shape[0]  
                            val_total_samples += len_sample
                            if 'val_' + k not in loss_record:
                                loss_record['val_' + k] = loss_out.item() * len_sample
                            else:
                                loss_record['val_' + k] += loss_out.item() * len_sample


                    #w
                   
                    cls_logits = out if out is not None else torch.randn([4, 1])

                #w 每次循环都执行
                self._record_batch(cls_logits, target_dict['series_idx'], **records)
                progress_bar.update(min(bs, num_examples - num_evaluated))
                num_evaluated +=bs



        if for_tsne:
            print('done')
            sys.exit()
        #w 在循环外
        metrics =self._get_summary_dicts(data_loader,  device, **records)

        ###

        #w
        for k, v in loss_record.items():
            if k.startswith('val') and k.endswith('_loss'):
                loss_record[k] /= val_total_samples



        return metrics, loss_record





    @staticmethod
    def _record_batch(logits, targets, probs = None, keys = None):
        #w 初始传入的keys和probs不是None，是空列表
        #w 存在多个相同的key，所以用聚合解决了这一问题
        if probs is not None:
            with torch.no_grad():
                batch_probs = F.sigmoid(logits)
            probs.append(batch_probs.detach().cpu())
            keys.append(targets.detach().cpu())
        



    def _get_summary_dicts(self, data_loader, device, probs=None, keys=None):

        metrics = {}

        if probs is not None:


            probs = np.concatenate(probs).ravel().tolist()
            keys = np.concatenate(keys).ravel().tolist()

            #w 所有的输出和key都拿到之后，由于一个key会有多个输出，所有用了聚合，最后得到一个输出
            key2prob = self.aggregator.aggregate(keys, probs, data_loader, device)
            probs, labels=[], []
            for idx, prob in key2prob.items():
                probs.append(prob)
                labels.append(data_loader.get_series_label(idx))
            probs, labels = np.array(probs), np.array(labels)



            #w 在这里的loss就是交叉熵
            metrics.update({
                'val' + '_' + 'loss': sk_metrics.log_loss(labels, probs, labels = [0, 1]),
                'val' + '_' + 'AUPRC': sk_metrics.average_precision_score(labels, probs),
                'val' + '_' + 'AUROC': sk_metrics.roc_auc_score(labels, probs),
            })




        return metrics
