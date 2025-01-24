from model.simple_model import SimpleModel, SimpleModelQ, SimpleModelAttach
from dataset import MyDataset, TestDataset, PosEmbTestDataset, MainDataset, PosMainDataset, AugmentOriMainDataset
from train import Experiment, ExperimentTest, ExperimentPosEmbTest, MainExperiment, PosMainExperiment
from config.log_conf import logger, file_logger
import torch
import random
import numpy as np
from utils.tar import *
from glob import glob
from model.attention_model import ModelTestNodeAggregate, ModelTestNodePosEmbAggregate, MainModel, PosMainModel, MainSparseModel
from config.conf import args
import sys
sys.path.append('./config')
sys.path.append('./utils')
sys.path.append('./model')

from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
from utils.early_stopping import EarlyStopping
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import DataLoader
from config.conf import args
from config.log_conf import logger, file_logger
import os
from sklearn.metrics import roc_auc_score
import numpy as np
import time

my_seed = 1
np.random.seed(seed=my_seed)
random.seed(a=my_seed)
torch.manual_seed(my_seed)
torch.cuda.manual_seed(my_seed)
torch.cuda.manual_seed_all(my_seed)

class TestExperiment():
    def __init__(self, model: nn.Module, test_dataset) -> None:
        self.test_dataset = test_dataset
        self.test_dataloader = DataLoader(
            self.test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=self.test_dataset.coll_fn)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model = model.to(self.device)

        self.optim = optim.Adam(self.model.parameters(
        ), lr=args.lr, weight_decay=args.weight_decay)
        self.epochs = args.epochs
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # self.save_dir='./best_parameter/leader_four/best_base.pt'

    @torch.no_grad()
    def _test(self):
        self.model.eval()

        self.pre_label = list()
        self.label = list()

        self.total_test_correct = 0
        self.total_test_loss = 0

        for graph, data, row_mask, node_mask, label in tqdm(self.test_dataloader, desc='test'):
            # data = data.to(self.device)
            # label = label.to(self.device)
            # row_mask = row_mask.to(self.device)
            # node_mask = node_mask.to(self.device)
            # graph = graph.to(self.device)
            

            res, attn = self.model(graph, data, row_mask, node_mask)
            loss = self.loss_fn(res, label)

            correct = (torch.argmax(res, dim=-1) == label).sum().item()


            self.total_test_correct += correct
            self.total_test_loss += loss.item()*res.shape[0]

            self.pre_label.extend(
                torch.nn.functional.softmax(res, dim=-1)[:, 1].tolist())
            self.label.extend(label.tolist())
            


        acc = self.total_test_correct/len(self.test_dataset)
        loss = self.total_test_loss/len(self.test_dataset)
        print(self.pre_label[:5],self.label[:5])
        score = roc_auc_score(np.array(self.label), np.array(self.pre_label))

        print(
            f'test dataset acc:{acc:8.4f}|loss:{loss:8.4f}|auc_roc_score:{score:8.6f}')
        
        # print(
        #     f'test dataset acc:{acc:8.4f}|loss:{loss:8.4f}')

    def __call__(self) -> None:

        # self.writer.close()
        # self.model.load_state_dict(torch.load(self.save_dir))
        self._test()


start=time.time()
model = MainSparseModel()
model.load_state_dict(torch.load('best_parameter/leader_three/aug_l1_3_best.pt'))
test_dataset = AugmentOriMainDataset(mode='test')

start_train = TestExperiment(
    model, test_dataset)
start_train()
end=time.time()

print(f'total use time:{end-start:4f}')