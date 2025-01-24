from dataset import  AugmentOriMainDataset
from train import  MainExperiment
from config.log_conf import logger, file_logger
import logging
import torch
import random
import numpy as np
from utils.tar import *
from glob import glob
from model.attention_model import MainSparseModel
from config.conf import args
import sys
sys.path.append('./config')
sys.path.append('./utils')
sys.path.append('./model')




my_seed = args.seed
np.random.seed(seed=my_seed)
random.seed(a=my_seed)
torch.manual_seed(my_seed)
torch.cuda.manual_seed(my_seed)
torch.cuda.manual_seed_all(my_seed)

if __name__ == '__main__':

    file_logger.info(
        f'parameter config lr:{args.lr},weight_decay:{args.weight_decay}, augment level:{args.aug_level}, seed:{args.seed}')
    
    model = MainSparseModel()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device)

    train_dataset = AugmentOriMainDataset(mode='train')
    eval_dataset = AugmentOriMainDataset(mode='eval')
    test_dataset = AugmentOriMainDataset(mode='test')
    

    start_train = MainExperiment(
        model, train_dataset, eval_dataset, test_dataset)
    start_train()
