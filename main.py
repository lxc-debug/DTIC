from model.simple_model import SimpleModel, SimpleModelQ, SimpleModelAttach
from dataset import MyDataset, TestDataset, PosEmbTestDataset, MainDataset, PosMainDataset, AugmentMainDataset, AugmentOriMainDataset
from train import Experiment, ExperimentTest, ExperimentPosEmbTest, MainExperiment, PosMainExperiment
from config.log_conf import logger, file_logger
import logging
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


# use test
# tar_one()
# print(glob('./log/*'))

my_seed = args.seed
np.random.seed(seed=my_seed)
random.seed(a=my_seed)
torch.manual_seed(my_seed)
torch.cuda.manual_seed(my_seed)
torch.cuda.manual_seed_all(my_seed)

if __name__ == '__main__':
    if not args.augment:
        if not args.main_pos:
            if not args.main:
                if not args.test:
                    file_logger.info(
                        f'parameter config lr:{args.lr},weight_decay:{args.weight_decay}')

                    train_dataset = MyDataset(mode='train')
                    eval_dataset = MyDataset(mode='eval')
                    test_dataset = MyDataset(mode='test')

                    if args.use_q:
                        model = SimpleModelQ()
                    elif args.use_three:
                        model = SimpleModelAttach()
                    else:
                        model = SimpleModel()

                    start_train = Experiment(model, train_dataset,
                                            eval_dataset, test_dataset)
                    start_train()

                else:
                    if not args.add_par_pos_emb:
                        file_logger.info(
                            f'parameter config lr:{args.lr},weight_decay:{args.weight_decay}')

                        train_dataset = TestDataset(mode='train')
                        eval_dataset = TestDataset(mode='eval')
                        test_dataset = TestDataset(mode='test')

                        model = ModelTestNodeAggregate()

                        start_train = ExperimentTest(
                            model, train_dataset, eval_dataset, test_dataset)
                        start_train()

                    else:
                        file_logger.info(
                            f'parameter config lr:{args.lr},weight_decay:{args.weight_decay}')

                        train_dataset = PosEmbTestDataset(mode='train')
                        eval_dataset = PosEmbTestDataset(mode='eval')
                        test_dataset = PosEmbTestDataset(mode='test')

                        model = ModelTestNodePosEmbAggregate()

                        start_train = ExperimentPosEmbTest(
                            model, train_dataset, eval_dataset, test_dataset)
                        start_train()

            else:
                file_logger.info(
                            f'parameter config lr:{args.lr},weight_decay:{args.weight_decay}')

                train_dataset = MainDataset(mode='train')
                eval_dataset = MainDataset(mode='eval')
                test_dataset = MainDataset(mode='test')

                model = MainModel()

                start_train = MainExperiment(
                    model, train_dataset, eval_dataset, test_dataset)
                start_train()
        else:
            file_logger.info(
                            f'parameter config lr:{args.lr},weight_decay:{args.weight_decay}')

            train_dataset = PosMainDataset(mode='train')
            eval_dataset = PosMainDataset(mode='eval')
            test_dataset = PosMainDataset(mode='test')

            model = PosMainModel()

            start_train = PosMainExperiment(
                model, train_dataset, eval_dataset, test_dataset)
            start_train()
    else:
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
        # train_dataset=test_dataset
        # eval_dataset=test_dataset

        start_train = MainExperiment(
            model, train_dataset, eval_dataset, test_dataset)
        start_train()
