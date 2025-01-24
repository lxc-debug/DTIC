import sys
sys.path.append('./config')
sys.path.append('./utils')
sys.path.append('./model')
import subprocess
import shlex
from config.log_conf import file_logger
import os

lr_li = ['1e-3']
weight_decay_li = ['1e-7']

for weight_decay in weight_decay_li:
    for lr in lr_li:
        try:
            subprocess.run(shlex.split(f'python main.py --lr {lr} --weight_decay {weight_decay}   --logdir {"./log_tmp/main_augment/one_test.log"}  --use_list {"leader_one"} --augment --use_base --use_q_node --process_data --aug_level {3} --seed {0}'),check=True)
        except subprocess.CalledProcessError:
            file_logger.error('something wrong in run.py')
