import sys
sys.path.append('./config')
sys.path.append('./utils')
sys.path.append('./model')
import subprocess
import shlex
from config.log_conf import file_logger
import os


lr_li = ['1e-3']
weight_decay_li = ['1e-6', '1e-7','1e-8']

for weight_decay in weight_decay_li:
    for lr in lr_li:
        try:
            subprocess.run(shlex.split(f'python main.py --lr {lr} --weight_decay {weight_decay}   --logdir {"./log_tmp/mix/two_aug3_best.log"}  --use_list {"leader_two"} --augment --use_base --use_q_node --process_data --aug_level {3} --seed {2}'),check=True)
        except subprocess.CalledProcessError:
            file_logger.error('something wrong in run.py')

