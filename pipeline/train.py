import os
import sys
import torch
import random
import argparse
import numpy as np

from glob import glob

# PROJ ROOT DIR
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.join(DIR_PATH, os.path.pardir)
sys.path.append(ROOT_PATH)

import pipeline.constants as const
from pipeline.utils import * 
from pipeline.model_setup import ModelSetup
from model.run import run_model

SEED = 12345
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def train(model_path):
    model_setup = ModelSetup(load=model_path)

    for epoch in range(model_setup.epoch, model_setup.max_epoch):
        model_setup.epoch = epoch
        train_out = run_model(model_setup, train=True, test=False)
        model_setup.sched.step(epoch)
        print(f"epoch: {epoch},\t"\
                f"avg_loss: {train_out['avg_loss']:1.4f},\t"\
                f"regression loss: {train_out['avg_reg_loss']:1.4f},\t"\
                f"classification loss: {train_out['avg_class_loss']:1.4f}"\
                f"mean error: {train_out['mean_err'] : 1.4f},\t"\
                f"max error: {train_out['max_err']:1.4f}\t"\
                f"lr rate: {model_setup.sched.get_lr()[0]:1.9f}")

        if (epoch+1) % model_setup.save_freq == 0:
            train_val, mean_error_list, max_error_list = run_model(model_setup, train=False)
            print(f"\n\nvalidation epoch: {epoch},\t"\
                f"avg_loss validation: {train_val['avg_loss']:1.4f},\t"\
                f"regression loss validation: {train_val['avg_reg_loss']:1.4f},\t"\
                f"classification loss validation: {train_val['avg_class_loss']:1.4f}"\
                f"mean error: {train_val['mean_err'] : 1.4f},\t"\
                f"max error: {train_val['max_err']:1.4f}\n\n")
            model_setup.save(epoch)

    
    model_setup.save(epoch)
    

def main():
    args = get_arguments()
    if args.resume_from_model:
        model_path = args.resume_from_model
    elif args.resume:
        model_path = get_latest_model(args)
    else:
        model_path = None
    train(model_path)


if __name__=="__main__":
    main()