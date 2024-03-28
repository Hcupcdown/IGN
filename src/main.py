import warnings

warnings.filterwarnings('ignore')
import os

from apps.inference import *
from apps.train import *
from config import *
from dataset import build_dataloader
from models import ConditionUNet, UNet
from utils import *


def main():

    args = get_config()
    args = args_dict(args)
    print(args.ex_name)
    print(vars(args))

    seed_init(1234)


    if args.action == 'train':

        data_loader = build_dataloader(args)
        model = ConditionUNet() if args.network == 'condition' else UNet()
        model_copy = ConditionUNet() if args.network == 'condition' else UNet()
        trainer = IGNTrainer(model_copy, model, data_loader, args)

        trainer.train()
    elif args.action == 'test':
        model = ConditionUNet() if args.network == 'condition' else UNet()
        inferencer = Inferencer(model, args)
        input_patten = r"E:\LibriMix\Libri2Mix\wav8k\max"
        input_dirs = ["train-100/s1","train-100/s2"]
        for input_dir in input_dirs:
            inferencer.interfere(os.path.join(input_patten, input_dir), os.path.join(input_patten, input_dir + "_radar"))
        # inferencer.interfere(r"F:\LibriMix\Libri2Mix\wav8k\max\train-100\s1",
        #                      r"F:\LibriMix\Libri2Mix\wav8k\max\train-100\s1_radar")

if __name__ == "__main__":
    main()
