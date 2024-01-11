import warnings

warnings.filterwarnings('ignore')

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
        if args.network == 'condition':
            trainer = ConditionIGNTrainer(model_copy, model, data_loader, args)
        else:
            trainer = IGNTrainer(model_copy, model, data_loader, args)

        trainer.train()
    elif args.action == 'test':
        model = ConditionUNet() if args.network == 'condition' else UNet()
        inferencer = Inferencer(model, args)
        inferencer.interfere(r"F:\LibriMix\Libri2Mix\wav8k\max\test\s1",
                             r"e:\radar_sound_dataset\test\mn_radar")

if __name__ == "__main__":
    main()
    