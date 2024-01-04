import warnings

warnings.filterwarnings('ignore')

from apps.inference import *
from apps.train import *
from config import *
from dataset import build_dataloader
from models import UNet
from utils import *


def main():

    args = get_config()
    args = args_dict(args)
    print(args.ex_name)
    print(vars(args))

    seed_init(1234)


    if args.action == 'train':

        data_loader = build_dataloader(args)

        if args.network == 'VAE':
            model = VQVAE(**args.VAE)
            trainer = VAETrainer(model, data_loader, args)
        else:
            model = UNet()
            model_copy = UNet()
            trainer = IGNTrainer(model_copy, model, data_loader, args)

        trainer.train()


if __name__ == "__main__":
    main()
    