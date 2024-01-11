import argparse
import os


def get_config():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('action', type=str, default='train', help='Action') # train / test
    parser.add_argument('--network', type=str, default='condition', help='train condition or IGN') # train / test
    # dataset
    parser.add_argument('--train', type=str, default=r'D:\radar_sound_dataset\sound2radar_dataset', help='Train path')
    parser.add_argument('--val', type=str, default=r'D:\radar_sound_dataset\sound2radar_dataset', help='Val path')
    parser.add_argument('--test', type=str, default=r'D:\radar_sound_dataset\sound2radar_dataset', help='Test path')
    parser.add_argument('--segment', type=int, default=320, help='Segment')

    #vqvae
    parser.add_argument('--hidden', type=int, default=32, help='Hidden')
    parser.add_argument('--codebook_size', type=int, default=256, help='codebook size')
    parser.add_argument('--depth', type=int, default=4, help='Depth')

    #basic 
    parser.add_argument('--model_path', type=str, default='log/24-01-10-22-07-46/model/temp_train.pth', help='Model path')
    parser.add_argument('--model_name', type=str, default='best_model.pth', help='Model name') # select manner_ {small, base, large}
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--epoch', type=int, default=500, help='Epoch')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--checkpoint', type=bool, default=False, help='Checkpoint') # If you want to train with pre-trained, or resume set True

    # device 
    parser.add_argument('--device', type=str, default='cuda:0', help='Gpu device')
    parser.add_argument('--env', type=str, default='local', help='Enviornment')
    parser.add_argument('--num_worker', type=int, default=4, help='Num workers')

    # logging setting
    parser.add_argument('--logging', type=bool, default=False, help='Logging')
    parser.add_argument('--logging_cut', type=int, default=-1, help='Logging cut') # logging after the epoch of logging_cut

    arguments = parser.parse_args()

    return arguments
