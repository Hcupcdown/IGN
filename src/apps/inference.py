import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm

from dataset import *
from utils import *
from utils.log import Log

from .evaluation import *


class Inferencer():
    def __init__(self, model, args):

        self.args = args
        self.model = model.to(args.device)

        self.n_fft = 1024
        self.hop = 12
        self.device = args.device
        self.resampler = torchaudio.transforms.Resample(8000, 1017).to(self.device)
        # checkpoint
        if args.checkpoint:
            self._load_checkpoint()

    def _load_checkpoint(self):
        print("------load checkpoint---------")
        checkpoint = torch.load(self.args.model_path)
        self.model.load_state_dict(checkpoint['state_dict'])

    def process_data(self, clean):
        clean = clean.to(self.args.device)
        clean = self.resampler(clean)
        clean_spec = torch.stft(
            clean,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(self.device),
            onesided=True,
            return_complex=True
        )
        clean_mag = torch.abs(clean_spec)
        clean_mag = clean_mag / (clean_mag.std(dim=-1, keepdim=True) + 1e-8)
        clean_mag = clean_mag.unsqueeze(1)
        return clean_mag

  
    def interfere(self, clean_dir, output_dir):
        file_list = os.listdir(clean_dir)
        os.makedirs(output_dir, exist_ok=True)

        for i, file in enumerate(tqdm(file_list)):
            file_path = os.path.join(clean_dir, file)
            radar_file = os.path.join(clean_dir.replace('clean', 'radar'), file.replace('.wav', '.npy'))
            radar = np.load(radar_file)
            radar = torch.tensor(radar).to(self.args.device)
            radar = radar / (radar.std(dim=-1, keepdim=True) + 1e-8)
            radar = radar.detach().cpu().numpy()
            clean, _ = torchaudio.load(file_path)

            # torch.cuda.empty_cache()
            clean_mag = self.process_data(clean)

            est_radar = self.model(clean_mag)
            est_radar = est_radar.squeeze(1)
            est_radar = est_radar.detach().cpu().numpy()
            plt.subplot(2,1,1)
            plt.imshow(est_radar[0])
            plt.subplot(2,1,2)
            plt.imshow(radar)
            plt.savefig(os.path.join(output_dir, file.replace('.wav', '.png')))
            np.save(os.path.join(output_dir, file.replace('.wav', '.npy')), est_radar)