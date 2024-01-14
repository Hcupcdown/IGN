import torch
import torchaudio
from matplotlib import pyplot as plt
from tqdm import tqdm

from dataset import *
from utils import *


class Inferencer():
    def __init__(self, model, args):

        self.args = args
        self.model = model.to(args.device)

        self.device = args.device
        self.net = args.network
        # self.resampler = torchaudio.transforms.Resample(12800, 8000).to(self.device)
        self.stft_shift = torchaudio.transforms.Spectrogram(n_fft=1024,
                                                            hop_length=90,
                                                            win_length=1024,
                                                            power=1).to(self.device)
        checkpoint = torch.load(self.args.model_path)
        self.model.load_state_dict(checkpoint['state_dict'])

    def process_data(self, clean):
        clean = clean.to(self.args.device)
        clean = clean + torch.randn_like(clean) *5e-3
        # clean = self.resampler(clean)
        clean_spec = self.stft_shift(clean)
        clean_spec = clean_spec ** 0.3
        clean_spec = clean_spec / (clean_spec.max()+1e-8)

        return clean_spec

    def interfere(self, clean_dir, output_dir):
        file_list = os.listdir(clean_dir)
        os.makedirs(output_dir, exist_ok=True)
        self.model.eval()
        for i, file in enumerate(tqdm(file_list)):
            file_path = os.path.join(clean_dir, file)
            clean, sr = torchaudio.load(file_path)
            
            clean_mag = self.process_data(clean)
            # plt.subplot(121)
            # plt.imshow((clean_mag[0]).cpu().numpy())
            # plt.colorbar()

            clean_mag_list = torch.split(clean_mag, 320, dim=-1)
            # clean_mag = clean_mag.unsqueeze(0)
            est_radar_list = []
            for clean_mag in clean_mag_list:
                if self.net == 'condition':
                    z = torch.rand_like(clean_mag)
                    est_radar = self.model(z, clean_mag)
                else:
                    est_radar = self.model(clean_mag)
                est_radar_list.append(est_radar)
            est_radar = torch.cat(est_radar_list, dim=-1)
            est_radar = est_radar.squeeze(1)
            est_radar = est_radar.detach().cpu().numpy()
            
            # plt.subplot(122)
            # plt.imshow(est_radar[0])
            # plt.colorbar()
            # plt.show()

            # plt.savefig(os.path.join(output_dir, file.replace('.wav', '.png')))
            np.save(os.path.join(output_dir, file.replace('.wav', '.npy')), est_radar)