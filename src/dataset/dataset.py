import json
import os
import random

import numpy as np
import torch
import torchaudio
from torch.nn import functional as F
from torch.utils.data import DataLoader


class TrainDataset:
    def __init__(self,
                 dataset_dir,
                 segment=None):
        """
        TrainDataset:
            dataset_dir: directory containing both clean.json and noisy.json.
            segment: segment of every audio.
        """
        self.dataset_dir = dataset_dir
        self.segment = segment
        self.downsampler = torchaudio.transforms.Resample(8000, 1200)
        self._gen_data_list()

    def _gen_data_list(self):
        self.sound_list = []
        self.radar_list = []
        for person_id in os.listdir(os.path.join(self.dataset_dir)):
            for sample in os.listdir(os.path.join(self.dataset_dir, person_id, "audio")):
                self.sound_list.append(os.path.join(self.dataset_dir, person_id,"audio", sample))
                self.radar_list.append(os.path.join(self.dataset_dir, person_id, "radar", sample))

    def __getitem__(self, index):
        sound_file = self.sound_list[index]
        radar_file = self.radar_list[index]

        sound, sr = torchaudio.load(sound_file)
        radar = torch.tensor(np.load(radar_file), dtype = torch.float32)
        assert sr == 8000, "Sample rate is not 8000"
        sound = self.downsampler(sound)

        return sound, radar

    def __len__(self):
        return len(self.sound_list)

def collate_fn(batch):
    sound = [item[0] for item in batch]
    radar = [item[1] for item in batch]
    sound = torch.stack(sound, dim=0)
    radar = torch.stack(radar, dim=0)
    return {"sound":sound, "radar":radar}

def build_dataloader(args):
    train_dataset = TrainDataset(args.dataset['train'],
                                 segment=args.setting['segment'])
    train_loader  = DataLoader(train_dataset,
                               batch_size=args.batch_size,
                               shuffle=True,
                               num_workers=args.num_worker,
                               collate_fn=collate_fn)
    val_dataset = TrainDataset(args.dataset['val'])
    val_loader  = DataLoader(val_dataset,
                               batch_size=1,
                               shuffle=False,
                               num_workers=args.num_worker,
                               collate_fn=collate_fn)
    
    return {'train':train_loader, 'val':val_loader}