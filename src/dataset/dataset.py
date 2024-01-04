import json
import os
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader


class TrainDataset:
    def __init__(self, dataset_dir, segment=None):
        """
        TrainDataset:
            dataset_dir: directory containing both clean.json and noisy.json.
            segment: segment of every audio.
        """
        self.dataset_dir = dataset_dir
        self.segment = segment
        self._gen_data_list()

    def _gen_data_list(self):
        self.sound_list = []
        self.radar_list = []
        for file in os.listdir(os.path.join(self.dataset_dir, 'sound')):
            self.sound_list.append(os.path.join(self.dataset_dir, 'sound', file))
            self.radar_list.append(os.path.join(self.dataset_dir, 'radar', file))

    def __getitem__(self, index):
        sound_file = self.sound_list[index]
        radar_file = self.radar_list[index]

        sound = torch.tensor(np.load(sound_file), dtype = torch.float32)
        radar = torch.tensor(np.load(radar_file), dtype = torch.float32)
        min_length = min(sound.shape[-1], radar.shape[-1])
        sound = sound[:, :min_length]
        radar = radar[:, :min_length]
        file_length = min_length
        if self.segment is None:
            return sound, radar
        
        #如果长度小于段长度，填充0
        if file_length < self.segment:
            sound_out = F.pad(sound, (0, self.segment - file_length))
            radar_out = F.pad(radar, (0, self.segment - file_length))

        #否则截取一段
        else:
            index = random.randint(0, file_length - self.segment)
            sound_out = sound[:, index: index+self.segment]
            radar_out = radar[:, index: index+self.segment]

        return sound_out, radar_out

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