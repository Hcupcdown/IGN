import torch
import torch.nn as nn
import torchvision.transforms as T
from tqdm import tqdm

from dataset import *
from utils import *
from utils.log import Log


class Trainer():
    def __init__(self, model, data, args):

        self.args = args
        self.model = model.to(args.device)
        self.device = args.device
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=args.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    mode="min",
                                                                    patience=2,
                                                                    verbose=True,
                                                                    factor=0.5)
        self.train_loader = data['train']
        self.log = Log(args)

        self.device = args.device
        # checkpoint
        if args.checkpoint:
            self._load_checkpoint()

    def _load_checkpoint(self):

        print("------load checkpoint---------")
        checkpoint = torch.load(self.args.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        
    def train(self):
        
        best_train_loss = 1e5
        for epoch in range(self.args.epoch):
            torch.cuda.empty_cache()
            self.model.train()
            train_loss = self._run_epoch(self.train_loader, epoch=epoch, valid=False)
            checkpoint = {'loss': train_loss,
                          'state_dict': self.model.state_dict(),
                          'optimizer': self.optimizer.state_dict()}
            self.log.save(checkpoint, "temp_train.pth")

            if best_train_loss > train_loss:
                best_train_loss = train_loss

                self.log.save(checkpoint, "best_train.pth")
    
    def _run_epoch(self, data_loader, epoch, valid=False):

        total_loss = 0

        for i, data in enumerate(tqdm(data_loader)):
            loss, imgs = self._run_batch(data)

            if not valid:
                self.log.add_scalar(cate="train",
                                    global_step = epoch*len(data_loader) + i,
                                    **loss
                                    )
                if i == 0:
                    self.log.add_plot_image(cate="train",
                                           global_step = epoch,
                                           **imgs
                                           )

                self.optimizer.zero_grad()
                loss["loss"].backward()
                self.optimizer.step()
                total_loss += loss["loss"].item()

        self.scheduler.step(total_loss/(i+1))

        return total_loss/(i+1)

class IGNTrainer(Trainer):

    def __init__(self, model_copy ,*args):
    
        super().__init__(*args)
        self.model_copy = model_copy.to(self.device)
        self.mse = nn.MSELoss()

    def _run_batch(self, data):
    
        radar = data["radar"]
        sound = data["sound"]
        x = radar.to(self.device)
        z = sound.to(self.device)
        self.model_copy.load_state_dict(self.model.state_dict())
        
        fx = self.model(x)
        fz = self.model(z)
        f_z = fz.detach()
        ff_z = self.model(f_z)
        f_fz = self.model_copy(fz)
        loss_rec = self.mse(fx, x)
        loss_idem = self.mse(f_fz,fz)
        loss_tight = -self.mse(ff_z, f_z)
        loss_me = self.mse(fz, x)
        loss = loss_rec + loss_idem + loss_tight * 0.1

        return {
            "loss_rec":loss_rec,
            "loss_idem":loss_idem,
            "loss_tight":loss_tight,
            "loss_me":loss_me,
            "loss":loss
        }, {
            "x":x,
            "fx":fx,
            "z":z,
            "fz":fz,
            "f_z":f_z,
            "ff_z":ff_z
        }

class ConditionIGNTrainer(Trainer):

    def __init__(self, model_copy ,*args):
    
        super().__init__(*args)
        self.model_copy = model_copy.to(self.device)
        self.mse = nn.MSELoss()

    def _run_batch(self, data):
    
        radar = data["radar"]
        sound = data["sound"]
        x = radar.to(self.device)
        condition = sound.to(self.device)
        z = torch.randn_like(x)
        self.model_copy.load_state_dict(self.model.state_dict())
        
        fx = self.model(x, condition)
        fz = self.model(z, condition)
        f_z = fz.detach()
        ff_z = self.model(f_z, condition)
        f_fz = self.model_copy(fz, condition)
        loss_rec = self.mse(fx, x)
        loss_idem = self.mse(f_fz,fz)
        loss_tight = -self.mse(ff_z, f_z)
        loss_me = self.mse(fz, x)
        loss = loss_rec + loss_idem + loss_tight * 0.1

        return {
            "loss_rec":loss_rec,
            "loss_idem":loss_idem,
            "loss_tight":loss_tight,
            "loss_me":loss_me,
            "loss":loss
        }, {
            "x":x,
            "fx":fx,
            "condition":condition,
            "fz":fz,
            "f_z":f_z,
            "ff_z":ff_z
        }