import os
import torch
import numpy as np
from tqdm         import tqdm
from collections  import defaultdict
from tensorboardX import SummaryWriter

from utils import *


class Runner():
    def __init__(self, model, loader, device, criterion=None, optimizer=None, args=None):
        self.epoch   = 0
        self.metrics = defaultdict(float)

        self.model     = model
        self.loader    = loader
        self.device    = device
        self.criterion = criterion
        self.optimizer = optimizer

        if args.train:
            self.writer = SummaryWriter(os.path.join(args.log_dir, args.title))

        if args.patience != -1 and args.max_epoch == -1:
            args.max_epoch = np.iinfo(int).max
        
        for mtr in args.monitor.keys():
            if args.monitor[mtr]['mode'] == 'min':
                args.monitor[mtr]['record'] = np.finfo(float).max
            else:
                args.monitor[mtr]['record'] = np.finfo(float).min

            args.monitor[mtr]['cnt'] = 0

        vars(self).update({ key: val for key, val in vars(args).items() 
                if key not in list(vars(self).keys()) + dir(self) })


    def save_checkpoint(self, checkpoint='ckpt.pt'):
        state_dict = {
            'epoch'    : self.epoch,
            'model'    : self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        
        os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
        torch.save(state_dict, checkpoint)


    def load_checkpoint(self, checkpoint='ckpt.pt', params_only=False):
        state_dict = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(state_dict['model'])

        if not params_only:
            self.epoch = state_dict['epoch']
            self.optimizer.load_state_dict(state_dict['optimizer'])


    def _update_callback(self, save_best_only=True):
        if not save_best_only:
            self.save_checkpoint(os.path.join(self.exp_dir, self.title, f'ckpt/epoch_{self.epoch}.pt')) 

        for mtr in self.monitor.keys():
            if (self.monitor[mtr]['mode'] == 'min' and self.metrics[mtr] < self.monitor[mtr]['record']) or \
               (self.monitor[mtr]['mode'] == 'max' and self.metrics[mtr] > self.monitor[mtr]['record']):
                
                self.monitor[mtr]['record'] = self.metrics[mtr]
                self.monitor[mtr]['cnt'] = 0
                
                self.save_checkpoint(os.path.join(self.exp_dir, self.title, f'ckpt/best_{"_".join(mtr.split("/"))}.pt')) 
                
            else:
                self.monitor[mtr]['cnt'] += 1


    def _check_early_stopping(self):
        return self.patience != -1          and \
               self.epoch >= self.min_epoch and \
               all([ info['cnt'] >= self.patience for info in self.monitor.values() ])


    def _write_to_tensorboard(self, iteration):
        for key, val in self.metrics.items():
            self.writer.add_scalar(key, val, iteration)


    @staticmethod
    def _display(phase='Train', iteration=None, **kwargs):
        disp = f'[{phase}]'

        if iteration is not None:
            disp += f" Iter {iteration}"
        
        for key, value in kwargs.items():
            if key.endswith('loss'):
                disp += f" - {'_'.join(key.split('/'))}: {value:4.3e}"
            else:
                disp += f" - {'_'.join(key.split('/'))}: {value * 100:4.2f}"
        
        print(disp, flush=True)


    def _train_step(self, batch_x, batch_y, batch_pos):
        predict, position = self.model(batch_x)

        # the fake utterances only        
        position  = position[batch_y == 0]
        batch_pos = batch_pos[batch_y == 0]

        position  = position.permute(0, 2, 1).contiguous() # [batch, frame, 2] -> [batch, 2, frame] 
        position  = position.flatten(end_dim=1)            # [batch * 2, frame]
        batch_pos = batch_pos.flatten(end_dim=1)           # [batch * 2]

        cls_loss = self.criterion(predict , batch_y)
        pos_loss = self.criterion(position, batch_pos)

        loss = cls_loss + pos_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.metrics['train/cls_loss'] += cls_loss.item()
        self.metrics['train/pos_loss'] += pos_loss.item()
        self.metrics['train/loss']     += loss.item()


    @torch.no_grad()
    def _valid_step(self, batch_x, batch_y):
        predict, _ = self.model(batch_x)
        cls_loss   = self.criterion(predict , batch_y)

        self.metrics['valid/cls_loss'] += cls_loss.item() * len(batch_x)

        return predict[:, 1]

    
    def train(self):
        while self.epoch < self.max_epoch:
            self.epoch  += 1
            self.metrics = defaultdict(float)

            self.model.train()
            for batch_x, batch_y, batch_pos in tqdm(self.loader['train']):
                batch_x   = batch_x.to(self.device)
                batch_y   = batch_y.to(self.device)
                batch_pos = batch_pos.to(self.device)
                
                self._train_step(batch_x, batch_y, batch_pos)

            self.model.eval()            
            true, pred = [], []
            for batch_x, batch_y, _ in tqdm(self.loader['valid']):
                batch_x   = batch_x.to(self.device)
                batch_y   = batch_y.to(self.device)

                true += list(batch_y.cpu().numpy())
                pred += list(self._valid_step(batch_x, batch_y).cpu().numpy())

            self.metrics['valid/eer'] = compute_eer(true, pred)

            for key, value in self.metrics.items():
                if key.endswith('loss'):
                    if key.startswith('train'):
                        self.metrics[key] = value / len(self.loader['train'])
                    else:       
                        self.metrics[key] = value / len(self.loader['valid'].dataset)

            self._display('Train', self.epoch, **self.metrics)
            
            self._write_to_tensorboard(self.epoch)
            
            self._update_callback(self.save_best_only)

            if self._check_early_stopping(): 
                break

                
    @torch.no_grad()
    def test(self, checkpoint=None):
        self.load_checkpoint(checkpoint, params_only=True)
        
        self.model.eval()
        true, pred = [], []
        for batch_x, batch_y, _ in tqdm(self.loader['test']):
            batch_x   = batch_x.to(self.device)
            batch_y   = batch_y.to(self.device)

            predict, _ = self.model(batch_x)
            predict = predict[:, 1]

            true += list(batch_y.cpu().numpy())
            pred += list(predict.cpu().numpy())

        self.metrics['test/eer'] = compute_eer(true, pred)
        
        self._display('Test', **self.metrics)


    @torch.no_grad()
    def evaluate(self, checkpoint=None, outfile=None):
        self.load_checkpoint(checkpoint, params_only=True)
        if outfile is None:
            outfile = os.path.join(self.exp_dir, self.title, 'scores.txt')
        
        self.model.eval()
        for i, (batch_x, names) in enumerate(tqdm(self.loader['eval'])):
            batch_x = batch_x.to(self.device)
            predict , _ = self.model(batch_x)

            predict = predict[:, 1].cpu().numpy()

            if i == 0:
                os.makedirs(os.path.dirname(outfile), exist_ok=True)
                fout = open(outfile, 'w')
            else:
                fout = open(outfile, 'a')
                
            for name, pred in zip(names, predict):
                fout.write(f'{name} {pred}\n')
                
            fout.close()