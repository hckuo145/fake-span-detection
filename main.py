import yaml
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model   import MainModel
from runner  import Runner
from dataset import AddDataset, EvalDataset


parser = argparse.ArgumentParser()
parser.add_argument('--eval'  , action='store_true', default=False)
parser.add_argument('--test'  , action='store_true', default=False)
parser.add_argument('--train' , action='store_true', default=False)
    
parser.add_argument('--seed'  , type=int, default=0)
parser.add_argument('--batch' , type=int, default=256)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--params', type=str, default=None)

parser.add_argument('--title'  , type=str, default='Untitled')
parser.add_argument('--outfile', type=str, default=None)

parser.add_argument('--model_conf', type=str, default='../config/model/SeNet.yaml')
parser.add_argument('--hyper_conf', type=str, default='../config/hyper/hyper.yaml')
args = parser.parse_args()

with open(args.hyper_conf) as conf:
    vars(args).update(yaml.load(conf, Loader=yaml.Loader))

with open(args.model_conf) as conf:
    model_args = yaml.load(conf, Loader=yaml.Loader)


np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
device = torch.device(args.device)


print(args.title, flush=True)

model = MainModel(**model_args).to(device)
n_params = sum( p.numel() for p in model.parameters() )
print(f'[Model] -# params: {n_params}', flush=True)

if args.train:
    dataset = {
        'train': AddDataset(**args.train_dataset_args),
        'valid': AddDataset(**args.valid_dataset_args)
    }
    
    loader = {
        'train': DataLoader(dataset['train'], batch_size=args.batch, num_workers=4, pin_memory=True, \
            shuffle=True, drop_last=True),
        'valid': DataLoader(dataset['valid'], batch_size=1, num_workers=4, pin_memory=True)
    }

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = getattr(optim, args.optim['name'])(model.parameters(), **args.optim['args'])

    runner = Runner(model, loader, device, criterion, optimizer, args=args)
    runner.train()

if args.test:
    dataset = {
        'test': AddDataset(**args.test_dataset_args)
    }

    loader = {
        'test': DataLoader(dataset['test'], batch_size=1, num_workers=4, pin_memory=True)
    }

    runner = Runner(model, loader, device, args=args)
    runner.test(args.params)

if args.eval:
    dataset = {
        'eval': EvalDataset(**args.eval_dataset_args)
    }

    loader = {
        'eval': DataLoader(dataset['eval'], batch_size=1, num_workers=4, pin_memory=True)
    }    

    runner = Runner(model, loader, device, args=args)
    runner.evaluate(args.params, args.outfile)