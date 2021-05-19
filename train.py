import pandas as pd
import torch.optim.optimizer
import subprocess
from dataset import HSAFingertipDataset, calculate_image_population_stats
import torchvision
from torch.utils.data import DataLoader, random_split
import wandb
from models.simple_cnn import *
import torch.nn.functional as F
import os
import tqdm
import argparse
import numpy as np


DEVICE = torch.device("cuda:1")



parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--checkpoint_freq', type=int, help="Checkpoint frequency in epochs", default=10)
parser.add_argument('--trans_str', type=str, default="r")
parser.add_argument('--calculate_stats', action='store_true')

args = parser.parse_args()
pd.set_option('display.max_colwidth', None)


def get_transform(trans_str):
    if trans_str == "r":
        return torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)),
                                               torchvision.transforms.ToTensor()])
    elif trans_str == "rn":
        return torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)),
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize([0.4047, 0.3870, 0.5299],
                                                                                [0.0110**(1/2), 0.0096**(1/2), 0.0165**(1/2)])])
    elif trans_str == "rl":
        return torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)),
                                               torchvision.transforms.ColorJitter(brightness=.1, contrast=.1, saturation=.1),
                                               torchvision.transforms.ToTensor()])
    elif trans_str == "rln":
        raise NotImplementedError
        return torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)),
                                               torchvision.transforms.ColorJitter(brightness=.1, contrast=.1, saturation=.1),
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize])
    else:
        raise NotImplementedError

def get_model(model_str):
    if model_str == "scnn":
        return SimpleCNN
    elif model_str == "scnn5":
        return SimpleCNN5Conv

data_working_dir = "/home/richard/data/hsa_data"
# TODO apply transforms
dataset = HSAFingertipDataset(data_working_dir, transform=get_transform(args.trans_str))

if args.calculate_stats:
    mean, variance = calculate_image_population_stats(dataset)

train_len = int(len(dataset)*.6)
val_len = int(len(dataset)*.2)
train_d, val_d, test_d = random_split(dataset, [train_len, val_len, len(dataset) - train_len - val_len],
                                      generator=torch.Generator().manual_seed(0))
datasets = dict(train=train_d,
                val=val_d,
                test=test_d)

train_dataloader = DataLoader(train_d, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
val_dataloader = DataLoader(val_d, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
test_dataloader = DataLoader(test_d, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
dataloaders = dict(train=train_dataloader,
                   val=val_dataloader,
                   test=test_dataloader)

model_kwargs = dict(batchnorm=True)
model = SimpleCNN5Conv()
model = model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters())
wandb.init(project="richards_runs",
           entity="6869_final_project_team")

# this is where you log hyperparameters
# for now, just make sure the git ID and string of the model class are logged
# make a commit each time there is a new model
# you can also log model parameters such as num_layers, or learning rate, in the config

git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode(("utf-8")).split("\n")[0]

wandb.config['git_id'] = git_hash
wandb.config['model_class'] = model.__module__ + model.__class__.__name__
wandb.config['optimizer'] = optimizer.__module__ + optimizer.__class__.__name__
wandb.config['trans_str'] = args.trans_str
wandb.config['model_kwargs'] = model_kwargs

wandb.save(os.path.join(wandb.run.dir, "checkpoint*"))

for epoch in range(args.num_epochs):
    print(f"wandb log directory name {wandb.run.dir}")
    for phase in ['train', 'val']:
        tq_obj = tqdm.tqdm(dataloaders[phase])
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode


        total_samples = len(datasets[phase])
        total_sum_vec = np.array([0., 0., 0.])[None, :]
        for batch_idx, sample_batched in enumerate(tq_obj):
            tq_obj.set_description(f"{phase} epoch {batch_idx}")
            position_predictions = model(sample_batched['image'].to(DEVICE))

            input_ = position_predictions
            target = sample_batched['label']['pos'].float().to(DEVICE)

            assert input_.shape == target.shape

            loss = F.mse_loss(input_, target, reduction='none')

            summed_loss = loss.data.cpu().numpy().sum(axis=0)


            total_sum_vec += summed_loss

            tq_obj.set_description(f"{phase} minibatch loss {summed_loss / sample_batched['image'].shape[0]}")

            optimizer.zero_grad()

            if phase == 'train':
                loss.mean().backward()
                optimizer.step()


        wandb.log({f"{phase} aggregate X loss": total_sum_vec[0][0].squeeze() / total_samples})
        wandb.log({f"{phase} aggregate Y loss": total_sum_vec[0][1].squeeze() / total_samples})
        wandb.log({f"{phase} aggregate Z loss": total_sum_vec[0][2].squeeze() / total_samples})

        print(f"ln62: {phase} aggregate loss {total_sum_vec / total_samples}")
        print("\n")

    if epoch % args.checkpoint_freq == 0:
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, f"checkpoint{epoch}"))

