import pandas as pd
import subprocess

import misc_util
from dataset import HSAFingertipDataset, calculate_image_population_stats
from torch.utils.data import DataLoader, random_split
import wandb
from models.simple_cnn import *
import torch.nn.functional as F
import os
import tqdm
import argparse
import numpy as np
from misc_util import *





parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str)
parser.add_argument('device', type=int)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--checkpoint_freq', type=int, help="Checkpoint frequency in epochs", default=10)
parser.add_argument('--trans_str', type=str, default="r")
parser.add_argument('--model_str', type=str, default="scnn")
parser.add_argument('--calculate_stats', action='store_true')
parser.add_argument('--split_based_on_takes', action='store_true')

args = parser.parse_args()
pd.set_option('display.max_colwidth', None)

DEVICE = torch.device(f"cuda:{args.device}")

data_working_dir = args.data_dir
# TODO apply transforms
cj_kwargs = dict(cj_coeff=.4)
dataset = HSAFingertipDataset(data_working_dir, transform=get_transform(args.trans_str, cj_kwargs))

if args.calculate_stats:
    mean, variance = calculate_image_population_stats(dataset)

train_len = int(len(dataset)*.6)
val_len = int(len(dataset)*.2)

if args.split_based_on_takes:
    train_d, val_d, test_d = misc_util.split_datasets_based_on_takes(dataset, [.6, .2, .2], seed=0)
else:
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
model = get_model(args.model_str)(**model_kwargs)
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
wandb.config['model_class'] = model.__module__ + "." + model.__class__.__name__
wandb.config['optimizer'] = optimizer.__module__ + "." + optimizer.__class__.__name__
wandb.config['trans_str'] = args.trans_str
wandb.config['model_kwargs'] = model_kwargs
wandb.config['split_based_on_takes'] = args.split_based_on_takes
wandb.config['cj_kwargs'] = cj_kwargs

wandb.save(os.path.join(wandb.run.dir, "checkpoint*"))

best_val_loss = float("inf")

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


        wandb.log({f"{phase} aggregate X loss": total_sum_vec[0][0].squeeze() / total_samples}, step=epoch)
        wandb.log({f"{phase} aggregate Y loss": total_sum_vec[0][1].squeeze() / total_samples}, step=epoch)
        wandb.log({f"{phase} aggregate Z loss": total_sum_vec[0][2].squeeze() / total_samples}, step=epoch)

        if phase == "val":
            current_val_loss = total_sum_vec[0][1].squeeze() / total_samples
            if best_val_loss > current_val_loss and epoch > 10:
                best_val_loss = current_val_loss
                torch.save(model.state_dict(), os.path.join(wandb.run.dir, f"checkpoint{epoch}"))

        print(f"ln62: {phase} aggregate loss {total_sum_vec / total_samples}")
        print("\n")

    if epoch % args.checkpoint_freq == 0:
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, f"checkpoint{epoch}"))
    wandb.log({f"epoch": epoch}, step=epoch)
