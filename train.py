import pandas as pd
import torch.optim.optimizer
import subprocess
from dataset import HSAFingertipDataset
import torchvision
from torch.utils.data import DataLoader, random_split
import wandb
from models.simple_cnn import SimpleCNN
import torch.nn.functional as F
import os
import tqdm
import argparse

DEVICE = torch.device("cuda:1")



parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--checkpoint_freq', type=int, help="Checkpoint frequency in epochs", default=10)
args = parser.parse_args()
pd.set_option('display.max_colwidth', None)

data_working_dir = "/home/richard/Dropbox (MIT)/6.869 Project/data"
# TODO apply transforms
dataset = HSAFingertipDataset(data_working_dir, transform=torchvision.transforms.Resize((224, 224)))

train_len = int(len(dataset)*.6)
val_len = int(len(dataset)*.2)
train_d, val_d, test_d = random_split(dataset, [train_len, val_len, len(dataset) - train_len - val_len],
                                      generator=torch.Generator().manual_seed(0))

train_dataloader = DataLoader(train_d, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_d, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_d, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
dataloaders = dict(train=train_dataloader,
                   val=val_dataloader,
                   test=test_dataloader)

model = SimpleCNN()
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

wandb.save(os.path.join(wandb.run.dir, "checkpoint*"))

for epoch in range(args.num_epochs):
    print(f"wandb log directory name {wandb.run.dir}")
    for phase in ['train', 'val']:
        tq_obj = tqdm.tqdm(dataloaders[phase])
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode
        for batch_idx, sample_batched in enumerate(tq_obj):
            tq_obj.set_description(f"{phase} epoch {batch_idx}")
            position_predictions = model(sample_batched['image'].to(DEVICE))

            input_ = position_predictions
            target = sample_batched['label']['pos'].float().to(DEVICE)

            assert input_.shape == target.shape

            loss = F.mse_loss(input_, target, reduction='mean')
            optimizer.zero_grad()

            if phase == 'train':
                loss.backward()
                optimizer.step()

        wandb.log({f"{phase} loss": loss})
        print(f"ln62: {phase} loss {loss}")
        print("\n")

    if epoch % args.checkpoint_freq == 0:
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, f"checkpoint{epoch}"))

