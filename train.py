import pandas as pd
import torch.optim.optimizer
import subprocess
from dataloader import HSAFingertipDataset
import torchvision
from torch.utils.data import DataLoader
import wandb
from models.simple_cnn import SimpleCNN
import torch.nn.functional as F

pd.set_option('display.max_colwidth', None)

data_working_dir = "/home/richard/Dropbox (MIT)/6.869 Project/data"
# TODO apply transforms
dataset = HSAFingertipDataset(data_working_dir, transform=torchvision.transforms.Resize((224, 224)))

dataloader = DataLoader(dataset, batch_size=16,
                        shuffle=True, num_workers=0)

model = SimpleCNN()
optimizer = torch.optim.Adam(model.parameters())
wandb.init(project="6869_final_project")

# this is where you log hyperparameters
# for now, just make sure the git ID and string of the model class are logged
# make a commit each time there is a new model
# you can also log model parameters such as num_layers, or learning rate, in the config

git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode(("utf-8")).split("\n")[0]

wandb.config['git_id'] = git_hash
wandb.config['model_class'] = model.__module__ + model.__class__.__name__
wandb.config['optimizer'] = optimizer.__module__ + optimizer.__class__.__name__

for batch_idx, sample_batched in enumerate(dataloader):
    position_predictions = model(sample_batched['image'])

    input_ = position_predictions
    target = sample_batched['label']['pos'].float()

    assert input_.shape == target.shape

    loss = F.mse_loss(input_, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"ln24: Training loss {loss}")

    wandb.log(dict(loss=loss))