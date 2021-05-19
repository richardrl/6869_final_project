import sys
from models.simple_cnn import SimpleCNN
import torch
from torch.utils.data import DataLoader, random_split
from dataset import HSAFingertipDataset
import torchvision
import tqdm
import argparse
import numpy as np

import torch.nn.functional as F

DEVICE = torch.device("cuda:1")


def evaluate(checkpoint_dir, data_working_dir):
    # load model
    model = SimpleCNN().to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_dir))
    model.eval()

    # test dataloader
    dataset = HSAFingertipDataset(data_working_dir, transform=torchvision.transforms.Resize((224, 224)))

    train_len = int(len(dataset) * .6)
    val_len = int(len(dataset) * .2)
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

    for phase in ['train', 'val', 'test']:
        tq_obj = tqdm.tqdm(dataloaders[phase])
        model.eval()  # Set model to evaluate mode

        total_samples = len(tq_obj)
        total_sum_vec = np.array([0., 0., 0.])[None, :]

        for batch_idx, sample_batched in enumerate(tq_obj):
            position_predictions = model(sample_batched['image'].to(DEVICE))

            input_ = position_predictions
            target = sample_batched['label']['pos'].float().to(DEVICE)

            assert input_.shape == target.shape

            loss = F.mse_loss(input_, target, reduction='mean')

            summed_loss = loss.data.cpu().numpy().sum(axis=0)
            total_sum_vec += summed_loss
            tq_obj.set_description(f"{phase} loss {summed_loss}")

        print(f"{phase} loss aggregate {total_sum_vec / total_samples}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_dir')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--checkpoint_freq', type=int, help="Checkpoint frequency in epochs", default=10)
    args = parser.parse_args()
    data_working_dir = "/home/richard/data/hsa_data"
    evaluate(args.checkpoint_dir, data_working_dir)