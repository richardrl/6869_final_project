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
import yaml
from misc_util import get_model, get_transform, split_datasets_based_on_takes

DEVICE = torch.device("cuda:1")


def evaluate(checkpoint_dir, data_working_dir, wandb_yaml):
    wyaml = yaml.load(open(wandb_yaml))

    # load model
    model = SimpleCNN(**wyaml['model_kwargs']['value']).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_dir))
    model.eval()

    # test dataloader
    dataset = HSAFingertipDataset(data_working_dir, transform=get_transform(wyaml['trans_str']['value']))

    train_len = int(len(dataset) * .6)
    val_len = int(len(dataset) * .2)

    if args.split_based_on_takes:
        train_d, val_d, test_d = split_datasets_based_on_takes(dataset, [.6, .2, .2], seed=0)
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

    for phase in ['test']:
        tq_obj = tqdm.tqdm(dataloaders[phase])
        model.eval()  # Set model to evaluate mode

        total_samples = len(datasets[phase])
        total_sum_vec = np.array([0., 0., 0.])[None, :]

        for batch_idx, sample_batched in enumerate(tq_obj):
            position_predictions = model(sample_batched['image'].to(DEVICE))

            input_ = position_predictions
            target = sample_batched['label']['pos'].float().to(DEVICE)

            assert input_.shape == target.shape

            loss = F.mse_loss(input_, target, reduction='none')

            summed_loss = loss.data.cpu().numpy().sum(axis=0)

            total_sum_vec += summed_loss
            tq_obj.set_description(f"{phase} loss {summed_loss}")

        print(f"{phase} loss aggregate {total_sum_vec / total_samples}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_dir', type=str)
    parser.add_argument('wandb_yaml', type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--checkpoint_freq', type=int, help="Checkpoint frequency in epochs", default=10)
    parser.add_argument('--split_based_on_takes', action='store_true')
    parser.add_argument('--num_workers', type=int, default=8)

    args = parser.parse_args()
    data_working_dir = "/home/richard/data/hsa_data"
    evaluate(args.checkpoint_dir, data_working_dir, args.wandb_yaml)