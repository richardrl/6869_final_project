import sys
from models.simple_cnn import SimpleCNN
import torch
from torch.utils.data import DataLoader, random_split
from dataset import HSAFingertipDataset
import torchvision
import tqdm
import argparse

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

    tq_obj = tqdm.tqdm(dataloaders['test'])
    model.eval()  # Set model to evaluate mode
    for batch_idx, sample_batched in enumerate(tq_obj):
        position_predictions = model(sample_batched['image'].to(DEVICE))

        input_ = position_predictions
        target = sample_batched['label']['pos'].float().to(DEVICE)

        assert input_.shape == target.shape

        loss = F.mse_loss(input_, target, reduction='mean')
        tq_obj.set_description(f"Test loss {loss}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_dir')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--checkpoint_freq', type=int, help="Checkpoint frequency in epochs", default=10)
    args = parser.parse_args()
    data_working_dir = "/home/richard/Dropbox (MIT)/6.869 Project/data"
    evaluate(args.checkpoint_dir, data_working_dir)