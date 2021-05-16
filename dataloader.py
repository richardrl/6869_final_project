from __future__ import print_function, division
import numpy as np
from torch.utils.data import Dataset, DataLoader
from vid_util import *
import PIL

# IGNORE: requires installing nvidia-dali-cuda100
#   pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/ nvidia-dali-cuda100
data_working_dir = "/home/richard/Dropbox (MIT)/6.869 Project/data"

# get filepaths of all images by traversing file structure
# associate each image also with the correpsonding take
from vid_util import read_all_images_and_labels

class HSAFingertipDataset(Dataset):
    def __init__(self, data_dir, ):
        self.data_dir = data_dir

        # pandas dataframe containing filelists, dates, and takes
        self.data_df = read_all_images_and_labels(data_dir)

    def __len__(self):
        # returns number of rows in dataframe
        index = self.data_df.index
        return len(index)

    def __getitem__(self, item):
        # load image from filepath
        df_row = self.data_df.loc[item]
        image_filepath = df_row['filepath']
        image = PIL.Image.open(image_filepath)
        sample = np.array(image)

        pos= np.array([df_row['posX'],
                       df_row['posY'],
                       df_row['posZ']])
        quat = np.array([df_row['quatX'],
                         df_row['quatY'],
                         df_row['quatZ'],
                         df_row['quatW']])
        label = dict(pos=pos,
                     quat=quat)
        return dict(sample=sample,
                    label=label)

pd.set_option('display.max_colwidth', None)


# TODO apply transforms
dataset = HSAFingertipDataset(data_working_dir)

dataloader = DataLoader(dataset, batch_size=16,
                        shuffle=True, num_workers=0)
for i_batch, sample_batched in enumerate(dataloader):
    print(sample_batched.keys())
