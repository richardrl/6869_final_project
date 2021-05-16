from __future__ import print_function, division
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from vid_util import read_all_images_and_labels


class HSAFingertipDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir

        # pandas dataframe containing filelists, dates, and takes
        self.data_df = read_all_images_and_labels(data_dir)
        self.transform = transform

    def __len__(self):
        # returns number of rows in dataframe
        index = self.data_df.index
        return len(index)

    def __getitem__(self, item):
        # load image from filepath
        df_row = self.data_df.loc[item]
        image_filepath = df_row['filepath']
        image = Image.open(image_filepath)

        if self.transform:
            image = self.transform(image)


        image = np.array(image)

        pos= np.array([df_row['posX'],
                       df_row['posY'],
                       df_row['posZ']])
        quat = np.array([df_row['quatX'],
                         df_row['quatY'],
                         df_row['quatZ'],
                         df_row['quatW']])
        label = dict(pos=pos,
                     quat=quat)
        return dict(image=image,
                    label=label)

