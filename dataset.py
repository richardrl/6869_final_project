from __future__ import print_function, division
import numpy as np
import tqdm
from torch.utils.data import Dataset
from PIL import Image
from vid_util import read_all_images_and_labels
import cv2


class HSAFingertipDataset(Dataset):
    def __init__(self, data_dir, opencv_preprocessing=False, transform=None, vertically_stack_images=True):
        self.data_dir = data_dir

        # pandas dataframe containing filelists, dates, and takes
        self.data_df = read_all_images_and_labels(data_dir)
        self.transform = transform
        self.opencv_preprocessing = opencv_preprocessing
        self.vertically_stack_images = vertically_stack_images

    def __len__(self):
        # returns number of rows in dataframe
        index = self.data_df.index
        return len(index)

    def __getitem__(self, item):
        # load image from filepath
        df_row = self.data_df.loc[item]
        image_filepath = df_row['filepath']
        image = Image.open(image_filepath)

        if self.opencv_preprocessing:
            opencv_image = np.array(image)
            opencv_image = cv2.GaussianBlur(opencv_image, (5,5), 3)
            opencv_image = cv2.Laplacian(opencv_image, cv2.CV_8U)
            opencv_image = (opencv_image / np.max(opencv_image) * 255).astype(np.uint8)
            opencv_image = cv2.GaussianBlur(opencv_image, (5, 5), 1)
            opencv_image = (opencv_image / np.max(opencv_image) * 255).astype(np.uint8)
            image = Image.fromarray(opencv_image)

        if self.vertically_stack_images:
            img_left_area = (0, 0, 480, 360)
            img_right_area = (480, 0, 960, 360)
            img_left = image.crop(img_left_area)
            img_right = image.crop(img_right_area)
            image = Image.new('RGB', (480, 720))
            image.paste(img_left, (0, 0))
            image.paste(img_right, (0, 360))

        if self.transform:
            # print("ln50")
            # print(image.size)
            image = self.transform(image)
            # print(image.shape)

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


def calculate_image_population_stats(dataset):
    from torch.utils.data import DataLoader
    import torch
    dl = DataLoader(dataset, batch_size=128,
               shuffle=True, num_workers=8)

    # not complete, seems too slow
    running_sum = torch.zeros(3)
    total_pixels = 0
    for sample_idx, batch in tqdm.tqdm(enumerate(dl)):
        assert batch['image'].shape[1] == 3, batch['image'].shape
        # [batch_size, c, h, w]
        channel_pixel_sum = batch['image'].sum((0, 2, 3))

        running_sum += channel_pixel_sum
        total_pixels += batch['image'].shape[0] * batch['image'].shape[2] * batch['image'].shape[3]

    mean = running_sum/total_pixels

    sum2 = torch.zeros(3)
    for sample_idx, batch in tqdm.tqdm(enumerate(dl)):
        # [batch_size, c, h, w] -> [batch_size, h, w, c == 3]
        sum2 += torch.sum((batch['image'].permute(0, 2, 3, 1).reshape(-1, 3) - mean)**2, dim=0)

    variance = sum2/(total_pixels- 1)
    print(f"mean: {mean}")
    print(f"variance: {variance}")
    return mean, variance