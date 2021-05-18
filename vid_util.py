import cv2
print(cv2.__version__)
import os
from pathlib import Path
import sys
import pandas as pd


thirty_fps_sampling_interval = 1000/30

#example command
# extractImages("/home/richard/Dropbox (MIT)/6.869 Project/data/2021-04-30/take1-processed.mp4", "/home/richard/Dropbox (MIT)/6.869 Project/data/2021-04-30/take1-processed-images/")
def extractImages(pathIn, pathOut, samplingIntervalInMs=thirty_fps_sampling_interval):
    # adapted from https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames/49011190#49011190
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success, image = vidcap.read()
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*samplingIntervalInMs))    # added this line
        success,image = vidcap.read()
        print ('Read a new frame: ', success)
        print("pathout")
        print(pathOut)
        try:
            cv2.imwrite(pathOut + "frame%d.jpg" % count, image)     # save frame as JPEG file
        except:
            return
        count = count + 1


def read_all_images_and_labels(data_working_dir):
    assert data_working_dir[-1] != "/"
    date_dirs = os.listdir(data_working_dir)

    # create new dataframe
    column_names = ["filepath", "date", "take", "posX", "posY", "posZ", "quatX", "quatY", "quatZ", "quatW"]
    df = pd.DataFrame(columns=column_names)

    for date_dir in date_dirs:
        take_dirs = os.listdir(str(Path(data_working_dir) / date_dir))

        # get all unique takedirs
        for take_dir in take_dirs:
            if not "take" in take_dir:
                continue
            print(f"Appending take.. {date_dir}-{take_dir}")

            take_dir_fp = f"{data_working_dir}/{date_dir}/{take_dir}"
            # load list of labels first
            csv_path = f"{take_dir_fp}/{take_dir}-processed.csv"


            # quatW is the real part of the quaternion
            labels_df = pd.read_csv(csv_path, names=column_names[3:])

            print(f"Found {len(labels_df.index)} csv rows")


            images_out_dir = f"{take_dir_fp}/images/"
            file_dir = next(os.walk(images_out_dir))[0]
            filenames = next(os.walk(images_out_dir))[2]

            files_list = sorted([file_dir + file for file in filenames],
                                key=lambda filename: int(filename.split("frame")[1].split(".jpg")[0]))
            print(f"Found {len(files_list)} files")

            date_list = [date_dir for _ in range(len(filenames))]
            take_list = [take_dir for _ in range(len(filenames))]

            image_df = pd.DataFrame(zip(files_list, date_list, take_list),
                                  columns=column_names[:3])

            assert len(labels_df.index) == len(image_df.index), (len(labels_df.index), len(image_df.index))

            # merge columns of image and labels df
            # axis=1 ensures we horizontally stack the different sets of column
            # -> https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html
            merged_df = pd.concat([image_df, labels_df.reset_index(drop=True)], axis=1)

            df = df.append(merged_df, ignore_index=True)
    return df


def iterate_over_mp4s(data_working_dir):
    assert data_working_dir[-1] != "/"
    # iterate over mp4s and generate folders of temporally-subsampled images

    date_dirs = os.listdir(data_working_dir)

    for date_dir in date_dirs:
        # each dir is a date
        take_dirs = os.listdir(str(Path(data_working_dir) / date_dir))

        # get all unique takedirs
        for take_dir in take_dirs:
            csv_name = take_dir+"-processed.csv"
            mp4_name = take_dir + "-processed.mp4"

            images_out_dir = f"{data_working_dir}/{date_dir}/{take_dir}/images/"
            Path(images_out_dir).mkdir(parents=True, exist_ok=True)
            print("Extracting...")
            print(f"{date_dir}/{take_dir}/{mp4_name}")
            extractImages(f"{data_working_dir}/{date_dir}/{take_dir}/{mp4_name}", images_out_dir)


if __name__ == '__main__':
    # RUNNING THIS REGENERATES ALL IMAGES FROM MP4
    # e.g. /home/richard/Dropbox (MIT)/6.869 Project/data
    iterate_over_mp4s(sys.argv[1])