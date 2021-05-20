import torchvision

from models.simple_cnn import SimpleCNN, SimpleCNN5Conv


def get_model(model_str):
    if model_str == "scnn":
        return SimpleCNN
    elif model_str == "scnn5":
        return SimpleCNN5Conv


def get_transform(trans_str, trans_kwargs):
    if trans_str == "r":
        return torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)),
                                               torchvision.transforms.ToTensor()])
    elif trans_str == "rn":
        return torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)),
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize([0.4047, 0.3870, 0.5299],
                                                                                [0.0110**(1/2), 0.0096**(1/2), 0.0165**(1/2)])])
    elif trans_str == "rl":
        cj_coeff = trans_kwargs['cj_coeff']
        return torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)),
                                               torchvision.transforms.ColorJitter(brightness=cj_coeff, contrast=cj_coeff, saturation=cj_coeff, hue=cj_coeff),
                                               torchvision.transforms.ToTensor()])
    elif trans_str == "rln":
        # mean: tensor([0.4258, 0.4127, 0.5183])
        # variance: tensor([0.0178, 0.0165, 0.0207])
        return torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)),
                                               torchvision.transforms.ColorJitter(brightness=.1, contrast=.1, saturation=.1),
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize([0.4258, 0.4127, 0.5183], [0.0178**(1/2), 0.0165**(1/2), 0.0207**(1/2)])])
    else:
        raise NotImplementedError


from torch.utils.data import Subset
import numpy as np

def split_datasets_based_on_takes(dataset, take_percentages_list, seed=0):
    """
    Splits dataset based on takes
    :param dataset:
    :return:
    """
    assert len(take_percentages_list) == 3
    assert np.sum(take_percentages_list) == 1

    # get all unique takes
    dataset.data_df['date_take'] = dataset.data_df['date'] + "_" + dataset.data_df['take']

    # split the takes according to the percentages
    dt_uniques = dataset.data_df['date_take'].unique()

    a = int(len(dt_uniques) * take_percentages_list[0])
    b = int(len(dt_uniques) * take_percentages_list[1])
    c = len(dt_uniques) - a - b
    lengths = [a, b, c]

    rng = np.random.default_rng(seed)
    # for each set of takes, get the corresponding samples indices
    rng.shuffle(dt_uniques)

    def flatten_list(list_of_lists):
        flat_list = [item for sublist in list_of_lists for item in sublist]
        return flat_list

    train_indices_list = flatten_list([dataset.data_df.index[dataset.data_df['date_take'] == dt].to_list() for dt in dt_uniques[:a]])
    val_indices_list = flatten_list([dataset.data_df.index[dataset.data_df['date_take'] == dt].to_list() for dt in dt_uniques[a:a+b]])
    test_indices_list = flatten_list([dataset.data_df.index[dataset.data_df['date_take'] == dt].to_list() for dt in dt_uniques[a+b:]])

    # for each set of indices, make a subset
    return [Subset(dataset, train_indices_list),
            Subset(dataset, val_indices_list),
            Subset(dataset, test_indices_list)]