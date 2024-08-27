from torch.utils.data import DataLoader
from ImageDataset import ImageDataset

def set_dataset(csv_file, part, bs, data_set, num_workers, test):

    data = ImageDataset(
        csv_file=csv_file,
        part=part,
        img_dir=data_set, test=test)

    if test:
        shuffle = False
    else:
        shuffle = True

    loader = DataLoader(data, batch_size=bs, shuffle=shuffle, pin_memory=True, num_workers=num_workers)

    return loader
