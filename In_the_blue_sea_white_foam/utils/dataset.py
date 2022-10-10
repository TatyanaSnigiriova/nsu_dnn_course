import random
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt


class Multiclass_Dataset(Dataset):
    def __init__(self, csv_file_patch, imgs_transforms=transforms.ToTensor()):
        """
        Args:
            csv_file_patch (string): Path to the csv file with annotations.
            imgs_transforms (callable, optional): Optional transforms to be applied to each sample.
        """
        data_df = pd.read_csv(csv_file_patch, index_col=None)
        self.patchs = data_df["file_path"].to_numpy()
        self.labels = data_df["label"].to_numpy()

        del data_df
        self.imgs_transforms = imgs_transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_pil = Image.open(self.patchs[idx])
        image_tensor = self.imgs_transforms(image_pil)
        label = self.labels[idx]
        return {"image": image_tensor, "label": label}


class RandomRotationBySpace:
    """Rotate by one of the given angles."""
    torchvision.transforms.InterpolationMode
    def __init__(self, angles, interpolation=transforms.InterpolationMode.NEAREST, expand=False):
        self.angles = angles
        self.interpolation = interpolation
        self.expand = expand

    def __call__(self, img):
        return transforms.functional.rotate(
            img,
            angle=random.choice(self.angles),
            interpolation=self.interpolation,
            expand=self.expand
        )


def show_images_batch(labels_decoder, sample_batch):
    if type(labels_decoder) is not pd.Series:
        labels_decoder_s = pd.Series(labels_decoder)
    else:
        labels_decoder_s = labels_decoder

    images_batch, labels_batch = \
        sample_batch['image'], sample_batch['label']
    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    print("labels:", labels_decoder_s.iloc[labels_batch].values)
    plt.title('Batch from dataloader')


def show_some_data_samples(dataloader, labels_decoder, nbatches=4):
    assert nbatches > 0
    nbatches -= 1
    for ibatch, sample_batch in enumerate(dataloader):
        print(ibatch, sample_batch['image'].size(), sample_batch['label'].size())
        plt.figure(figsize=(25, 40))
        show_images_batch(labels_decoder, sample_batch)
        plt.axis('off')

        if ibatch == nbatches:
            plt.show()
            break


def get_label_for_enc_label_pred(labels_decoder, enc_label_pred_t):
    idx = ((enc_label_pred_t == enc_label_pred_t.max()).nonzero(as_tuple=True)[0])
    if type(labels_decoder) is not pd.Series:
        return labels_decoder[idx]
    else:
        return labels_decoder.iloc[idx].values[0]


def get_labels_for_enc_labels_batch_pred(labels_decoder, enc_labels_pred_t):
    if type(labels_decoder) is not pd.Series:
        labels_decoder_s = pd.Series(labels_decoder)
    else:
        labels_decoder_s = labels_decoder

    idxs = np.zeros(len(enc_labels_pred_t))
    for idx, enc_label_t in enumerate(enc_labels_pred_t):
        idxs[idx] = ((enc_label_t == enc_label_t.max()).nonzero(as_tuple=True)[0])
    return labels_decoder_s.iloc[idxs].values
