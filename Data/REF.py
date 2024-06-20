import os
import sys
import pickle
import random
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor, resize
from torchvision import transforms
from PIL import Image


base_folder = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
if base_folder not in sys.path:
    sys.path.append(base_folder)
if True:
    import config
    from Log.Logger import getLogger


# The file structure was reorganized as follows:
#     SuperResolution/Data/:
#         original:
#             ref:
#                 xxx.png
#                 xxx.png
#                 .
#                 .
#                 .
#                 xxx.png
#         preprocessed:
#             ref:
#                 train.pickle    imgs: List[torch.Tensor] of RGB images
#                 val.pickle      imgs: List[torch.Tensor] of RGB images
#                 test.pickle     imgs: List[torch.Tensor] of RGB images
#         predicted:
#             xxx_orig.png
#             xxx_gt.png
#             xxx_pred.png
#             .
#             .
#             .

def join_path(*args):
    return os.path.abspath(os.path.join(*args))


in_channels = 1

original_folder = join_path(base_folder, "./Data/original/ref")
preprocessed_folder = join_path(base_folder, "./Data/preprocessed/ref")
os.makedirs(preprocessed_folder, exist_ok=True)

train_path = join_path(preprocessed_folder, "./train.pickle")
val_path = join_path(preprocessed_folder, "./val.pickle")
test_path = join_path(preprocessed_folder, "./test.pickle")


train = os.listdir(join_path(original_folder, "./train"))
train = [join_path(original_folder, f"./train/{i}")
         for i in train if i.endswith(".png")]
val = os.listdir(join_path(original_folder, "./val"))
val = [join_path(original_folder, f"./val/{i}")
       for i in val if i.endswith(".png")]
test = os.listdir(join_path(original_folder, "./test"))
test = [join_path(original_folder, f"./test/{i}")
        for i in test if i.endswith(".png")]


def _preprocess(files: list[str], output_path: str, patchify: bool = False, patch_size: int = 40, stride: int = 10):
    scales = torch.tensor((1, 0.9, 0.8, 0.7)).reshape(4, 1)
    imgs = []

    print(f"Processing {output_path}")
    for file in tqdm(files):
        img = to_tensor(Image.open(file)).to(config.device)
        img.requires_grad_(False)
        if patchify:
            shapes = (torch.tensor(img.shape[-2:]) * scales).int()
            for shape in shapes:
                tmp = resize(img, shape)
                for h in range(0, shape[0]-patch_size+1, stride):
                    for w in range(0, shape[1]-patch_size+1, stride):
                        imgs.append(
                            tmp[:, h:h+patch_size, w:w+patch_size].cpu())
        else:
            imgs.append(img.cpu())

    with open(output_path, "wb") as f:
        pickle.dump(imgs, f)


class _REF_Dataset(Dataset):
    def __init__(self, path, std=config.std, transform=None, device=config.device):
        with open(path, "rb") as f:
            self.imgs = pickle.load(f)
        self.LEN = len(self.imgs)
        self.device = device
        self.std = std / 255
        self.transform = transform

    def __len__(self):
        return self.LEN

    def __getitem__(self, index):
        target = self.imgs[index].clone().to(self.device)
        noise = torch.randn(target.shape, device=self.device) * self.std
        input = target + noise
        input = 1 - F.relu(1 - F.relu(input))
        if self.transform is not None:
            tmp = torch.cat((input, target), dim=0)
            tmp = self.transform(tmp)
            input, target = tmp[:in_channels], tmp[in_channels:]
        return input, target


train_transform = transforms.Compose((
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
))
try:
    if config.reload_data or not os.path.exists(preprocessed_folder):
        raise Exception()
    train_set = _REF_Dataset(train_path, transform=train_transform)
    val_set = _REF_Dataset(val_path)
    test_set = _REF_Dataset(test_path)
except:
    _preprocess(train, train_path, patchify=True)
    _preprocess(val, val_path)
    _preprocess(test, test_path)
    train_set = _REF_Dataset(train_path, transform=train_transform)
    val_set = _REF_Dataset(val_path)
    test_set = _REF_Dataset(test_path)
len_train, len_val, len_test = \
    len(train_set), len(val_set), len(test_set)

train_loader = DataLoader(
    train_set, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(
    val_set, batch_size=1, shuffle=True)
test_loader = DataLoader(
    test_set, batch_size=1, shuffle=True)

if __name__ == "__main__":
    for x, y in train_loader:
        print(x.shape, y.shape)
        print(x.min(), x.max(), y.min(), y.max())
