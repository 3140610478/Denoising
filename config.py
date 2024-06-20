import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

reload_data = False
# reload_data = True

batch_size = 128

std = 25

layers = 20

base_folder = os.path.dirname(os.path.abspath(__file__))
save_folder = os.path.abspath(os.path.join(
    base_folder, "./Networks/save"
))
os.makedirs(save_folder, exist_ok=True)