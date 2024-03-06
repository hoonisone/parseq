import torch
from PIL import Image
from strhub.data.module import SceneTextDataModule
from strhub.models.parseq.system import *
from pathlib import Path
import glob
# Load model and image transforms
# parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
# print(parseq.optimizer)
url = 'https://github.com/baudm/parseq/releases/download/v1.0.0/parseq-bb5792a6.pt'
x = torch.hub.load_state_dict_from_url(url=url, map_location='cpu', check_hash=True)
print(type(x))
print(x["pos_queries"])
# for key in x.keys():
#     print(key)

# torch.save({
#             'epoch': 1,
#             'model_state_dict': parseq.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': LOSS,
#             }, PATH)

# print(parseq)
# print(type(parseq))
# path = "/home/MH2/PARSeq/ray_results/parseq/2023-09-24_11-16-19/train_cc9700f8_1_lr=0.0020_2023-09-24_11-16-25/checkpoint_epoch=7-step=3632/checkpoint"
# x = torch.load(path)
# parseq.load_state_dict(x["state_dict"])
