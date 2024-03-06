import torch
from PIL import Image
from strhub.data.module import SceneTextDataModule
from strhub.models.parseq.system import *
from pathlib import Path
import glob
# Load model and image transforms
parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()

# path = "/home/MH2/PARSeq/ray_results/parseq/2023-09-24_11-16-19/train_cc9700f8_1_lr=0.0020_2023-09-24_11-16-25/checkpoint_epoch=7-step=3632/checkpoint"
# x = torch.load(path)
# parseq.load_state_dict(x["state_dict"])

def f(path):
    img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)

    # img = Image.open('/path/to/image.png').convert('RGB')
    # img = Image.open('./demo_images/art-01107.jpg').convert('RGB')/
    img = Image.open(path).convert('RGB')
    img.show()
    # img = Image.open('./demo_images/aa.jpg').convert('RGB')
    img = img_transform(img).unsqueeze(0)

    logits = parseq(img)
    logits.shape  # torch.Size([1, 26, 95]), 94 characters + [EOS] symbol
    # print(f"logits: {logits}")

    # Greedy decoding
    pred = logits.softmax(-1)

    # print(f"logits: {logits}")
    label, confidence = parseq.tokenizer.decode(pred)
    print(f'{path.split("/")[-1]} => {label[0]}')



x = glob.glob(str(Path("./demo_images").absolute()/"**"))
print(x)
for path in x:
    f(path)
# f(name)