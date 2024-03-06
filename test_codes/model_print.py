import torch
from PIL import Image
from strhub.data.module import SceneTextDataModule
from strhub.models.parseq.system import *
from pathlib import Path
import glob
import torchsummary
from pytorch_lightning.utilities.model_summary import summarize
# Load model and image transforms

import argparse
import string
import sys
from dataclasses import dataclass
from typing import List

import torch

from tqdm import tqdm

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args


model_names = [
    'parseq-tiny',
    'parseq',
    'abinet',
    'trba',
    'vitstr',
    'crnn']


for model_name in model_names:
    model = load_from_checkpoint(f"pretrained={model_name}").eval()
    print(summarize(model, max_depth=1))