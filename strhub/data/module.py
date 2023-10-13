# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import PurePath
# 경로는 I/O와 직결되는데
# 보통 I/O는 환경에 종속된다.
# 따라서 윈도우에서 돌아가던게 맥에서는 안될 수도 있고 그런데
# pathlib를 구현하는 사람은 일단 I/O와 독립적으로 순수한 path 연산을 먼저 구현하고
# 실제 환경에 특화된 기능은 그 순수한 path를 상속하여 구현하였다.
# 이때 그 순수한 연산에 대한 모듈을 purepath라고 한다.
# PurePath 
from typing import Optional, Callable, Sequence, Tuple

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms as T

from .dataset import build_tree_dataset, LmdbDataset


class SceneTextDataModule(pl.LightningDataModule):
    # pl = pytorch_lightning
    # LightningDataModule
    TEST_BENCHMARK_SUB = ('IIIT5k', 'SVT', 'IC13_857', 'IC15_1811', 'SVTP', 'CUTE80')
    TEST_BENCHMARK = ('IIIT5k', 'SVT', 'IC13_1015', 'IC15_2077', 'SVTP', 'CUTE80')
    TEST_NEW = ('ArT', 'COCOv1.4', 'Uber')
    TEST_ALL = tuple(set(TEST_BENCHMARK_SUB + TEST_BENCHMARK + TEST_NEW))

    def __init__(self, root_dir: str, train_dir: str, img_size: Sequence[int], max_label_length: int,
                 charset_train: str, charset_test: str, batch_size: int, num_workers: int, augment: bool,
                 remove_whitespace: bool = True, normalize_unicode: bool = True,
                 min_image_dim: int = 0, rotation: int = 0, collate_fn: Optional[Callable] = None):
        super().__init__()
        self.root_dir = root_dir
        self.train_dir = train_dir
        self.img_size = tuple(img_size)
        self.max_label_length = max_label_length
        self.charset_train = charset_train
        self.charset_test = charset_test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment = augment # data augmentation을 의미하는듯 (bool) 인것을 봐서는 할까 말까를 선택하는 듯
        self.remove_whitespace = remove_whitespace
        self.normalize_unicode = normalize_unicode
        self.min_image_dim = min_image_dim
        self.rotation = rotation
        self.collate_fn = collate_fn
        self._train_dataset = None
        self._val_dataset = None

    @staticmethod
    def get_transform(img_size: Tuple[int], augment: bool = False, rotation: int = 0):
        # 필요한 transform들을 하나의 연산자(함수)로 구성하여 반환하는 함수이다.
        # 왜 이렇게 할까?
        # transform은 입력 데이터의 크기, 형태 등에 따라 파라미터가 제공될 필요가 있다.
        # 이때 목적에 따라 transform에 제공되는 파라미터를 달리하여 transform을 수행할 필요가 있다.
        # 예로 이미지를 2배 키우는 transform과 3배 키우는 transform??
        # 이 경우 필요한 transform을 하나로 묶어서 반환하는 get_transform함수를 만들고
        # 필요할 때 마다 인자를 넘겨서, 해당 인자에 맞는 하나의 transform 연산을 받아서 사용하면
        # 복잡도가 줄어들 수 있다.
        # 이 함수가 그런 역할인 듯 
        transforms = []
        if augment:
            # augment여부에 따라 augment용 transform을 넣는 코드네
            from .augment import rand_augment_transform
            # 기능, 옵션의 사용 여부에 따라서 동적으로 import를 할 수 도 있구나.
            # 항상 맨 앞에서 다 해야 할거라 생각했는데 ...
            transforms.append(rand_augment_transform())
            # transform들을 transforms에 모아 나중에 한 번에 compose 할 텐데
            # augment 여부에 따라 여기에 넣거나 말거나를 해주면 ㅇㅇ
        if rotation:
            transforms.append(lambda img: img.rotate(rotation, expand=True))
            # rotation도 할 지 말지에 따라 넣어주는 것 같은데
            # if rotation인 것을 보면
            # 일단 rotation은 얼마나 회전할지를 나타내는 회전량 값을 갖는 변수인 것 같고
            # 그게 각도인지 라디안인지는 몰라도 
            # 어쨌든 여기서 조건문을 쓴 것은 회전량이 0이면 굳이 변형(transform)을 할 필요가 없으니
            # 그냥 연산 자체를 추가하지 않는듯
            # 나중에 image가 정확히 어떤 객체인지 데이터 타입을 확인하고 
            # img.rotate가 뭘 해주는 녀석인지를 보면 더 좋을 듯
        transforms.extend([
            # list1.extend(list2)는 list2를 list1 뒤에 추가하는 연산, 원소로 추가가 아니라 모든 원소를 연결시킴
            T.Resize(img_size, T.InterpolationMode.BICUBIC),
            # T는 torchvision package에 있는 transforms이고 여기에는 
            # 기본적으로 쓰이는 다양한 transform operation들이 미리 구현되어있다.
            # 이 중 interpolationMode는
            # 우선 interpolation은 한국말로 보간법을 의미하며 
            # 변형 과정에서 빈 공간이 생겼을 때 여기를 어떻게 채울 것인가를 결정하는 방법을 의미한다.
            # 이 중  BICUBIC은 여러 보간 법 중에 하나인 것이고
            # 정확한 방법은 모르지만 선형 보간 보다는 자연스럽다는 것만 알아두자. 
            T.ToTensor(),
            T.Normalize(0.5, 0.5)
        ])
        # 결과적으로 반환 값은 augment, translation, resize 수행하 tensor로 바꾸고 
        # 노멀라이제이션 해주는 과정을 하나의 연산(함수)으로 묶은 것이다.
        return T.Compose(transforms)

    @property
    def train_dataset(self):
        
        if self._train_dataset is None:
            # 한 번 세팅 하면 그 다음부터는 저장했다가 하네
            # 일반적으로 train_dataset을 호출하면 해당 함수의 기능을 쭉 수행하는데
            # _train_dataset에 train_dataset에 대한 결과를 저장해놓고
            # 두 번째 호출 부터는 저장된 결과를 반환하도록 하고 있다.
            transform = self.get_transform(self.img_size, self.augment)
            root = PurePath(self.root_dir, 'train', self.train_dir)
            # 여기서 root는 train_dataset에 대한 경로인 것이고
            # pure를 쓴 이유는 특정 디바이스의 I/O system에 종속되지 않기 위함인듯
            # 일단 경로를 보면 root/train/train_dir 인데
            # 기본적으로 train_dataset이 여러개 일 수 있는데 그 데이터 셋들을 모두
            # root/train/self.train_dir에 두는 것으로 가정했다고 볼 수 있다.
            # 이때 root_dir, augment의 구체적인 값은 전달하는 인자값을 봐야 하고
            # 그 값은 아마 hydra를 사용하고 있기 때문에config.yaml을 봐야 한다.
            # 현재는 그 값이 각각 tada와 ???로 적혀있음 (확인 결과)
            # 여기서 ???는 아직 미정이라 이렇게 해놨는데 내가 직접 정해야 할 듯
            # 파일을 수정하거나 CLI? 에서 직접 입력해서 쓰면 될듯 
            self._train_dataset = build_tree_dataset(root, self.charset_train, self.max_label_length,
                                                     self.min_image_dim, self.remove_whitespace, self.normalize_unicode,
                                                     transform=transform)
            # 경로에 있는 모든 데이터 셋을 각각 LMDBDataset으로 만들고 합쳐서 반환
            # 각 디비에는 lmdb 형식으로 디비가 있어야 하는 것 같고
            # 디비에는 이미지와 레이블 쌍이 있는 듯
        return self._train_dataset

    @property
    def val_dataset(self):
        if self._val_dataset is None:
            transform = self.get_transform(self.img_size)
            root = PurePath(self.root_dir, 'val')
            self._val_dataset = build_tree_dataset(root, self.charset_test, self.max_label_length,
                                                   self.min_image_dim, self.remove_whitespace, self.normalize_unicode,
                                                   transform=transform)
        return self._val_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, persistent_workers=self.num_workers > 0,
                          pin_memory=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, persistent_workers=self.num_workers > 0,
                          pin_memory=True, collate_fn=self.collate_fn)

    def test_dataloaders(self, subset):
        transform = self.get_transform(self.img_size, rotation=self.rotation)
        root = PurePath(self.root_dir, 'test')
        datasets = {s: LmdbDataset(str(root / s), self.charset_test, self.max_label_length,
                                   self.min_image_dim, self.remove_whitespace, self.normalize_unicode,
                                   transform=transform) for s in subset}
        return {k: DataLoader(v, batch_size=self.batch_size, num_workers=self.num_workers,
                              pin_memory=True, collate_fn=self.collate_fn)
                for k, v in datasets.items()}
