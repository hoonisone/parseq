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
import glob
# glob이라는 녀석은 파일 경로들을 찾을 때 사용하는 패키지인 것 같다.
# 특정 확장자나 파일 형식 등을 규칙 기반으로 찾아서 리스트로 반환해준다.
import io
import logging
import unicodedata
# 유니코드가 겁나 많을 것인데 그게 디비로 관리되는 모양
# 파이선에서 유니코드 디비에 접근하는 코드를 갖고 있는 패키지인듯
from pathlib import Path, PurePath
from typing import Callable, Optional, Union

import lmdb
# LMDB라는 매우 경량화 되고 대규모 데이터 처리에 적합한 데이터베이스가 있다.
# 해당 디비에 접근하는 패키지인듯
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset

from strhub.data.utils import CharsetAdapter

log = logging.getLogger(__name__)


def build_tree_dataset(root: Union[PurePath, str], *args, **kwargs):
    #*args는 전달 된 argments 중에 앞에서 매핑되어 받은 것 외에 나머지 중
    # keywork 즉 parameter name이 없는 argment들을 튜플 형태로 받아온다.
    # **kwargs는 마찬가지로 앞에서 매핑되지 않은 argment들 중 parameter name이 함께
    # key-value 형태로 넘어온 것들을 보두 받아 dict 형태로 받아온다. 
    try:
        kwargs.pop('root')  # prevent 'root' from being passed via kwargs
        # kwargs는 dict이고 여기서 dict 객체.pop(key)는
        # 해당 딕셔너리에서 키에 해당하는 key-value쌍을 제거하며 반환하는 기능이다. 
        # 이때 해당 키가 없으면 에러가 발생하는 것 같다.
        # 근데 root를 어케 쓴다는 거지?
        # 사용을 안하네?

    except KeyError:
        # KeyError에 대해 예외 처리를 함
        pass
    root = Path(root).absolute()
        # 여기서 부터는 아마 코드가 실행되는 환경(디바이스)에 I/O system에 종속된 경로가 나올 것이다.

    log.info(f'dataset root:\t{root}')
        # 로그는 나중에 한 번에 공부하는 것으로
    datasets = [] # 존체하는 모든 데이터를 다 가져오니까 리스트네
    # ******************* 드디어!!!!! 디비 관련 코드다!!!
    print(f"root: {root}")
    for mdb in glob.glob(str(root / '**/data.mdb'), recursive=True):
        # 위 문장에서 glob.glob(str(root / '**/data.mdb'), recursive=True) 부분은
        # root 경로 아래에 임의의 디렉터리 아래에 data.mdb라고 하는 모든 파일들을 찾고 그 경로를 받아오는 것이며
        # recursive=True는 하위 디렉터리에 대해서도 계속 찾는 기능인 듯 하다.
        mdb = Path(mdb)
        print("1")
        # glob의 결과는 각 경로를 string으로 주기 때문에 다시 Path객체로 변환하여 사용
        ds_name = str(mdb.parent.relative_to(root)) # 디렉토리 이름이 데이터 셋 이름
        print("2")
        ds_root = str(mdb.parent.absolute()) # 
        print("3")
        dataset = LmdbDataset(ds_root, *args, **kwargs)
        print("4")
        log.info(f'\tlmdb:\t{ds_name}\tnum samples: {len(dataset)}')
        print("5")
        datasets.append(dataset)
    print(f"datasets: {datasets}")
    return ConcatDataset(datasets)
    # 만약 datasets = [d1, d2, d3] 이런 식으로 구성되어있으면
    # ConcatDataset(datasets) == d1+d2+d3 라고한다.
    # 근데 뭐 요소가 많을 때는 그냥 리스트 한 번에 넘겨서 합치는 것이 훨씬 간편하지..


class LmdbDataset(Dataset):
    """Dataset interface to an LMDB database.

    It supports both labelled and unlabelled datasets. For unlabelled datasets, the image index itself is returned
    as the label. Unicode characters are normalized by default. Case-sensitivity is inferred from the charset.
    Labels are transformed according to the charset.
    """
    # 디비에는 이미지와 레이블이 쌍을 지어 차례로 저장되어있는 듯
    # 쌍을 짓기 보단 이미지와 레이블이 같은 순서대로 image1, image2, .../ label1, labe2, ... 이렇게 저장되는 듯
    # 디비에 구성 방식에 종속되려나?
    # 디비 내용을 아직 못봐서 모르겠다.

    def __init__(self, root: str, charset: str, max_label_len: int, min_image_dim: int = 0,
                 remove_whitespace: bool = True, normalize_unicode: bool = True,
                 unlabelled: bool = False, transform: Optional[Callable] = None):
        # Optional[A]은 변수가 A 타입인데 bool도 허용한다는 뜻
        self._env = None
        self.root = root
        self.unlabelled = unlabelled
        self.transform = transform
        self.labels = [] # 여기에는 실제로 사용될 샘플의 레이블만 저장됨
        # 레이블은 디비에 저장되어있지만 이렇게 따로 하는 이유는 전처리를 미리 다 해서 보관하고 있기 때문
        # 그게 아니라면 디비에서 매번 가져와도 될 듯
        # 근데 이미지는 너무 커서 다 불러올 수 없지만 레이블 정도는 이렇게 할 수 있지~
        self.filtered_index_list = [] # 여기서는 실제로 사용될 샘플의 인덱스만 저장됨
        self.num_samples = self._preprocess_labels(charset, remove_whitespace, normalize_unicode,
                                                   max_label_len, min_image_dim)
        # 위 함수는 데이터 셋에서 샘플들을 정규화 하고 필터링 하여 labels와 filtered_index_list를 세팅하고
        # 필터링 된 개수만 반환하는 함수이다.

    def __del__(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    def _create_env(self):
        return lmdb.open(self.root, max_readers=1, readonly=True, create=False,
                         readahead=False, meminit=False, lock=False)

    @property
    def env(self):
        if self._env is None:
            self._env = self._create_env()
        return self._env

    def _preprocess_labels(self, charset, remove_whitespace, normalize_unicode, max_label_len, min_image_dim):
        # STR 데이터셋에는 이미지와 레이블 쌍이 있을 텐데
        # 여기서 레이블은 정규화하고, 레이블과 이미지의 크기가 원하는 사이즈와 다르면 제거해서
        # 최종적으로 데이터 셋에서 필요한 데이터만 잘 정리해서 반환하는 함수이다.
        # 결과값은 self.labels에 정리된다.
        # 이미지 값은 필요 없는 것이 디비에서 가져오면 되니까?
        # 결과는 최종적으로 필터링 된 index와 label이 self.fitered_index_list와 self.labels에 각각 들어간다.
        # 그리고 필터링 된 결과의 개수를 반환
        
        # whitespace는 스페이스, 엔터, 탭 등 공백을 찍는 문자를 말한다.
        # 언어모델이기 때문에 white-space를 제거 등의 옵션이 있을 수 있겠다.
        charset_adapter = CharsetAdapter(charset) # target_dataset으로 필터링 하는 객체
        with self._create_env() as env, env.begin() as txn:
            # txn은 transaction을 의미하는 변수라 하네? (Chat GPT)
            # begin()은 파일을 실제로 여는 것인가?
            # begin에는 wirte(bool) 인자가 있다. 작성할지 말지 결정하는 듯 
            num_samples = int(txn.get('num-samples'.encode()))
            if self.unlabelled:
                return num_samples
            for index in range(num_samples):
                index += 1  # lmdb starts with 1 => range(1, num_samples+1) 로 했어도 될 듯
                label_key = f'label-{index:09d}'.encode()
                label = txn.get(label_key).decode()
                # Normally, whitespace is removed from the labels.
                if remove_whitespace:
                    label = ''.join(label.split()) # 어쨋든 이렇게 하면 공백이 사라짐
                # Normalize unicode composites (if any) and convert to compatible ASCII characters
                if normalize_unicode:
                    label = unicodedata.normalize('NFKD', label).encode('ascii', 'ignore').decode()
                    # 인코딩 방식이 다를 수 있으며 유니코드도 다양한 인코딩 방식(utf-8, utf-16 등)이 있기 때문에 
                    # 정규화를 해줘야 한다. 해당 정규화 코드가 unicodedata에 구현되있는 것 같고
                    # 4가지 정규화 방법(NFC, NFD, NFKC, NFKD) 중 뭐든 일관되게만 쓰면 된다는데
                    # 여기서는 그 중 NFKD를 사용하고 있는 것
                    # 그래서 정규화 하고
                # Filter by length before removing unsupported characters. The original label might be too long.
                if len(label) > max_label_len:
                    continue
                    # 아래에 self.labels에 append 하는 코드가 있다.
                    # continue한다는 것은 결국 무시하고 버린다는 말
                label = charset_adapter(label)
                # We filter out samples which don't contain any supported characters
                if not label: # adapter를 거친 label이 "" 즉 공백인 경우 
                    continue # 버림
                # Filter images that are too small.
                if min_image_dim > 0:
                    img_key = f'image-{index:09d}'.encode()
                    buf = io.BytesIO(txn.get(img_key))
                    w, h = Image.open(buf).size
                    if w < self.min_image_dim or h < self.min_image_dim:
                        continue # 사이즈가 임계치 보다 작으면 제외
                self.labels.append(label) # 원본 데이터에서 필터링 된 것만 사용하는듯
                self.filtered_index_list.append(index) 
                # 필터링 된 것이 원본에서 각각 몇번째인지를 알아야 디비에서 불러올 수 있으니~
        return len(self.labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if self.unlabelled:
            label = index
        else:
            label = self.labels[index]
            index = self.filtered_index_list[index]

        img_key = f'image-{index:09d}'.encode() # 이미지는 직접 디비에서 가져옴
        # 키 구성이 위에처럼 되어있나보다
        # image-1:09d
        # 여기서 09d는 % 매핑에서 %09d와 같은 기능
        # f"" 매핑에서도 이게 되는구나~
        # 이건 DB자체가 이런식으로 구성되어있다고 가정하는 것 같은데?
        # 그럼 DB의 구성을 정확히 알아야 이 코드에서 동작 가능하도록 만들 수 있겠다.
        with self.env.begin() as txn:
            imgbuf = txn.get(img_key)
        buf = io.BytesIO(imgbuf)
        img = Image.open(buf).convert('RGB')
        # Image 객체는 python image library (PIL)의 이미지 객체
        # 위 코드의 정확한 분석은 io.BytesIO()에 입력과 출력이 뭔지를 정확히 알아야 할듯

        if self.transform is not None:
            img = self.transform(img)

        return img, label
