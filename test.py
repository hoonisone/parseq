#!/usr/bin/env python3
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



import argparse
import string
import sys
from dataclasses import dataclass
from typing import List

import torch

from tqdm import tqdm

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args


@dataclass
class Result:
    dataset: str
    num_samples: int
    accuracy: float
    ned: float
    confidence: float
    label_length: float


def print_results_table(results: List[Result], file=None):
    w = max(map(len, map(getattr, results, ['dataset'] * len(results))))
    w = max(w, len('Dataset'), len('Combined'))
    print('| {:<{w}} | # samples | Accuracy | 1 - NED | Confidence | Label Length |'.format('Dataset', w=w), file=file)
    print('|:{:-<{w}}:|----------:|---------:|--------:|-----------:|-------------:|'.format('----', w=w), file=file)
    c = Result('Combined', 0, 0, 0, 0, 0)
    for res in results:
        c.num_samples += res.num_samples
        c.accuracy += res.num_samples * res.accuracy
        c.ned += res.num_samples * res.ned
        c.confidence += res.num_samples * res.confidence
        c.label_length += res.num_samples * res.label_length
        print(f'| {res.dataset:<{w}} | {res.num_samples:>9} | {res.accuracy:>8.2f} | {res.ned:>7.2f} '
              f'| {res.confidence:>10.2f} | {res.label_length:>12.2f} |', file=file)
    c.accuracy /= c.num_samples
    c.ned /= c.num_samples
    c.confidence /= c.num_samples
    c.label_length /= c.num_samples
    print('|-{:-<{w}}-|-----------|----------|---------|------------|--------------|'.format('----', w=w), file=file)
    print(f'| {c.dataset:<{w}} | {c.num_samples:>9} | {c.accuracy:>8.2f} | {c.ned:>7.2f} '
          f'| {c.confidence:>10.2f} | {c.label_length:>12.2f} |', file=file)


@torch.inference_mode()
def main():
    # argment 정리

    # 미리 정의된 argments
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    # 코드를 보니 checkpoint는 일단 학습된 모델에 대한 정보를 담는 파일이며
    # 두 가지 경우가 있다. (1) 저자의 pretrained 모델 학습 (2) 그 외의 모델 학습
    # (1)을 위해서는 pretrained= 형식으로 하게 되어있음
    # 이렇게 하면 서버에 있는 저자가 학습시킨 가중치를 불러와서 하용하게 됨
    # 이때 id는 configs/experiments에 있는 파일 이름 사용
    # (2) 의 경우 파일 이름 안에 모델 이름이 들어있다고 가정하고
    # 해당 모델에 checkpoint 경로에 있는 가중치를 넣어 모델을 로드함
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    # type을 주면 값을 해당 type으로 바꾸어서 받아줌
    # 아마 type말고도 콜백을 주면 해당 콜백의 반환 값으로 바꾸어서 받아줄 듯 (해보니 진짜 그렇게 되네)

    parser.add_argument('--cased', action='store_true', default=False, help='Cased comparison')
    # action은 parameter value가 어떻게 handle될 지를 결정
    # 미리 정의된 handling 들이 있으며
    # store_true는 인자의 값을 넘겨받는 경우 true를 아닌 경우 false로 값을 받는 handling이다.
    parser.add_argument('--punctuation', action='store_true', default=False, help='Check punctuation')
    parser.add_argument('--new', action='store_true', default=False, help='Evaluate on new benchmark datasets')
    parser.add_argument('--rotation', type=int, default=0, help='Angle of rotation (counter clockwise) in degrees.')
    parser.add_argument('--device', default='cuda')
    # 여기까지는 어떤 parameter 들이 있는지 정의하는 부분
    
    args, unknown = parser.parse_known_args()
    # parser.parse_known_args()를 수행하면 명령어와 함께 받아온 인자들일 parser에 정의한 대로 정리하여 반환
    # 이때 정의된 파라미터들을 정리한 것(args)과 정의되지 않은 것들(unknown)에 대해 구분하여 반환
    # args: <class 'argparse.Namespace'> like dictionary
    # unknown: list -> [key:type=value, ... ]
    # unknown은 위에 parser에서 정해지지 않은 옵션들에 대해 따로 모아서 list로 만들어짐
    
    # 미리 정의되지 않은 argments 관리
    kwargs = parse_model_args(unknown)
    # kwargs: dict
    # 알려지지 않은 argment token(key:type=value)들에 대해 토크닝하고 지정된 타입으로 바꾸어 dictionary로 만들어준다.

    
    charset_test = string.digits + string.ascii_lowercase
    # string.digits = 숫자 문자들의 나열 (문자열) "0123456789"
    # string.문자 리스트  이런식으로 다양한 문자 셋을 쉽게 접근할 수 있도록 해주는 것 같음
    
    if args.cased: # cased 설정이 true이면 (아마 대문자를 구분할지?)
        charset_test += string.ascii_uppercase # 문자 셋에 대문자도 추가 하겠다.
    if args.punctuation:
        charset_test += string.punctuation
        # punctuation 구두점

    kwargs.update({'charset_test': charset_test})
    # dict1.update(dict2) => dict2에 있는 쌍들에 대해 dict1에 이미 있는 것은 값을 덮어 씌우고 없는 경우 추가한다. (덧셈 느낌?)

    print(f'Additional keyword arguments: {kwargs}')
    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    # model에는 checkpoint 내에 있는 정보에 기반하여 로딩된 모델이 담김
    # model 클래스와 가중치 정보가 checkpoint 경로에 있는 yaml파일에 있음

    hp = model.hparams
    datamodule = SceneTextDataModule(args.data_root, '_unused_', hp.img_size, hp.max_label_length, hp.charset_train,
                                     hp.charset_test, args.batch_size, args.num_workers, False, rotation=args.rotation)

    test_set = SceneTextDataModule.TEST_BENCHMARK_SUB + SceneTextDataModule.TEST_BENCHMARK
    print(1, test_set)
    if args.new:
        test_set += SceneTextDataModule.TEST_NEW
    test_set = sorted(set(test_set)) # 중복이 있을 수 있어 set으로 ..
    results = {}
    max_width = max(map(len, test_set))
    

    # 테스트 단계
    for name, dataloader in datamodule.test_dataloaders(test_set).items():
        total = 0
        correct = 0
        ned = 0
        confidence = 0
        label_length = 0
        for imgs, labels in tqdm(iter(dataloader), desc=f'{name:>{max_width}}'):
            res = model.test_step((imgs.to(model.device), labels), -1)['output']
            total += res.num_samples
            correct += res.correct
            ned += res.ned
            confidence += res.confidence
            label_length += res.label_length
        accuracy = 100 * correct / total
        mean_ned = 100 * (1 - ned / total)
        mean_conf = 100 * confidence / total
        mean_label_length = label_length / total
        results[name] = Result(name, total, accuracy, mean_ned, mean_conf, mean_label_length)
    
    # 결과 출력
    result_groups = {
        'Benchmark (Subset)': SceneTextDataModule.TEST_BENCHMARK_SUB,
        'Benchmark': SceneTextDataModule.TEST_BENCHMARK
    }
    if args.new:
        result_groups.update({'New': SceneTextDataModule.TEST_NEW})
    with open(args.checkpoint + '.log.txt', 'w') as f:
        for out in [f, sys.stdout]:
            for group, subset in result_groups.items():
                print(f'{group} set:', file=out)
                print_results_table([results[s] for s in subset], out)
                print('\n', file=out)


if __name__ == '__main__':
    main()
