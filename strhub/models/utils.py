from pathlib import PurePath
from typing import Sequence

import torch
from torch import nn

import yaml


class InvalidModelError(RuntimeError):
    """Exception raised for any model-related error (creation, loading)"""


_WEIGHTS_URL = {
    'parseq-tiny': 'https://github.com/baudm/parseq/releases/download/v1.0.0/parseq_tiny-e7a21b54.pt',
    'parseq-patch16-224': 'https://github.com/baudm/parseq/releases/download/v1.0.0/parseq_small_patch16_224-fcf06f5a.pt',
    'parseq': 'https://github.com/baudm/parseq/releases/download/v1.0.0/parseq-bb5792a6.pt',
    'abinet': 'https://github.com/baudm/parseq/releases/download/v1.0.0/abinet-1d1e373e.pt',
    'trba': 'https://github.com/baudm/parseq/releases/download/v1.0.0/trba-cfaed284.pt',
    'vitstr': 'https://github.com/baudm/parseq/releases/download/v1.0.0/vitstr-26d0fcf4.pt',
    'crnn': 'https://github.com/baudm/parseq/releases/download/v1.0.0/crnn-679d0e31.pt',
}


def _get_config(experiment: str, **kwargs):
    """Emulates hydra config resolution"""
    root = PurePath(__file__).parents[2]
    # PurePath.parents 를 하게 되면 경로에서 파일 이름을 제외한 경로 내 모든 디렉터리를 순서대로 리스트로 표현
    # 이중 2라고 하면 최상위 경로(디스트)로 부터 깊이 2에 위치한 폴더까지의 경로를 얻을 수 있음
    # 0이면 최상위 루트, 1이면 루트/디렉터리1 ...
    # __file__은 현재 주행중인 코드가 있는 파일의 경로를 반환
    # PurePath 는 pathlib에 있는 객체로
    # 특정 디바이스에 종속되지 않는 순수하게 경로에 대해서만 정보를 다루는 객체이다.
    # 아마 실제 경로를 디바이스에서 접근하기 전에 디바이스에 맞게 변경하지 않을까 싶다.
    # root를 현재 파일 기준으로 찾는 것은 앞으로 이 파일은 정확히 root로 부터 고정된 위치에 놓일 거라는 가정이 있는 것

    # 경로를 세는 위치가 뒤에서 부터가 아니라 앞에서 부터, 즉 디스크 부터면
    # 프로젝트를 어디에 위치시키는가에 따라 달라지는게 아닌가 싶지만
    # 일단 아래에서 configs/main.yaml이 있는데
    # 이것은 현재 프로젝트 루트 부터 시작하는 경로인 것을 고려해볼 때 
    # 일단 root변수에는 프로젝트의 루트가 담기는 것으로 간주하자.

    with open(root / 'configs/main.yaml', 'r') as f:
        config = yaml.load(f, yaml.Loader)['model']
        # mail.yaml파일에서 model에 대한 config 정보를 불러옴
    with open(root / f'configs/charset/94_full.yaml', 'r') as f:
        config.update(yaml.load(f, yaml.Loader)['model'])
        # 동일한 형식
    with open(root / f'configs/experiment/{experiment}.yaml', 'r') as f:
        exp = yaml.load(f, yaml.Loader)
        # 동일한 형식
        # experiment는 모델의 유형 이름인듯
    # Apply base model config
    model = exp['defaults'][0]['override /model']
    # experiment.yaml 파일에는 모두 같은 구조로 defaults 안에 첫번째 인자로 override /model 이 있음
    # 여기에는 parseq 처럼 베이스 모델의 이름이 적혀있음
    with open(root / f'configs/model/{model}.yaml', 'r') as f:
        config.update(yaml.load(f, yaml.Loader))
        # 해당 모델의 config를 가져옴
    # Apply experiment config
    if 'model' in exp:
        config.update(exp['model'])
    config.update(kwargs)
    # Workaround for now: manually cast the lr to the correct type.
    config['lr'] = float(config['lr'])
    return config


def _get_model_class(key):
    if 'abinet' in key:
        from .abinet.system import ABINet as ModelClass
    elif 'crnn' in key:
        from .crnn.system import CRNN as ModelClass
    elif 'parseq' in key:
        from .parseq.system import PARSeq as ModelClass
    elif 'trba' in key:
        from .trba.system import TRBA as ModelClass
    elif 'trbc' in key:
        from .trba.system import TRBC as ModelClass
    elif 'vitstr' in key:
        from .vitstr.system import ViTSTR as ModelClass
    else:
        raise InvalidModelError(f"Unable to find model class for '{key}'")
    return ModelClass


def get_pretrained_weights(experiment):
    # experiment는 미리 학습된 모델 정보들이 저장된 url의 key값이다.
    # _WEIGHTS_URL에 미리 학습된 모델의 url들이 있다.
    # 본 함수는 url에 맞는 학습된 모델의 state dict를 반환한다.

    try:
        url = _WEIGHTS_URL[experiment]
        # 직접 보유하고 있는 파일이 아니라 url에 해당하는 경우 (정확히는 url의 key 값)
        # 아마 저자가 제공하는 pretrained model 일 듯
        # 등록된 key에 해당하는 url을 가져온다.
    except KeyError:
        # 등록된 키가 아닌 경우
        raise InvalidModelError(f"No pretrained weights found for '{experiment}'") from None
    
    return torch.hub.load_state_dict_from_url(url=url, map_location='cpu', check_hash=True)
    # url이 있는 경우 거기서 부터 모델 정보를 불러오는 듯

def create_model(experiment: str, pretrained: bool = False, **kwargs):
    # experiment: 모델 id 등 모델을 식별할 수 있는 문자열인듯

    try:
        config = _get_config(experiment, **kwargs)
        # 여기서는 experiment가 ./configs/experiment/ 내에 파일 이름 중 하나라고 가정함
        # config 파일에서 해당 파일에 맞는 정보 config를 가져옴 (Hydra)

    except FileNotFoundError:
        raise InvalidModelError(f"No configuration found for '{experiment}'") from None
    ModelClass = _get_model_class(experiment)
    model = ModelClass(**config)
    if pretrained:
        model.load_state_dict(get_pretrained_weights(experiment))
    return model


def load_from_checkpoint(checkpoint_path: str, **kwargs):
    # checkpoint_path: str
    # checkpoint 정보로 부터 모델을 로드 하는 함수

    if checkpoint_path.startswith('pretrained='): # 저자가 해 놓은 사전 학습 결과물을 돌리는 경우
        # 이건 저자가 정한 checkpoint 파일 저장 형식이 있는듯
        model_id = checkpoint_path.split('=', maxsplit=1)[1] #문자(=) 뒷 부분
        # model_id도 결국은 str, 이름정도에 해당
        model = create_model(model_id, True, **kwargs)
        # ./configs/experiments 안에 있는 파일 이름 x에 대해
        # checkpoint_path가 "pretrained=x" 형식으로 되어있는 경우로 가정한다고 볼 수 있음
        # 그 경우 해당 
    else: # 그외
        ModelClass = _get_model_class(checkpoint_path) # 프로젝트 종속 코드
        # 파일 이름으로 부터 모델의 이름을 확인하고 모델의 클래스 객체를 반환
        # 저자는 파일 이름에 모델의 이름을 삽입했다고 가정( 실제로 그렇게 되어있음)
        model = ModelClass.load_from_checkpoint(checkpoint_path, **kwargs)
    return model


def parse_model_args(args) -> dict: # args: list
    # argment들을 dictionary에 담아서 반환하는 함수
    # 해결 이슈는 string으로 된 토큰 (name:type=value)을 잘 분할하여
    # value는 type으로 잘 변환하여 name과 매핑해야 한다.
    # 아래 코드는 이를 수행함
    kwargs = {}
    arg_types = {t.__name__: t for t in [int, float, str]} # ex) {int: <class 'int'>, ...}
    # <class 'int'>는 값을 넣으면 int로 변환해주는 자료형 변환 함수로 볼 수 있다.
    arg_types['bool'] = lambda v: v.lower() == 'true'  # special handling for bool
    # 위는 bool값 string에 대해 bool 타입으로 바꾸어 주는 함수이다.
    # true 외의 나머지 모든 값은 false로 처리하는 것 같다.

    for arg in args:
        # tokenization
        name, value = arg.split('=', maxsplit=1) # maxsplit=1을 의미적으로 해석하면 맨 처음 나오는 문자에 대해서만 나누겠다~
        name, arg_type = name.split(':', maxsplit=1)
        # type transform
        kwargs[name] = arg_types[arg_type](value) 

    return kwargs


def init_weights(module: nn.Module, name: str = '', exclude: Sequence[str] = ()):
    """Initialize the weights using the typical initialization schemes used in SOTA models."""
    if any(map(name.startswith, exclude)):
        return
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
