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
import math
from pathlib import Path

from omegaconf import DictConfig, open_dict
import hydra
from hydra.core.hydra_config import HydraConfig

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.model_summary import summarize

from strhub.data.module import SceneTextDataModule
from strhub.models.base import BaseSystem
from strhub.models.utils import get_pretrained_weights


# Copied from OneCycleLR
def _annealing_cos(start, end, pct):
    "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    cos_out = math.cos(math.pi * pct) + 1
    return end + (start - end) / 2.0 * cos_out


def get_swa_lr_factor(warmup_pct, swa_epoch_start, div_factor=25, final_div_factor=1e4) -> float:
    """Get the SWA LR factor for the given `swa_epoch_start`. Assumes OneCycleLR Scheduler."""
    total_steps = 1000  # Can be anything. We use 1000 for convenience.
    start_step = int(total_steps * warmup_pct) - 1
    end_step = total_steps - 1
    step_num = int(total_steps * swa_epoch_start) - 1
    pct = (step_num - start_step) / (end_step - start_step)
    return _annealing_cos(1, 1 / (div_factor * final_div_factor), pct)


@hydra.main(config_path='configs', config_name='main', version_base='1.2')
# 원래 argparse 같은 아그먼트 라이브러리를 사용하면 명령어와 함께 인자를 일일이 입력해줘야 함
# 또한 파일에 저장하여 파일로 한다고 해도 파일의 내용을 불러와 인자로 매핑하는 코드를 작성해야 함
# hidra에서는 yaml 파일에 config를 설정하고 경로만 주면 알아서  config 객체에 해당 내용을 로딩해줌

def main(config: DictConfig):
    # config에서는 configs/main.yaml에 있는 hydra config 내용이 자동으로 담김
    trainer_strategy = None
    with open_dict(config):
        # omegaconf의 open_dict 함수는 config의 struct 속성을 임시로 False로 하여 
        # 존재하지 않는 속성에 접근시 자동으로 생성되는 것을 허용함
        # Resolve absolute path to data.root_dir
        config.data.root_dir = hydra.utils.to_absolute_path(config.data.root_dir)
        # Special handling for GPU-affected config
        gpu = config.trainer.get('accelerator') == 'gpu'
        devices = config.trainer.get('devices', 0)
        # DictConfig.get()은 원하는 속성을 반환하는 함수
        # arg1: 속성 key
        # arg2: 속성이 없는 경우 대체 값 (default value)

        if gpu:
            # Use mixed-precision training
            config.trainer.precision = 16
            # 그 속도 빠르게 하기 위해서 비트 낮은 소수를 사용하는 그건가?
            # 현재 config의 struct flag가 False여서 precision이 없어도 자동 생성되어 추가됨
        if gpu and devices > 1: # 멀티 gpu를 사용하는 경우
            # Use DDP
            config.trainer.strategy = 'ddp' # distributed data parallel
            # DDP optimizations
            trainer_strategy = DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True) # ???
            # 뒤에서 traininer에게 넘겨주는 객체임


            # Scale steps-based config
            config.trainer.val_check_interval //= devices
            # val_check_intervals은 validation 수행 주기인 것 같음
            # 장치가 늘었으니 같은 데이터 양 마다 val을 수행하는 거라면 주기는 장치 개수로 나뉘는 것이 맞음
            if config.trainer.get('max_steps', -1) > 0:
                # 위 조건문은 max_steps가 있는지 묻는 것과 같음
                config.trainer.max_steps //= devices
                # 있다면 나눈다.

    # Special handling for PARseq
    if config.model.get('perm_mirrored', False):
        # config.model에 perm_mirrored라는 변수가 있으며 True이면
        assert config.model.perm_num % 2 == 0, 'perm_num should be even if perm_mirrored = True'
        # 짝수인지 체크
        # perm_num이 뭘까?

    model: BaseSystem = hydra.utils.instantiate(config.model)
    # config에는 model이라는 정보가 있고 이는 하나의 객체 생성을 위한 정보들을 담는다.
    # main.yaml에는 defalt로 model: parseq가 되어있으므로
    # model.parseq.config가 디폴트로 설정된다.
    
    # If specified, use pretrained weights to initialize the model
    if config.pretrained is not None:
        # config.pretrained는 사전 학습된 모델의 checkpoint 파일이 있는 경로이다.
        # 있다면 가져와서 모델에 씌우고 학습을 시작
        # 경로 또는 url인듯
        model.load_state_dict(get_pretrained_weights(config.pretrained))
        # config.pretrained로 부터 모델의 state_dict 객체를 뽑고 모델에 해당 내용을 세팅함
        # state dict는 모델의 각 모듈 별 파라미터를 저장하는 객체 (ordered dict)
        #  

    print(summarize(model, max_depth=1 if model.hparams.name.startswith('parseq') else 2))

    datamodule : SceneTextDataModule = hydra.utils.instantiate(config.data)
    # main.yaml에 있는 data 생성 정보를 가지고 데이터 모듈 생성
    # 이때 SceneTextDataModule은 LightningDataModule을 구현한 것으로
    # LightningDataModule은 train, val, test 데이터 셋을 모두 관리하며 split, transform 등을 모두 관리
    #  

    checkpoint = ModelCheckpoint(monitor='val_accuracy', mode='max', save_top_k=3, save_last=True,
                                 filename='{epoch}-{step}-{val_accuracy:.4f}-{val_NED:.4f}')
    
    # moditor: 저장할 수량이라 하는데, 양을 뜻하기 보단 어떤 값을 기록할지를 나타내는 걸까?..
    # save_last: 마지막 checkpoint는 특별히 lask.ckpt로 저장함 아마도 다음에 이어서 학습을 하기 용이하도록~
    # save_top_k: checkpoint 중 성능이 가장 높은 k개는 저장을 하겠다는 뜻
    # ModelCheckpoint는 torch lightning에 있는 객체
    # finename은 checkpoint 마다 파일 이름을 어떻게 할지 규칙을 정하는 부분
    # {} 안에 값은 알아서 대입해주는 모양
    # 예시는 클래스 코드 내에 잘 설명됨
    
    swa_epoch_start = 0.75
    # swa가 뭘까? SWA(Stochastic Weight Averaging): parameter space를 서치할 때 좋은 기법 중 하나인 듯
    swa_lr = config.model.lr * get_swa_lr_factor(config.model.warmup_pct, swa_epoch_start)
    # warm up: 맨 처음 학습 할 때 랜덤 파라미터로 초기화 된 상태에서 학습을 하는 것이 불안정하다고 함
    #           그래서 초기에 조금 천천히 시작하는 것이 좋다고.. 그러한 작업을 learning rate warm up이라 한다고 함
    swa = StochasticWeightAveraging(swa_lr, swa_epoch_start)
    # SWA은 여러 지점의 파라미터 평균을 이용하므로서 좀 원하는 해에 가까운 지점을 찾을 수 있다고~
    cwd = HydraConfig.get().runtime.output_dir if config.ckpt_path is None else \
        str(Path(config.ckpt_path).parents[1].absolute())
    # cwd는 컴퓨터 공학에서 current working directory를 의미함
    trainer: Trainer = hydra.utils.instantiate(config.trainer, logger=TensorBoardLogger(cwd, '', '.'),
                                               strategy=trainer_strategy, enable_model_summary=False,
                                               callbacks=[checkpoint, swa])
    
    # callbacks을 []로 넘겨서 원하는 콜백들을 동적 개수 만큼 수행시키는 방식이 너무 매력적인 것 같다.
    trainer.fit(model, datamodule=datamodule, ckpt_path=config.ckpt_path)


if __name__ == '__main__':
    main()
