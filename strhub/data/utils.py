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

import re # regular expression
from abc import ABC, abstractmethod
from itertools import groupby
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


class CharsetAdapter:
    """Transforms labels according to the target charset."""
    """
    label(사실 임의의 문자열)에 대해 사용하기로 한 문자 외의 것들을 모두 제거해주는 객체
    당연히 대상 문자열을 주어야 하고, 대문자만 쓰는지 소문자만 쓰는지고 검사를 해서
    예로 소문자만 쓰는 경우 검사 결과 문자열을 먼저 소문자로 바꾼 뒤 검사해줌
    한글만 사용한다면 큰 의미는 없을 듯
    사용방법은 객체를 직접 호출하는 방식 => 객체.(문자열)
    대상이 아닌 문자열을 모두 제거한 깨끗한 문자열 반환
    """

    def __init__(self, target_charset) -> None:
        # an example of target_charset => "0123456789abcdefghijklmnopqrstuvwxyz"
        super().__init__()
        self.lowercase_only = target_charset == target_charset.lower() 
        # 혹시 대상 문자열이 모두 소문자이면 소문자 전용이라 생각
        # 추후 검사 대상 문자열을 모두 소문자로 바꾸기 위함
        self.uppercase_only = target_charset == target_charset.upper()
        # 이것도 마찬가지로 모두 대문자만 사용하는지 검사
        self.unsupported = re.compile(f'[^{re.escape(target_charset)}]')
        # self.unsupported는 어떤 문자에 대해 지원되지 않는지 여부를 알기 위핸 re 객체
        # self.unsupported(문자열)을 넣었을 때 문자열 중 지원되지 않는 문자를 찾을 수 있다.
        # 정확한 것은 re의 사용법을 알아야 하겠다. 
        # 일단 [^abc] 라는 정규표현식은 abc 셋 중 하나로 시작하지 않는 문자를 말한다.
        # 그리고 한글자를 의미하기 때문에 글자 단위로 적용된다.

    def __call__(self, label):
        # 객체에 괄호 치고 인자 넘기며 함수취급했을 때 호출되는 함수
        # ex) 인스턴스(x, y,... )
        if self.lowercase_only:
            label = label.lower()
        elif self.uppercase_only:
            label = label.upper()
        # Remove unsupported characters
        label = self.unsupported.sub('', label)
        # label에서 제거 대상(unsupported에 검사에서 true가 나오는 것)을 ""로 바꾼다.
        # 즉 지운다. 그 후 결과를 반환
        # 결과적으로 대상이 아닌 문자는 모두 제거 
        return label


class BaseTokenizer(ABC):

    def __init__(self, charset: str, specials_first: tuple = (), specials_last: tuple = ()) -> None:
        self._itos = specials_first + tuple(charset) + specials_last
        self._stoi = {s: i for i, s in enumerate(self._itos)}

    def __len__(self):
        return len(self._itos)

    def _tok2ids(self, tokens: str) -> List[int]:
        return [self._stoi[s] for s in tokens]

    def _ids2tok(self, token_ids: List[int], join: bool = True) -> str:
        tokens = [self._itos[i] for i in token_ids]
        return ''.join(tokens) if join else tokens

    @abstractmethod
    def encode(self, labels: List[str], device: Optional[torch.device] = None) -> Tensor:
        """Encode a batch of labels to a representation suitable for the model.

        Args:
            labels: List of labels. Each can be of arbitrary length.
            device: Create tensor on this device.

        Returns:
            Batched tensor representation padded to the max label length. Shape: N, L
        """
        raise NotImplementedError

    @abstractmethod
    def _filter(self, probs: Tensor, ids: Tensor) -> Tuple[Tensor, List[int]]:
        """Internal method which performs the necessary filtering prior to decoding."""
        raise NotImplementedError

    def decode(self, token_dists: Tensor, raw: bool = False) -> Tuple[List[str], List[Tensor]]:
        """Decode a batch of token distributions.

        Args:
            token_dists: softmax probabilities over the token distribution. Shape: N, L, C
            raw: return unprocessed labels (will return list of list of strings)

        Returns:
            list of string labels (arbitrary length) and
            their corresponding sequence probabilities as a list of Tensors
        """
        batch_tokens = []
        batch_probs = []
        for dist in token_dists:
            probs, ids = dist.max(-1)  # greedy selection
            if not raw:
                probs, ids = self._filter(probs, ids)
            tokens = self._ids2tok(ids, not raw)
            batch_tokens.append(tokens)
            batch_probs.append(probs)
        return batch_tokens, batch_probs


class Tokenizer(BaseTokenizer):
    BOS = '[B]'
    EOS = '[E]'
    PAD = '[P]'

    def __init__(self, charset: str) -> None:
        specials_first = (self.EOS,)
        specials_last = (self.BOS, self.PAD)
        super().__init__(charset, specials_first, specials_last)
        self.eos_id, self.bos_id, self.pad_id = [self._stoi[s] for s in specials_first + specials_last]

    def encode(self, labels: List[str], device: Optional[torch.device] = None) -> Tensor:
        batch = [torch.as_tensor([self.bos_id] + self._tok2ids(y) + [self.eos_id], dtype=torch.long, device=device)
                 for y in labels]
        return pad_sequence(batch, batch_first=True, padding_value=self.pad_id)

    def _filter(self, probs: Tensor, ids: Tensor) -> Tuple[Tensor, List[int]]:
        ids = ids.tolist()
        try:
            eos_idx = ids.index(self.eos_id)
        except ValueError:
            eos_idx = len(ids)  # Nothing to truncate.
        # Truncate after EOS
        ids = ids[:eos_idx]
        probs = probs[:eos_idx + 1]  # but include prob. for EOS (if it exists)
        return probs, ids


class CTCTokenizer(BaseTokenizer):
    BLANK = '[B]'

    def __init__(self, charset: str) -> None:
        # BLANK uses index == 0 by default
        super().__init__(charset, specials_first=(self.BLANK,))
        self.blank_id = self._stoi[self.BLANK]

    def encode(self, labels: List[str], device: Optional[torch.device] = None) -> Tensor:
        # We use a padded representation since we don't want to use CUDNN's CTC implementation
        batch = [torch.as_tensor(self._tok2ids(y), dtype=torch.long, device=device) for y in labels]
        return pad_sequence(batch, batch_first=True, padding_value=self.blank_id)

    def _filter(self, probs: Tensor, ids: Tensor) -> Tuple[Tensor, List[int]]:
        # Best path decoding:
        ids = list(zip(*groupby(ids.tolist())))[0]  # Remove duplicate tokens
        ids = [x for x in ids if x != self.blank_id]  # Remove BLANKs
        # `probs` is just pass-through since all positions are considered part of the path
        return probs, ids
