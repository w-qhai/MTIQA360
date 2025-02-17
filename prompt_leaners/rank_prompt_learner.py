from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from clip import clip
from clip.model import CLIP

from ordinalclip.utils import get_logger

from .builder import PROMPT_LEARNERS
from .plain_prompt_learner import PlainPromptLearner
from itertools import product
logger = get_logger(__name__)


@PROMPT_LEARNERS.register_module()
class RankPromptLearner(PlainPromptLearner):
    interpolation_functions = {
        "linear": lambda weights, num_ranks: 1.0 - weights / (num_ranks - 1),
        "inv_prop": lambda weights, _, eps=1e-5: 1.0 / (weights + eps),
        "normal": lambda weights, _: torch.exp(-weights * weights),
    }

    def __init__(
        self,
        clip_model: CLIP,
        classnames: list,
        num_base_ranks: int,
        num_ranks: int,
        num_tokens_per_rank: Union[int, List],
        num_context_tokens: int,
        rank_tokens_position: str = "tail",
        init_rank_path: Optional[str] = None,
        init_context: Optional[str] = None,
        rank_specific_context: bool = False,
        interpolation_type: str = "linear",
        **kwargs,
    ) -> None:
        super(PlainPromptLearner, self).__init__()

        # if kwargs:
        #     logger.info(f"irrelevant kwargs: {kwargs}")

        dtype = dtype = clip_model.token_embedding.weight.dtype

        # distortion embeds
        classname_embeds, num_tokens_per_classname = self.create_classname_embeds(
            clip_model, classnames, logger, dtype
        )
        self.classname_embeds = nn.Parameter(
            classname_embeds, requires_grad=False
        )
        # classname_tokens = [clip._tokenizer.encode(name) for name in classnames]
        # num_tokens_per_classname = [len(tokens) for tokens in classname_tokens]
        # classname_embeds = torch.stack([clip_model.token_embedding(torch.tensor(tokens)) for tokens in classname_tokens])
        # print("classname ", classname_embeds.shape)
        # self.classname_embeds = nn.Parameter(
        #     classname_embeds, requires_grad=False
        # )
        # logger.info(f"num_tokens_per_classname: {num_tokens_per_classname}")
        
        # context embeds
        context_embeds, _num_context_tokens = self.create_context_embeds(
            clip_model, num_ranks, num_context_tokens, init_context, rank_specific_context, logger, dtype
        )
        num_context_tokens = _num_context_tokens
        self.context_embeds = nn.Parameter(
            context_embeds
        )  # (num_context_tokens, embeds_dim) or (num_ranks, num_context_tokens, embeds_dim)

        # rank embeds
        if isinstance(num_tokens_per_rank, int):
            num_tokens_per_rank = [num_tokens_per_rank] * num_base_ranks
        rank_embeds, _num_tokens_per_rank = self.create_rank_embeds(
            clip_model, num_base_ranks, num_tokens_per_rank, init_rank_path, logger, dtype, num_context_tokens
        )
        num_tokens_per_rank = [np.max(_num_tokens_per_rank)] * num_ranks
        self.rank_embeds = nn.Parameter(rank_embeds)  # (num_ranks, max_num_tokens_per_rank, embeddings_dim)
        assert (
            len(rank_embeds) == num_base_ranks
        ), f"len(rank_embeds) {len(rank_embeds)} == num_base_ranks {num_base_ranks}"

        # psudo sentence tokens
        psudo_sentence_tokens = self.create_psudo_sentence_tokens(
            num_tokens_per_rank, num_context_tokens, num_ranks, num_tokens_per_classname, len(classnames)
        )  # (num_ranks, clip_max_num_tokens)
        self.register_buffer("psudo_sentence_tokens", psudo_sentence_tokens, persistent=False)

        self.num_context_tokens = num_context_tokens
        self.num_tokens_per_rank = num_tokens_per_rank
        self.num_tokens_per_classname = num_tokens_per_classname
        if rank_tokens_position not in self.rank_tokens_position_candidates:
            raise ValueError(f"Invalid rank_tokens_position: {rank_tokens_position}")
        self.rank_tokens_positon = rank_tokens_position
        self.num_ranks = num_ranks
        self.num_classnames = len(classnames)
        self.embeddings_dim = clip_model.token_embedding.embedding_dim

        self.create_sentence_embeds_template(clip_model, num_ranks, len(classnames), psudo_sentence_tokens) # 补上开头和结尾的token

        self.create_interpolation_weights(num_base_ranks, num_ranks, interpolation_type, dtype)
        self.num_base_ranks = num_base_ranks

    def create_classname_embeds(
        self, clip_model, classnames, logger, dtype
    ):

        _classname_tokens = [clip._tokenizer.encode(classname) for classname in classnames]
        _num_tokens_per_classname = [len(classname_token) for classname_token in _classname_tokens]
        # logger.info(f"num_tokens_per_classname: {_num_tokens_per_classname}")
        num_tokens_per_classname = _num_tokens_per_classname
        max_num_tokens_per_rank = np.max(num_tokens_per_classname)

        classname_tokens = torch.zeros(len(_classname_tokens), max_num_tokens_per_rank, dtype=torch.long)
        for i, classname_token in enumerate(_classname_tokens):
            classname_tokens[i, : len(classname_token)] = torch.LongTensor(classname_token)
        classname_tokens = clip_model.token_embedding(classname_tokens).type(dtype)
        classname_tokens = classname_tokens[:, :max_num_tokens_per_rank]

        return (classname_tokens, _num_tokens_per_classname)


    def create_psudo_sentence_tokens(self, num_tokens_per_rank, num_context_tokens, num_ranks, num_tokens_per_classname, num_classname):
        psudo_sentence_tokens = torch.zeros(num_ranks * num_classname, self.clip_max_num_tokens, dtype=torch.long)

        if isinstance(num_tokens_per_rank, List):
            assert num_ranks * num_classname == len(num_tokens_per_rank) * len(num_tokens_per_classname)
            for i, (_num_token_per_rank, _num_token_per_classname) in enumerate(product(num_tokens_per_rank, num_tokens_per_classname)):
                sentence_length = 1 + num_context_tokens + _num_token_per_rank + _num_token_per_classname + 1 + 1
                psudo_sentence_tokens[i, :sentence_length] = torch.arange(0, sentence_length, dtype=torch.long)
                
            # for i, _num_tokens_per_rank in enumerate(num_tokens_per_rank):
            #     # <sot>, <context_0>, ..., <context_N>, <rank_i>, <full_stop>, <eot>
            #     sentence_length = 1 + num_context_tokens + _num_tokens_per_rank + 1 + 1
            #     psudo_sentence_tokens[i, :sentence_length] = torch.arange(0, sentence_length, dtype=torch.long)
        else:
            # <sot>, <context_0>, ..., <context_N>, <rank_i>, <full_stop>, <eot>
            sentence_length = 1 + num_context_tokens + num_tokens_per_rank + 1 + 1
            psudo_sentence_tokens[:, :sentence_length] = torch.arange(0, sentence_length, dtype=torch.long)
        return psudo_sentence_tokens


    def create_interpolation_weights(self, num_base_ranks, num_ranks, interpolation_type, dtype):
        if interpolation_type not in self.interpolation_functions:
            raise ValueError(f"Invalide interpolation_type: {interpolation_type}")
        interpolation_func = self.interpolation_functions[interpolation_type]

        interpolation_weights = torch.arange(num_ranks)[..., None].repeat(1, num_base_ranks).to(dtype)
        if num_base_ranks == 1:
            base_interpolation_weights = torch.linspace(0, num_ranks - 1, 3)[1:2].to(dtype)
        else:
            base_interpolation_weights = torch.linspace(0, num_ranks - 1, num_base_ranks).to(dtype)
        interpolation_weights = torch.abs(interpolation_weights - base_interpolation_weights[None])
        interpolation_weights = interpolation_func(interpolation_weights, num_ranks)
        interpolation_weights = interpolation_weights / interpolation_weights.sum(dim=-1, keepdim=True)
        self.register_buffer("interpolation_weights", interpolation_weights, persistent=False)


    def forward(self):
        # context_embeds: (num_ranks, num_context_tokens, embeds_dim)
        context_embeds = self.context_embeds

        # rank_embeds: (num_ranks, max_num_tokens_per_rank, embeddings_dim)
        # if context_embeds.dim() == 2:
            # context_embeds = context_embeds[None].expand(self.num_ranks * self.num_classnames, *context_embeds.shape)
        rank_embeds = torch.sum(self.interpolation_weights[..., None, None] * self.rank_embeds[None, ...], dim=1)
        classname_embeds = self.classname_embeds
        # sentence_embeds: (num_ranks, self.clip_max_num_tokens, embeddings_dim)
        sentence_embeds = self.sentence_embeds.clone()
        if self.rank_tokens_positon == "tail":
            for i in range(self.num_ranks):
                _num_tokens_per_rank = self.num_tokens_per_rank[i]
                pure_sentence_length = self.num_context_tokens + _num_tokens_per_rank
                sentence_embeds[i, 1 : 1 + pure_sentence_length] = torch.cat(
                    [context_embeds[i], rank_embeds[i, :_num_tokens_per_rank]], dim=0
                )
        elif self.rank_tokens_positon == "front":
            for i in range(self.num_ranks):
                _num_tokens_per_rank = self.num_tokens_per_rank[i]
                pure_sentence_length = self.num_context_tokens + _num_tokens_per_rank
                sentence_embeds[i, 1 : 1 + pure_sentence_length] = torch.cat(
                    [rank_embeds[i, :_num_tokens_per_rank], context_embeds[i]], dim=0
                )
        elif self.rank_tokens_positon == "middle":
            for i, (j, k) in enumerate(product(range(self.num_ranks), range(self.num_classnames))):
                _num_tokens_per_rank = self.num_tokens_per_rank[j]
                _num_tokens_per_classname = self.num_tokens_per_classname[k]
                # _context_embeds = context_embeds[i]           
                _context_embeds = context_embeds   
                pure_sentence_length = self.num_context_tokens + _num_tokens_per_rank + _num_tokens_per_classname
                half_range = self.num_context_tokens // 2
                sentence_embeds[i, 1 : 1 + pure_sentence_length] = torch.cat(
                    [
                        _context_embeds[:half_range],
                        classname_embeds[k, :_num_tokens_per_classname],
                        _context_embeds[half_range:],
                        rank_embeds[j, :_num_tokens_per_rank]
                    ],
                    dim=0,
                )

        return sentence_embeds

    # def forward(self):
    #     # context_embeds: (num_ranks, num_context_tokens, embeds_dim)
    #     context_embeds = self.context_embeds

    #     # rank_embeds: (num_ranks, max_num_tokens_per_rank, embeddings_dim)
    #     if context_embeds.dim() == 2:
    #         context_embeds = context_embeds[None].expand(self.num_ranks, *context_embeds.shape)
    #     rank_embeds = torch.sum(self.interpolation_weights[..., None, None] * self.rank_embeds[None, ...], dim=1)

    #     # sentence_embeds: (num_ranks, self.clip_max_num_tokens, embeddings_dim)
    #     sentence_embeds = self.sentence_embeds.clone()
    #     if self.rank_tokens_positon == "tail":
    #         for i in range(self.num_ranks):
    #             _num_tokens_per_rank = self.num_tokens_per_rank[i]
    #             pure_sentence_length = self.num_context_tokens + _num_tokens_per_rank
    #             sentence_embeds[i, 1 : 1 + pure_sentence_length] = torch.cat(
    #                 [context_embeds[i], rank_embeds[i, :_num_tokens_per_rank]], dim=0
    #             )
    #     elif self.rank_tokens_positon == "front":
    #         for i in range(self.num_ranks):
    #             _num_tokens_per_rank = self.num_tokens_per_rank[i]
    #             pure_sentence_length = self.num_context_tokens + _num_tokens_per_rank
    #             sentence_embeds[i, 1 : 1 + pure_sentence_length] = torch.cat(
    #                 [rank_embeds[i, :_num_tokens_per_rank], context_embeds[i]], dim=0
    #             )
    #     elif self.rank_tokens_positon == "middle":
    #         for i in range(self.num_ranks):
    #             _num_tokens_per_rank = self.num_tokens_per_rank[i]
    #             pure_sentence_length = self.num_context_tokens + _num_tokens_per_rank
    #             _context_embeds = context_embeds[i]
    #             half_range = self.num_context_tokens // 2
    #             sentence_embeds[i, 1 : 1 + pure_sentence_length] = torch.cat(
    #                 [
    #                     _context_embeds[:half_range],
    #                     rank_embeds[i, :_num_tokens_per_rank],
    #                     _context_embeds[half_range:],
    #                 ],
    #                 dim=0,
    #             )

    #     return sentence_embeds
