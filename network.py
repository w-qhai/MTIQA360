from clip import clip
import torch
from torch import nn
from collections import OrderedDict
from typing import Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from itertools import product
from prompt_leaners.rank_prompt_learner import RankPromptLearner

distortion_types = ['jpeg', 'j2k', 'avc', 'hevc', 'noise', 'blur', 'other']


class v9(nn.Module): # sota
    def __init__(self):
        super().__init__()
        self.predtrained, self.preprocess = clip.load("ViT-B/32", device="cpu", jit=False)
        self.predtrained.float()
        self.predtrained.requires_grad_(False)
        self.predtrained.visual.requires_grad_(False)

        self.num_ranks = 10
        self.prompt_learner = RankPromptLearner(self.predtrained, distortion_types, num_base_ranks=5, num_ranks=self.num_ranks, num_tokens_per_rank=2,
                                                num_context_tokens=16, rank_tokens_position="middle", init_rank_path=None)
        self.psudo_sentence_tokens = self.prompt_learner.psudo_sentence_tokens
        self.num_ranks = self.prompt_learner.num_ranks
            
        self.weigth = nn.Parameter(torch.arange(1, self.num_ranks + 1, dtype=torch.float).unsqueeze(1), requires_grad=False)

    def forward(self, x, text=None):
        batch_size = x.size(0)
        num_viewport = x.size(1)


        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        if self.predtrained.visual.training:
            self.predtrained.eval()
            
        image_features = self.predtrained.encode_image(x)
        # image_features = self.res(x)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        
        prompts = self.prompt_learner()
        psudo_sentence_tokens = self.psudo_sentence_tokens
        text_features = self.predtrained.encode_text_coco(prompts, psudo_sentence_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logit_scale = self.predtrained.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        
        logits_per_image = logits_per_image.view(batch_size, num_viewport, -1)
        logits_per_image = logits_per_image.mean(1)
        
        logits_per_image = F.softmax(logits_per_image, dim=1)
        logits_per_image = logits_per_image.view(-1, self.num_ranks, len(distortion_types))
        
        
        logits_level = logits_per_image.sum(2)
        logits_distortion = logits_per_image.sum(1)
        
        logits_quality = logits_level @ self.weigth
        logits_quality = logits_quality.squeeze(1)
        return logits_quality, logits_distortion


class MTIQA360(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = v9()
        
    def forward(self, image, text):
        res = self.model(image, text)
        return res