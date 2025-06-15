#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import numpy as np
import os
import time
import pdb

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig
from .modeling_llama import LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from .stamp_infer_utils import get_gsop_trajectory

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        self.prefilling_latency = []
        self.sample_latency = []
        self.visual_token_num = None

    def get_model(self):
        return self.model
    
    def get_visual_token_num(self):
        return self.visual_token_num
    
    def apply_pruning_policy(self, cur_policy, prune_config):
        self.model.prune_config = prune_config
        self.model.img_cls_vtokens = prune_config.img_cls_vtokens
        self.model.random_vtokens = prune_config.random_vtokens
        if prune_config.img_cls_vtokens:
            self.visual_token_num = prune_config.token_nums[1]
        
        if prune_config.mha_in_scale_path is None:
            mha_in_scale = np.ones((len(self.model.layers), 3))
        else:
            mha_in_scale = np.load(prune_config.mha_in_scale_path)
        mha_out_scale = np.ones((len(self.model.layers), 3))
        if prune_config.mlp_scale_path is None:
            mlp_scale = np.ones((len(self.model.layers), 3))
        else:
            mlp_scale = np.load(prune_config.mlp_scale_path)
        scale_nps = [mha_out_scale, mha_in_scale, mlp_scale]
        sys_cached_pos = []
        
        for pos in cur_policy:
            idl, idg, ido = pos
            idg_maping = {0:0,1:1,2:2}
            if len(prune_config.token_groups) > 3:
                # print(greedy_config.token_groups)
                idg_maping = {0:0,len(prune_config.token_groups)-1:2}
                for idig in range(1,len(prune_config.token_groups)-1):
                    idg_maping[idig] = 1
            scale = scale_nps[ido][idl][idg_maping[idg]]
            if idg == 0 and ido in [1, 2]:
                sys_cached_pos.append((idl, idg, ido))
            else:
                self.skip_group_op(idl, idg, ido, scale)
        print("target_pruning_strategy:", cur_policy)
        print("using cache fo sys kq:", sys_cached_pos)
    
    def skip_group_op(self, idl, idg, ido, pl_scale = 1.00):
        self.model.layers[idl].skip_group_op.append([idg, ido, pl_scale])
    
    def clean_pruning(self):
        for layer in self.model.layers:
            layer.skip_group_op = []
    
    def activate_iatt_grouping(self, img_grouping_config, random_vtokens = False, img_cls_vtokens = False):
        self.model.img_grouping_config = img_grouping_config
        self.model.random_vtokens = random_vtokens
        self.model.img_cls_vtokens = img_cls_vtokens
        if img_cls_vtokens:
            self.visual_token_num = int(img_grouping_config.img_group_ratios[0] * img_grouping_config.img_length)

    def prune_model_infer(self, prune_config):
        gsop_trajectory = get_gsop_trajectory(prune_config.prune_config_path, prune_config)
        for step in sorted(list(gsop_trajectory.keys())):
            print(step, gsop_trajectory[step][0])
            gsop_ratio = gsop_trajectory[step][0]
            if gsop_ratio < prune_config.desire_prune_ratio:
                gsop_policy = gsop_trajectory[step][1]
                break
        # pdb.set_trace()
        
        self.apply_pruning_policy(gsop_policy, prune_config)
        # pdb.set_trace()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                token_indices,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
            self.model.cls_token_indices = token_indices
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        # start_time = time.time()

        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        # end_time = time.time()
        end_event.record()
        torch.cuda.synchronize()
        latency_ms = start_event.elapsed_time(end_event)
        # cal_time = end_time - start_time
        # self.cur_latency.append(latency_ms)
        # pdb.set_trace()
        if position_ids.shape[1] == 1:
            self.sample_latency[-1] += latency_ms
        else:
            self.prefilling_latency.append(latency_ms)
            print("average latency, prefilling:", np.mean(self.prefilling_latency), "sample:",np.mean(self.sample_latency))
            self.sample_latency.append(latency_ms)
        return output

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                token_indices
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
            self.model.cls_token_indices = token_indices
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
