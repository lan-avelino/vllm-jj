# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-scoped scale scratch for serialized B12X expert execution."""

import weakref
from dataclasses import dataclass

import torch

from vllm.config import get_current_vllm_config


@dataclass
class ScaleScratch:
    scope: weakref.ReferenceType
    buffers: tuple[torch.Tensor, torch.Tensor]


# Prepared layers own these buffers. Weak entries cannot extend a model's
# lifetime; checking the scope reference also handles recycled object IDs.
_scratch: weakref.WeakValueDictionary = weakref.WeakValueDictionary()


def get_scale_scratch(w1, w2, group_size, moe_parallel_config):
    config = get_current_vllm_config()
    parallel = config.parallel_config
    moe = moe_parallel_config
    if (
        parallel.pipeline_parallel_size != 1
        or parallel.use_ubatching
        or moe.use_ep
        or moe.ep_size != 1
        or moe.dp_size != 1
        or moe.use_all2all_kernels
        or moe.enable_eplb
    ):
        raise NotImplementedError(
            "Online CSF compression requires TP with PP1/DP1, no EP or ubatching"
        )
    shapes = (
        (w1.shape[0], w1.shape[1], w1.shape[2] * 2 // group_size),
        (w2.shape[0], w2.shape[1], w2.shape[2] * 2 // group_size),
    )
    key = (id(config), w1.device, shapes)
    owner = _scratch.get(key)
    if owner is None or owner.scope() is not config:
        owner = ScaleScratch(
            weakref.ref(config),
            tuple(torch.empty(s, dtype=torch.uint8, device=w1.device) for s in shapes),
        )
        _scratch[key] = owner
    return owner
