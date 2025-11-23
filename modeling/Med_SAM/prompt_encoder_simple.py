# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from typing import Any, Optional, Tuple, Type

from torch import Tensor, nn

class PromptEncoder_bug(nn.Module):
    def __init__(
        self,
        init_dim: int = 32
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(2, init_dim, kernel_size=3, padding=1),
            nn.InstanceNorm3d(init_dim),
            nn.ReLU(),
            nn.Conv3d(init_dim, init_dim * 2, kernel_size=3, padding=1, stride=2),
            nn.InstanceNorm3d(init_dim * 2),
            nn.ReLU(),
            nn.Conv3d(init_dim * 2, init_dim * 2, kernel_size=3, padding=1),
            nn.InstanceNorm3d(init_dim * 2),
            nn.ReLU(),
            nn.Conv3d(init_dim * 2, init_dim * 4, kernel_size=3, padding=1, stride=2),
            nn.InstanceNorm3d(init_dim * 4),
            nn.ReLU(),
            nn.Conv3d(init_dim * 4, init_dim * 4, kernel_size=3, padding=1),
            nn.InstanceNorm3d(init_dim * 4),
            nn.ReLU(),
            nn.Conv3d(init_dim * 4, init_dim * 8, kernel_size=3, padding=1, stride=2),
            nn.InstanceNorm3d(init_dim * 8),
            nn.ReLU(),
            nn.Conv3d(init_dim * 8, 768, kernel_size=1),
            nn.InstanceNorm3d(768),
            nn.ReLU(),
        )

    
    def forward(self, points_feat) :
        
        return self.encoder(points_feat)

class PromptEncoder(nn.Module):
    def __init__(
        self,
        init_dim: int = 32
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(2, init_dim, kernel_size=3, padding=1),
            nn.InstanceNorm3d(init_dim),
            nn.ReLU(),
            nn.Conv3d(init_dim, init_dim * 2, kernel_size=3, padding=1, stride=2),
            nn.InstanceNorm3d(init_dim * 2),
            nn.ReLU(),
            nn.Conv3d(init_dim * 2, init_dim * 2, kernel_size=3, padding=1),
            nn.InstanceNorm3d(init_dim * 2),
            nn.ReLU(),
            nn.Conv3d(init_dim * 2, init_dim * 4, kernel_size=3, padding=1, stride=2),
            nn.InstanceNorm3d(init_dim * 4),
            nn.ReLU(),
            nn.Conv3d(init_dim * 4, init_dim * 4, kernel_size=3, padding=1),
            nn.InstanceNorm3d(init_dim * 4),
            nn.ReLU(),
            nn.Conv3d(init_dim * 4, init_dim * 8, kernel_size=3, padding=1, stride=2),
            nn.InstanceNorm3d(init_dim * 8),
            nn.ReLU(),
            nn.Conv3d(init_dim * 8, 768, kernel_size=1),
            nn.InstanceNorm3d(768),
            nn.ReLU(),
        )

    
    def forward(self, points_feat) :
        
        return self.encoder(points_feat)


from ..unetr_pp.model_components import UnetrPPEncoderS
class PromptEncoderS(nn.Module):
    def __init__(
        self,
        init_dim: int = 32
    ) -> None:
        super().__init__()
        patch_size = [4, 4, 4]
        dims=[32, 64, 128, 256]
        depths=[3, 3, 3, 3]
        num_heads: int = 4
        img_size = [128, 128, 128]
        self.feat_size = (
            img_size[0]
            // patch_size[0]
            // 8,  # 8 is the downsampling happened through the four encoders stages
            img_size[1]
            // patch_size[1]
            // 8,  # 8 is the downsampling happened through the four encoders stages
            img_size[2]
            // patch_size[2]
            // 8,  # 8 is the downsampling happened through the four encoders stages
        )
        self.unetr_pp_encoder = UnetrPPEncoderS(
            input_size=[
                self.feat_size[0] * self.feat_size[1] * self.feat_size[2] * (2 ** (3 - i)) ** 3
                for i in range(4)
            ],
            spatial_dims=3,
            patch_size=patch_size,
            dims=dims,
            depths=depths,
            in_channels=2,
            num_heads=num_heads,
        )
        
        self.last_mapping = nn.Sequential(
            nn.Conv3d(64, 768, kernel_size=1),
            nn.InstanceNorm3d(768),
            nn.ReLU()
        )

    
    def forward(self, points_feat):

        x_output, hidden_states = self.unetr_pp_encoder(points_feat)
        x_output = self.last_mapping(x_output)

        return x_output




class PromptEncoder_h(nn.Module):
    def __init__(
        self,
        init_dim: int = 32
    ) -> None:
        super().__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv3d(2, init_dim, kernel_size=3, padding=1),
            nn.InstanceNorm3d(init_dim),
            nn.ReLU(),
            nn.Conv3d(init_dim, init_dim * 2, kernel_size=3, padding=1, stride=2),
            nn.InstanceNorm3d(init_dim * 2),
            nn.ReLU(),
            nn.Conv3d(init_dim * 2, init_dim * 2, kernel_size=3, padding=1),
            nn.InstanceNorm3d(init_dim * 2),
            nn.ReLU()
            )
            
        self.encoder2 = nn.Sequential(
            nn.Conv3d(init_dim * 2, init_dim * 4, kernel_size=3, padding=1, stride=2),
            nn.InstanceNorm3d(init_dim * 4),
            nn.ReLU(),
            nn.Conv3d(init_dim * 4, init_dim * 4, kernel_size=3, padding=1),
            nn.InstanceNorm3d(init_dim * 4),
            nn.ReLU()
            )
        
        self.encoder3 = nn.Sequential(
            nn.Conv3d(init_dim * 4, init_dim * 8, kernel_size=3, padding=1, stride=2),
            nn.InstanceNorm3d(init_dim * 8),
            nn.ReLU(),
            nn.Conv3d(init_dim * 8, init_dim * 8, kernel_size=3, padding=1),
            nn.InstanceNorm3d(init_dim * 8),
            nn.ReLU()
        )

        self.last = nn.Sequential(
            nn.Conv3d(init_dim * 8, 768, kernel_size=1),
            nn.InstanceNorm3d(768),
            nn.ReLU()
        )
    
    def forward(self, points_feat):
        ret_feats = []
        points_feat = self.encoder1(points_feat)
        ret_feats.append(points_feat)
        points_feat = self.encoder2(points_feat)
        ret_feats.append(points_feat)
        points_feat = self.encoder3(points_feat)
        ret_feats.append(points_feat)
        points_feat = self.last(points_feat)
        return points_feat, ret_feats