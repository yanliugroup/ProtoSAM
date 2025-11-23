import torch
import torch.nn as nn
import torch.nn.functional as F

class ClusterAwareUpsampling(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor, num_clusters):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels + num_clusters, out_channels, 3, padding=1), 
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, cluster_assign):
        cluster_feat = F.interpolate(
            cluster_assign, 
            scale_factor=self.scale_factor, 
            mode='trilinear'
        )
        x = F.interpolate(
            x, 
            size=cluster_feat.shape[2:], 
            mode='trilinear'
        )
        x = torch.cat([x, cluster_feat], dim=1) 
        x = self.conv(x)
        return x

class MaskDecoder(nn.Module):
    def __init__(self, mlahead_channels=256, num_classes=2, num_clusters=64, output_size=128):
        super().__init__()
        self.num_clusters = num_clusters
        self.mlahead_channels = mlahead_channels
        self.output_size = output_size

        self.cluster_projection = nn.Sequential(
            nn.Conv3d(mlahead_channels, num_clusters, 1),
            nn.Softmax(dim=1))
        
        self.branch0 = ClusterAwareUpsampling(mlahead_channels, mlahead_channels//2, 
                                           scale_factor=1, num_clusters=num_clusters)
        self.branch1 = ClusterAwareUpsampling(mlahead_channels, mlahead_channels//2, 
                                           scale_factor=2, num_clusters=num_clusters)
        self.branch2 = ClusterAwareUpsampling(mlahead_channels, mlahead_channels//2, 
                                           scale_factor=4, num_clusters=num_clusters)

        self.fusion = nn.Sequential(
            nn.Conv3d(mlahead_channels*3//2, mlahead_channels, 3, padding=1),
            nn.InstanceNorm3d(mlahead_channels),
            nn.ReLU(inplace=True))

        self.final = nn.Sequential(
            nn.Conv3d(mlahead_channels, mlahead_channels//2, 1),
            nn.InstanceNorm3d(mlahead_channels//2),
            nn.ReLU(inplace=True),
            nn.Upsample(size=output_size, mode='trilinear', align_corners=True),
            nn.Conv3d(mlahead_channels//2, num_classes, 1))

        self.morph_kernel = torch.ones(3, 3, 3) 

    def forward(self, input):

        cluster_assign = self.cluster_projection(input)  # [B, K, H, W, D]

        x0 = self.branch0(input, cluster_assign)
        x1 = self.branch1(input, cluster_assign)
        x2 = self.branch2(input, cluster_assign)

        target_size = (input.size(2)*4, input.size(3)*4, input.size(4)*4)
        x0 = F.interpolate(x0, size=target_size, mode='trilinear', align_corners=True)
        x1 = F.interpolate(x1, size=target_size, mode='trilinear', align_corners=True)
        x2 = F.interpolate(x2, size=target_size, mode='trilinear', align_corners=True)

        x = torch.cat([x0, x1, x2], dim=1)
        x = self.fusion(x)

        x = self.final(x)

        if self.training:
            return x  
        else:
            binary_mask = (torch.sigmoid(x) > 0.5)
            binary_mask = binary_mask.float()
            closed_mask = self._morphological_close(binary_mask)
            x = closed_mask * x  
            return x
            
    def _morphological_close(self, x):
        kernel = self.morph_kernel.to(x.device)[None,None].repeat(x.shape[1], 1, 1, 1, 1)  # [2,1,3,3,3]

        padded = F.pad(x, (1,1,1,1,1,1), mode='constant', value=0)
        dilated = F.conv3d(padded, kernel, groups=x.shape[1], padding=0) > 0

        padded = F.pad(dilated.float(), (1,1,1,1,1,1), mode='constant', value=1)
        eroded = F.conv3d(padded, kernel, groups=x.shape[1], padding=0) == 27
    
        return eroded.float()