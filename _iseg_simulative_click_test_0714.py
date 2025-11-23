from dataset.datasets import load_data_volume
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import argparse
from torch.optim import AdamW
import numpy as np
import logging
from utils.script_util import save_checkpoint
import sys
from monai.losses import DiceCELoss, DiceLoss
import ast
import torch.nn.functional as F
from modeling.Med_SAM.mask_decoder_mamba_kmeans import MaskDecoder
import torch
from modeling.Med_SAM.prompt_encoder_simple import PromptEncoderS
from utils.click_encoding import DistMaps
import torch.nn as nn
from functools import partial
import os
from utils.util import setup_logger, count_parameters
import time
import random
from utils.online_utils import *
from utils.click_utils import get_click_batch
from monai.metrics import HausdorffDistanceMetric
import matplotlib.pyplot as plt
from pathlib import Path

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_confidence_interval(data, confidence_level=0.95):
    mean = np.mean(data)
    std_dev = np.std(data)
    n = len(data)
    if confidence_level == 0.95:
        z_value = 1.96
    elif confidence_level == 0.99:
        z_value = 2.576
    elif confidence_level == 0.90:
        z_value = 1.645
    else:
        raise ValueError("Unsupported confidence level")
    margin_of_error = z_value * (std_dev / np.sqrt(n))
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
    return lower_bound, upper_bound

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", default=None, type=str, choices=["sam", "baidu", "tri_attn_loraAdapter_pEncodeS_miniDe"]
    )
    parser.add_argument(
        "--pretrained", action="store_true"
    )
    parser.add_argument(
        "--data", default=None, type=str
    )
    parser.add_argument(
        "--use_ft_weight", default='no', type=str
    )
    parser.add_argument(
        "--save_result", default='no', type=str
    )
    parser.add_argument(
        "--snapshot_path",
        default="",
        type=str,
    )
    parser.add_argument(
        "--load_weight",
        default="",
        type=str,
    )
    parser.add_argument(
        "--data_prefix",
        default="",
        type=str,
    )
    parser.add_argument(
        "--rand_crop_size",
        default=0,
        nargs='+', type=int,
    )
    parser.add_argument(
        "--input_image_size",
        default=256,
        type=int,
    )
    parser.add_argument(
        "--num_click",
        default=20,
        type=int,
    )
    parser.add_argument(
        "--device",
        default="cuda:1",
        type=str,
    )
    parser.add_argument("-bs", "--batch_size", default=1, type=int)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--lr", default=4e-4, type=float)
    parser.add_argument("--max_epoch", default=500, type=int)
    parser.add_argument("--eval_interval", default=4, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("-tolerance", default=5, type=int)

    args = parser.parse_args()

    if args.method == "tri_attn_loraAdapter_pEncodeS_miniDe":
        from modeling.Med_SAM.image_encoder_tri_attn_Lora_Adapter_kmeans import ImageEncoderViT_3d_v2 as ImageEncoderViT_3d
        import loralib as lora
    else:
        raise "unknown method"
    input_image_size = args.input_image_size
    device = args.device
    if args.rand_crop_size == 0:
        if args.data in ["kits", "colon"]:
            args.rand_crop_size = (256, 256, 256)
        if args.data in ["pancreas", "lits", "brain", "hepatic", "kits23"]:
            args.rand_crop_size = (128, 128, 128)
    else:
        if len(args.rand_crop_size) == 1:
            args.rand_crop_size = tuple(args.rand_crop_size * 3)
        else:
            args.rand_crop_size = tuple(args.rand_crop_size)
    if args.use_ft_weight == 'no':
        #args.weight_path = os.path.join(args.snapshot_path, "['lung','lung2','Lung421','pancreas','kits23','hepatic']")
        args.weight_path = os.path.join(args.snapshot_path, "['total_spleen','total_pancreas','total_lung_upper_lobe_right','total_kidney_right']")
    elif args.use_ft_weight == 'yes':
        args.weight_path = os.path.join(args.snapshot_path, args.data)
    else:
        raise "Error"
    args.snapshot_path = os.path.join(args.snapshot_path, args.data)
    if not os.path.exists(args.snapshot_path):
        os.makedirs(args.snapshot_path)

    setup_logger(logger_name="train", root=args.snapshot_path, screen=True, tofile=True)
    logger = logging.getLogger(f"train")
    logger.info(str(args))

    args.data = ast.literal_eval(args.data)
    args.data_prefix = []
    for dataset_name in args.data:
        args.data_prefix.append(f"datafile/{dataset_name}")
    print(args.data_prefix)

    val_data = load_data_volume(
        datas=args.data,
        path_prefixes=args.data_prefix,
        batch_size=1,
        augmentation=True,
        split="test",
        deterministic=True,
        rand_crop_spatial_size=args.rand_crop_size
    )
    
    if args.load_weight=="original":
        sam = sam_model_registry["vit_b"](checkpoint="ckpt/sam_vit_b_01ec64.pth")
    elif args.load_weight=="medsam":
        sam = sam_model_registry["vit_b"](checkpoint="ckpt/medsam_vit_b.pth")
    else:
        raise "Unknown pretrain weight."
    logger.info(f'Using pretrained weight: {args.load_weight}')

    mask_generator = SamAutomaticMaskGenerator(sam)
    img_encoder = ImageEncoderViT_3d(
        depth=12,
        embed_dim=768,
        img_size=1024,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=12,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=[2, 5, 8, 11],
        window_size=14,
        out_chans=256,
        num_slice = 16,
        cluster_layers=(4, 8),  
        num_clusters=64  
    )

    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean", get_not_nans=True)
    if args.method == "tri_attn_loraAdapter_pEncodeS_miniDe":
        lora.mark_only_lora_as_trainable(img_encoder)
        logger.info(f'LORA The img_encoder model has {count_parameters(img_encoder):,}M trainable parameters.')
        
        img_encoder.depth_embed.requires_grad = True
        for p in img_encoder.slice_embed.parameters():
            p.requires_grad = True
        for i in img_encoder.blocks:
            for p in i.norm1.parameters():
                p.requires_grad = True
            for p in i.norm2.parameters():
                p.requires_grad = True
        for p in img_encoder.neck.parameters():
            p.requires_grad = True
    else:
        raise "wtf network are you used?"

    load_pretrained = args.pretrained
    file = "best_debug.pth.tar"
    #file = "last_debug.pth.tar"
    
    print("load_pretrained", load_pretrained)
    if not load_pretrained:
        pretrained_dict = mask_generator.predictor.model.image_encoder.state_dict()
        model_dict = img_encoder.state_dict()
        filtered_dict = {k: v for k, v in pretrained_dict.items() 
                 if not k.endswith(('rel_pos_h', 'rel_pos_w'))}
        model_dict.update(filtered_dict)
        img_encoder.load_state_dict(model_dict, strict=False)
        print("Loaded pretrained weights (excluding position encodings)")
    else:
        pretrained_dict = torch.load(os.path.join(args.weight_path, file), map_location='cpu', weights_only=False)["encoder_dict"]
        model_dict = img_encoder.state_dict()
        filtered_dict = {k: v for k, v in pretrained_dict.items() 
                 if not k.endswith(('rel_pos_h', 'rel_pos_w'))}
        model_dict.update(filtered_dict)
        img_encoder.load_state_dict(model_dict, strict=False)
        print("Loaded pretrained weights (excluding position encodings)")
    del sam
    img_encoder.to(device)
    
    prompt_encoder = PromptEncoderS(32)
    if load_pretrained:
        prompt_encoder.load_state_dict(
            torch.load(os.path.join(args.weight_path, file), map_location='cpu', weights_only=False)["feature_dict"], strict=True)
    prompt_encoder.to(device)
    
    mask_decoder = MaskDecoder()
    
    if load_pretrained:
        mask_decoder.load_state_dict(torch.load(os.path.join(args.weight_path, file), map_location='cpu', weights_only=False)["decoder_dict"],
                            strict=False)
    mask_decoder.to(device)

    logger.info(f'The img_encoder model has {count_parameters(img_encoder):,}M trainable parameters.')
    logger.info(f'The mask_decoder model has {count_parameters(mask_decoder):,}M trainable parameters.')

    dice_loss = DiceLoss(include_background=False, softmax=True, to_onehot_y=True, reduction="none")
    patch_size = args.rand_crop_size[0]
    debug_time = True

    loss_summary = []
    hd95_overview = []
    loss_overview = []
    test_loss_history = []
    
    dis_map = DistMaps(2, use_disks=True)

    img_encoder.eval()
    prompt_encoder.eval()
    mask_decoder.eval()
    
    if debug_time:
        batch_end = time.perf_counter()
        
    os.makedirs(f"{args.data}", exist_ok=True)

    for idx, (img, seg, spacing, *_rest) in enumerate(val_data):
        
        if debug_time:
            batch_start = time.perf_counter()
            print("data loading spend time", batch_start - batch_end)
            logger.info(f"data loading {idx} time: {batch_end - batch_start:.4f} sec")
        
        out = F.interpolate(img.float(), scale_factor=input_image_size / patch_size, mode='trilinear')  
        input_batch = out.to(device)
        batchsize, d_channel, d_slice, d_h, d_w = input_batch.shape
        input_batch = input_batch.permute(0, 2, 1, 3, 4).reshape(-1, d_channel, d_h, d_w)
        
        seg = seg.to(device)
        pos_clicks_label, neg_clicks_label, pos_disk, neg_disk = get_random_click(seg, args.num_click, dis_map, device)
        print(pos_clicks_label)
        pos_clicks_label = torch.cat(pos_clicks_label, 0)
        neg_clicks_label = torch.cat(neg_clicks_label, 0)
        
        pos_disk = torch.cat(pos_disk, 0).to(device).long()
        neg_disk = torch.cat(neg_disk, 0).to(device).long()
        
        use_click = True
        if use_click:
            click_num = random.randint(1,10)
            this_batch_points_feature = []
            
            assert len(seg) == 1
            
            for _seg in seg:
                _seg = _seg.unsqueeze(0)
                assert _seg.shape[-1] == 128
                
                l = len(torch.where(_seg == 1)[0])
                if l > 0:
                    np.random.seed(2024)
                    sample = np.random.choice(np.arange(l), click_num, replace=True)
                    x = torch.where(_seg == 1)[1][sample].unsqueeze(1)
                    y = torch.where(_seg == 1)[2][sample].unsqueeze(1)
                    z = torch.where(_seg == 1)[3][sample].unsqueeze(1)
                    points_pos = torch.cat([x, y, z], dim=1).float()#.to(device)
                    positive_feat = dis_map.get_coord_features(points_pos, 1, 128, 128, 128)
                else:
                    print("no target")
                    positive_feat = torch.zeros(1, 1, 128, 128, 128).to(device)

                l = len(torch.where(_seg == 0)[0])
                np.random.seed(2024)
                sample = np.random.choice(np.arange(l), click_num, replace=True)
                x = torch.where(_seg == 0)[1][sample].unsqueeze(1)
                y = torch.where(_seg == 0)[2][sample].unsqueeze(1)
                z = torch.where(_seg == 0)[3][sample].unsqueeze(1)
                points_neg = torch.cat([x, y, z], dim=1).float()#.to(device)
                negative_feat = dis_map.get_coord_features(points_neg, 1, 128, 128, 128)
                
                this_batch_points_feature.append(
                    torch.cat([positive_feat, negative_feat], dim=1)
                )
            prompt_input = torch.cat(this_batch_points_feature, 0).float()
            
        else:
            raise "use click"
            prompt_input = torch.zeros(batchsize, 2, 128, 128, 128).to(device)

        with torch.no_grad():
            point_feature = prompt_encoder(prompt_input)
            batch_features = img_encoder(input_batch, batchsize, point_feature)  # bs, C, H, W, D
            masks = mask_decoder(batch_features)
        masks = masks.to(device)
        masks = masks.permute(0, 1, 4, 2, 3)  # bs, C, D, H, W   [2, 2, 128, 128, 128]
        
        seg = seg.unsqueeze(1)
        loss = dice_loss(masks, seg)
        print("original loss", round(float(loss), 5))

        for _sim_idx in range(40):
            masks_for_click = F.softmax(masks.clone().detach().cpu(), dim=1)[:,1] > 0.5
            seg_for_click = seg.clone().squeeze(1).detach().cpu()
            click_pos, is_positive = get_click_batch(masks_for_click, seg_for_click)
            #print(click_pos, is_positive)  # D, H, W
            click_pos = click_pos.to(device)
            
            if is_positive[0]:
                points_pos = torch.cat([points_pos, click_pos], dim = 0)
            else:
                points_neg = torch.cat([points_neg, click_pos], dim = 0)
            
            new_positive_feat = dis_map.get_coord_features(points_pos, 1, 128, 128, 128)
            new_negative_feat = dis_map.get_coord_features(points_neg, 1, 128, 128, 128)
            
            prompt_input = torch.cat([new_positive_feat, new_negative_feat], dim=1)
            
            with torch.no_grad():
                point_feature = prompt_encoder(prompt_input)
                batch_features = img_encoder(input_batch, batchsize, point_feature)  # bs, C, H, W, D
                masks = mask_decoder(batch_features)
            masks = masks.permute(0, 1, 4, 2, 3)  # bs, C, D, H, W   [2, 2, 128, 128, 128]
            loss = dice_loss(masks, seg)
            print(_sim_idx, "loss", round(float(loss), 5))
        
        pos_disk = dis_map.get_coord_features(points_pos, 1, 128, 128, 128)
        neg_disk = dis_map.get_coord_features(points_neg, 1, 128, 128, 128)
        
        if debug_time:
            batch_end = time.perf_counter() 
            print("batch spend time", batch_end - batch_start)
            logger.info(f"Batch {idx} time: {batch_end - batch_start:.4f} sec")

        if args.save_result == 'yes':
            torch.save(
                {
                    "img": img.detach().cpu(),
                    "predict": masks.detach().cpu(), # bs, C, D, H, W
                    "GT": seg.detach().cpu(), # bs, 1, D, H, W
                    "pos_clicks": points_pos.detach().cpu(),
                    "neg_clicks": points_neg.detach().cpu(),
                    "pos_disk": pos_disk.detach().cpu(),
                    "neg_disk": neg_disk.detach().cpu(),
                    "dice_score": round(float(loss), 5),
                }
                , f"{args.data}/data_{idx}.pt")
        
    
        loss_summary.append(loss.detach().cpu().numpy())
        logger.info(
            'iter: {}/{}'.format(idx, len(val_data)) + ": loss:" + str(
                loss_summary[-1].flatten()[0]))
        
        masks = F.softmax(masks, dim=1)#[:, 1]
        masks = masks > 0.5
        hd95 = hd95_metric(masks, seg)

        loss_overview.append(loss.mean().detach().cpu().numpy())
        if not torch.isnan(hd95):
            hd95_overview.append(hd95.mean().detach().cpu().numpy())
   
    logger.info("- Val metrics: " + str(np.mean(loss_summary)))

    print("dice score")
    dice_score = 1 - sum(loss_overview) / len(loss_overview)
    print(dice_score)
    logger.info(f"- Dice score: {dice_score:.4f}")
    dice_lower, dice_upper = calculate_confidence_interval(loss_overview)
    print(f"loss_summary: ({round(dice_lower, 4)}, {round(dice_upper, 4)})")
    logger.info(f"- Dice CI: ({round(1 - dice_upper, 4)}, {round(1 - dice_lower, 4)})")

    print("hd95")
    hd95_score = sum(hd95_overview) / len(hd95_overview)
    print(hd95_score)
    logger.info(f"- HD95: {hd95_score:.4f}")
    hd95_lower, hd95_upper = calculate_confidence_interval(hd95_overview)
    print(f"hd95_overview: ({round(hd95_lower, 4)}, {round(hd95_upper, 4)})")
    logger.info(f"- hd95_score CI: ({round(hd95_lower, 4)}, {round(hd95_upper, 4)})")

if __name__ == "__main__":
    set_seed(42)
    main()