from dataset.datasets import load_data_volume
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry # type: ignore
import argparse
from torch.optim import AdamW
import numpy as np
import logging
from utils.script_util import save_checkpoint
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
from sklearn.cluster import KMeans # type: ignore

from modeling.Med_SAM.image_encoder_tri_attn_Lora_Adapter_kmeans import DynamicClusterBlock

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_confidence_interval(data, confidence_level):
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

def init_cluster_centers(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        batch = next(iter(data_loader))
        img, seg, _ = batch[:3]
        img = batch[0].to(device)

        out = F.interpolate(img.float(), scale_factor=256/128, mode='trilinear')  
        input_batch = out.to(device)
        batchsize, d_channel, d_slice, d_h, d_w = input_batch.shape
        input_batch = input_batch.permute(0, 2, 1, 3, 4).reshape(-1, d_channel, d_h, d_w)

        model = model.to(device)
        features = model.patch_embed(input_batch)
        flat_features = features.reshape(-1, features.size(-1)).cpu().numpy()

        cluster_block = next(b for b in model.blocks if isinstance(b, DynamicClusterBlock))
        prototypes = cluster_block.attn.prototypes

        kmeans = KMeans(n_clusters=model.num_clusters)
        kmeans.fit(flat_features)
        prototypes.prototypes = torch.from_numpy(kmeans.cluster_centers_).float().to(device)

        for module in model.modules():
            if hasattr(module, 'centers'):
                module.centers.data = torch.from_numpy(kmeans.cluster_centers_).float().to(device)


def generate_pseudo_labels(model, unlabeled_imgs, seg, confidence_thresh=0.9):
    model.eval()
    with torch.no_grad():
        input_batch = F.interpolate(unlabeled_imgs.float(), scale_factor=2, mode='trilinear')
        batchsize, d_channel, d_slice, d_h, d_w = input_batch.shape
        input_batch = input_batch.permute(0, 2, 1, 3, 4).reshape(-1, d_channel, d_h, d_w)

        features = model(input_batch, batchsize=batchsize, points_feat=None, labels=seg, pseudo_labels=None)  
        probs = torch.softmax(features, dim=1)
        max_probs, pseudo_labels = torch.max(probs, dim=1, keepdim=True)
        confidence_mask = (max_probs > confidence_thresh).float()
        pseudo_labels = pseudo_labels * confidence_mask

        edge_mask = F.avg_pool3d(max_probs, kernel_size=3, stride=1, padding=1) - \
                   F.avg_pool3d(max_probs, kernel_size=3, stride=1, padding=1)
        edge_mask = (edge_mask.abs() > 0.1).float()
        confidence_mask = confidence_mask * (1 - edge_mask) 

    return pseudo_labels, confidence_mask

def main():
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument(   
        "--method", default=None, type=str, choices=["sam", "sam3d", "baidu", "tri_attn_loraAdapter_pEncodeS_miniDe"]
    )
    parser.add_argument(    
        "--pretrained", action="store_true"
    )
    parser.add_argument(    
        "--data", default=None, type=str
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
        "--device",
        default="cuda:3",
        type=str,
    )
    parser.add_argument("-bs", "--batch_size", default=1, type=int)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--lr", default=4e-4, type=float)
    parser.add_argument("--max_epoch", default=500, type=int)
    parser.add_argument("--eval_interval", default=4, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("-tolerance", default=5, type=int)
    parser.add_argument("--num_clusters", default=32, type=int, help="Number of clusters")

    args = parser.parse_args()
    if args.method == "tri_attn_loraAdapter_pEncodeS_miniDe":
        from modeling.Med_SAM.image_encoder_tri_attn_Lora_Adapter_kmeans import ImageEncoderViT_3d_v2 as ImageEncoderViT_3d
        import loralib as lora  # type: ignore
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
            
    args.weight_path = os.path.join(args.snapshot_path, "['lung','lung2','Lung421','pancreas','kits23','hepatic']")
    args.snapshot_path = os.path.join(args.snapshot_path, args.data)
    if not os.path.exists(args.snapshot_path):
        os.makedirs(args.snapshot_path)
    setup_logger(logger_name="train", root=args.snapshot_path, screen=True, tofile=True)
    logger = logging.getLogger(f"train")
    logger.info(str(args))
    args.data = ast.literal_eval(args.data)
    args.data_prefix = [f"datafile/{dataset_name}" for dataset_name in args.data]
    print(args.data_prefix)
    train_data = load_data_volume(
        datas=args.data,
        path_prefixes=args.data_prefix,
        batch_size=args.batch_size,
        augmentation=True,
        split="train",
        rand_crop_spatial_size=args.rand_crop_size
    )
    val_data = load_data_volume(
        datas=args.data,
        path_prefixes=args.data_prefix,
        batch_size=args.batch_size,
        augmentation=False,
        split="val",
        deterministic=True,
        rand_crop_spatial_size=args.rand_crop_size
    )

    if args.load_weight=="medsam":
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

    # Choose training weights
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
        init_cluster_centers(img_encoder, train_data, device)
        print("Loaded pretrained weights (excluding position encodings)")
    else:
        pretrained_dict = torch.load(os.path.join(args.weight_path, file), map_location='cpu', weights_only=False)["encoder_dict"]
        model_dict = img_encoder.state_dict()
        filtered_dict = {k: v for k, v in pretrained_dict.items() 
                 if not k.endswith(('rel_pos_h', 'rel_pos_w'))}
        model_dict.update(filtered_dict)
        img_encoder.load_state_dict(model_dict, strict=False)
        init_cluster_centers(img_encoder, train_data, device)
        print("Loaded pretrained weights (excluding position encodings)")
    del sam    
    img_encoder.to(device)
    prompt_encoder = PromptEncoderS(32)
    if load_pretrained:     # 
        prompt_encoder.load_state_dict(
            torch.load(os.path.join(args.weight_path, file), map_location='cpu', weights_only=False)["feature_dict"], strict=True)
    prompt_encoder.to(device)

    mask_decoder = MaskDecoder()
    if load_pretrained:
        mask_decoder.load_state_dict(torch.load(os.path.join(args.weight_path, file), map_location='cpu', weights_only=False)["decoder_dict"],
                            strict=True)
    mask_decoder.to(device)
    logger.info(f'The img_encoder model has {count_parameters(img_encoder):,}M trainable parameters.')
    logger.info(f'The mask_decoder model has {count_parameters(mask_decoder):,}M trainable parameters.')
    encoder_opt = AdamW(img_encoder.parameters(), lr=args.lr, weight_decay=0)
    encoder_scheduler = torch.optim.lr_scheduler.LinearLR(encoder_opt, start_factor=1.0, end_factor=0.01, total_iters=args.max_epoch)
    feature_opt = AdamW(prompt_encoder.parameters(), lr=args.lr, weight_decay=0)
    feature_scheduler = torch.optim.lr_scheduler.LinearLR(feature_opt, start_factor=1.0, end_factor=0.01,
                                                          total_iters=args.max_epoch)
    decoder_opt = AdamW([i for i in mask_decoder.parameters() if i.requires_grad == True], lr=args.lr, weight_decay=0)
    decoder_scheduler = torch.optim.lr_scheduler.LinearLR(decoder_opt, start_factor=1.0, end_factor=0.01, total_iters=args.max_epoch)
    dice_loss = DiceLoss(include_background=False, softmax=True, to_onehot_y=True, reduction="none")
    loss_cal = DiceCELoss(include_background=False, softmax=True, to_onehot_y=True, lambda_dice=0.5, lambda_ce=0.5)
    best_loss = np.inf
    patch_size = args.rand_crop_size[0]
    debug_time = True
    debug = False
    dis_map = DistMaps(2, use_disks=True)
    
    train_loss_history = []
    val_loss_history = []

    for epoch_num in range(args.max_epoch):
        loss_summary = []
        hd95_overview = []

        img_encoder.train()
        prompt_encoder.train()
        mask_decoder.train()

        if debug_time:
            batch_end = time.perf_counter()  

        for idx, (img, seg, spacing) in enumerate(train_data):
            if debug_time:
                batch_start = time.perf_counter()  
                print("data loading spend time", batch_start - batch_end)
                logger.info(f"data loading {idx} time: {batch_end - batch_start:.4f} sec")
            
            organ_types = torch.zeros(img.size(0), dtype=torch.long, device=device)  

            if idx <= 400:
                pseudo = False
            else:
                pseudo = True

            if pseudo is False:
                img = img.to(device)
                seg = seg.to(device)
                out = F.interpolate(img.float(), scale_factor=input_image_size / patch_size, mode='trilinear')  # TODO: 256 input
                input_batch = out.to(device)
                batchsize, d_channel, d_slice, d_h, d_w = input_batch.shape
                input_batch = input_batch.permute(0, 2, 1, 3, 4).reshape(-1, d_channel, d_h, d_w)

                use_click = 1
                if use_click:  
                    click_num = random.randint(1, 10)
                    # randomly sample click points
                    this_batch_points_feature = []  
                    for _seg in seg:
                        _seg = _seg.unsqueeze(0)
                        assert _seg.shape[-1] == 128
                        l = len(torch.where(_seg == 1)[0])
                        if l > 0:
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
                    #print("do not use click")
                    prompt_input = torch.zeros(batchsize, 2, 128, 128, 128).to(device)
            
                point_feature = prompt_encoder(prompt_input)
                batch_features = img_encoder(input_batch, batchsize, point_feature, labels=seg, pseudo_labels=None)  # bs, C, H, W, D

            if pseudo is True:
                img = img.to(device)
                seg = seg.to(device)
                pseudo_labels, confidence_mask = generate_pseudo_labels(
                    img_encoder, 
                    img,
                    seg,
                    confidence_thresh=0.5
                )
                out = F.interpolate(img.float(), scale_factor=input_image_size / patch_size, mode='trilinear')  # TODO: 256 input
                input_batch = out.to(device)
                batchsize, d_channel, d_slice, d_h, d_w = input_batch.shape
                input_batch = input_batch.permute(0, 2, 1, 3, 4).reshape(-1, d_channel, d_h, d_w)
                use_click = 1
                if use_click:   
                    click_num = random.randint(1, 10)
                    # randomly sample click points
                    this_batch_points_feature = [] 
                    for _seg in seg:
                        _seg = _seg.unsqueeze(0)
                        assert _seg.shape[-1] == 128
                        l = len(torch.where(_seg == 1)[0])
                        if l > 0:
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
                    #print("do not use click")
                    prompt_input = torch.zeros(batchsize, 2, 128, 128, 128).to(device)
            
                point_feature = prompt_encoder(prompt_input)
                batch_features = img_encoder(input_batch, batchsize, point_feature, labels=None, pseudo_labels=pseudo_labels)

            masks = mask_decoder(batch_features)
            masks = masks.permute(0, 1, 4, 2, 3)  # bs, C, D, H, W
            seg = seg.unsqueeze(1)
            loss = loss_cal(masks, seg)
            loss_summary.append(loss.detach().cpu().numpy())
            encoder_opt.zero_grad()
            decoder_opt.zero_grad()
            feature_opt.zero_grad()
            loss.backward()    
            logger.info(
                'epoch: {}/{}, iter: {}/{}'.format(epoch_num, args.max_epoch, idx, len(train_data)) + ": loss:" + str(
                    loss_summary[-1].flatten()[0]))
            torch.nn.utils.clip_grad_norm_(img_encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(mask_decoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(prompt_encoder.parameters(), 1.0)
            encoder_opt.step()
            feature_opt.step()
            decoder_opt.step()
            
            if debug_time:
                batch_end = time.perf_counter()  
                print("batch spend time", batch_end - batch_start)
                logger.info(f"Batch {idx} time: {batch_end - batch_start:.4f} sec")

        encoder_scheduler.step()
        feature_scheduler.step()
        decoder_scheduler.step()
        logger.info("- Train metrics: " + str(np.mean(loss_summary)))

        train_loss_history.append(np.mean(loss_summary))

        img_encoder.eval()
        prompt_encoder.eval()
        mask_decoder.eval()
        with torch.no_grad():  
            
            for idx, (img, seg, spacing, path) in enumerate(val_data):
                loss_summary = []   
                # print('seg: ', seg.sum())
                out = F.interpolate(img.float(), scale_factor=input_image_size / patch_size, mode='trilinear')  # TODO: for 256
                input_batch = out.to(device)
                batchsize, d_channel, d_slice, d_h, d_w = input_batch.shape
                input_batch = input_batch.permute(0, 2, 1, 3, 4).reshape(-1, d_channel, d_h, d_w)
                seg = seg.to(device)
                # randomly sample click points
                this_batch_points_feature = []
                for _seg in seg:
                    _seg = _seg.unsqueeze(0)
                    assert _seg.shape[-1] == 128
                    l = len(torch.where(_seg == 1)[0])
                    if l > 0:
                        sample = np.random.choice(np.arange(l), 5, replace=True)
                        x = torch.where(_seg == 1)[1][sample].unsqueeze(1)
                        y = torch.where(_seg == 1)[2][sample].unsqueeze(1)
                        z = torch.where(_seg == 1)[3][sample].unsqueeze(1)
                        points_pos = torch.cat([x, y, z], dim=1).float()#.to(device)
                        positive_feat = dis_map.get_coord_features(points_pos, 1, 128, 128, 128)
                    else:
                        print("no target")
                        positive_feat = torch.zeros(1, 1, 128, 128, 128).to(device)
                    l = len(torch.where(_seg == 0)[0])
                    sample = np.random.choice(np.arange(l), 5, replace=True)
                    x = torch.where(_seg == 0)[1][sample].unsqueeze(1)
                    y = torch.where(_seg == 0)[2][sample].unsqueeze(1)
                    z = torch.where(_seg == 0)[3][sample].unsqueeze(1)
                    points_neg = torch.cat([x, y, z], dim=1).float()#.to(device)
                    negative_feat = dis_map.get_coord_features(points_neg, 1, 128, 128, 128)
                    
                    this_batch_points_feature.append(
                        torch.cat([positive_feat, negative_feat], dim=1)
                    )
                prompt_input = torch.cat(this_batch_points_feature, 0).float()
                point_feature = prompt_encoder(prompt_input)
                batch_features = img_encoder(input_batch, batchsize, point_feature)  # bs, C, H, W, D
                masks = mask_decoder(batch_features)
                masks = masks.permute(0, 1, 4, 2, 3)  # bs, C, D, H, W
                seg = seg.unsqueeze(1)
                loss = dice_loss(masks, seg)
                loss_summary.append(loss.detach().cpu().numpy())
                logger.info(
                    'epoch: {}/{}, iter: {}/{}'.format(epoch_num, args.max_epoch, idx, len(val_data)) + ": loss:" + str(
                        loss_summary[-1].flatten()[0]))
                val_loss_history.append(np.mean(loss_summary))
        logger.info("- Val metrics: " + str(np.mean(loss_summary)))
        is_best = False
        if np.mean(loss_summary) < best_loss:
            best_loss = np.mean(loss_summary)
            is_best = True
            
        img_encoder.train()  # very tricky
        # NOTE: The devil lies in the detail:
        # Calling model.eval() will trigger the merging of LoRA parameters with the corresponding pretrained ones, which eliminates additional latency for subsequent forward passes. Calling model.train() again will undo the merge. This can be disabled by passing merge_weights=False to LoRA layers.
        save_checkpoint({"epoch": epoch_num + 1,
                        "best_val_loss": best_loss,
                         "encoder_dict": img_encoder.state_dict(),
                         "decoder_dict": mask_decoder.state_dict(),
                         "feature_dict": prompt_encoder.state_dict(),
                         "encoder_opt": encoder_opt.state_dict(),
                         "feature_opt": feature_opt.state_dict(),
                         "decoder_opt": decoder_opt.state_dict()
                         },
                        is_best=is_best,
                        checkpoint=args.snapshot_path)
        logger.info("- Val metrics best: " + str(best_loss))

if __name__ == "__main__":
    set_seed(42)
    main()