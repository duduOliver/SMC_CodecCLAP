#!/usr/bin/env python3
# coding: utf-8
# below codes are based and adapted from https://github.com/XinhaoMei/WavCaps/blob/master/retrieval/train.py
# Python 1.12 环境下，单机单卡下t2a, at2计算极慢，初步估计可能存在环境不兼容问题；[Pending]

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,4,5,6"
# os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"


# only record main process logs on wandb

import time
from pprint import PrettyPrinter
import wandb
import torch
import argparse
# import ruamel.yaml as yaml
from ruamel.yaml import YAML
from tqdm import tqdm
from loguru import logger
from data_handling.datamodule import AudioCaptionDataModule
from data_handling.pretrain_dataset import pretrain_dataloader
from models.ase_model import ASE
import torch.distributed as dist
from tools.optim_utils import get_optimizer, cosine_lr
from tools.utils import (
    get_rank,
    get_world_size,
    init_distributed_mode,
    is_dist_avail_and_initialized,
    is_main_process,
    setup_seed,
    AverageMeter, t2a, a2t, set_logger, log_results, log_results_wandb
)
import numpy as np
import json
from torch.utils.data import RandomSampler, DistributedSampler

# torch.serialization.add_safe_globals([("numpy._core.multiarray", "scalar")])
# torch.serialization.add_safe_globals(["scalar"])

WB_LOG = False

def train(model, dataloader, optimizer, scheduler, device, epoch, config):
    model.train()
     
    if is_dist_avail_and_initialized():
        if model.module.audio_encoder.config["audio_encoder_args"]["type"] == "dac":
                # Set the codec part to evaluation mode
                model.module.audio_encoder.eval()
        # print(f"model.module.audio_encoder in eval mode: {not model.module.audio_encoder.training}")
        elif model.module.audio_encoder.config["audio_encoder_args"]["type"] == "dac_embedder":
                # Set the codec part to evaluation mode
                model.module.audio_encoder.codes_enc.eval()
        elif model.module.audio_encoder.config["audio_encoder_args"]["type"] == "dac_htsat":
                # Set the codec part to evaluation mode
                # model.module.audio_encoder.codes_enc.to(device)
                model.module.audio_encoder.codes_enc.eval()
        elif model.module.audio_encoder.config["audio_encoder_args"]["type"] == "vamp":
                # Set the codec part to evaluation mode
                # model.module.audio_encoder.codes_enc.to(device)
                model.module.audio_encoder.codes_enc.eval()
        elif model.module.audio_encoder.config["audio_encoder_args"]["type"] == "encodec":
                # Set the codec part to evaluation mode
                # model.module.audio_encoder.codes_enc.to(device)
                model.module.audio_encoder.eval()
    else:
        if model.audio_encoder.config["audio_encoder_args"]["type"] == "dac":
                # Set the codec part to evaluation mode
                model.audio_encoder.eval()
        # print(f"model.audio_encoder in eval mode: {not model.audio_encoder.training}")
        elif model.audio_encoder.config["audio_encoder_args"]["type"] == "dac_embedder":
                # Set the codec part to evaluation mode
                model.audio_encoder.codes_enc.eval()
        elif model.audio_encoder.config["audio_encoder_args"]["type"] == "dac_htsat":
                # Set the codec part to evaluation mode
                # model.audio_encoder.codes_enc.to(device)
                model.audio_encoder.codes_enc.eval()
        elif model.audio_encoder.config["audio_encoder_args"]["type"] == "vamp":
                # Set the codec part to evaluation mode
                # model.audio_encoder.codes_enc.to(device)
                model.audio_encoder.codes_enc.eval()
        elif model.audio_encoder.config["audio_encoder_args"]["type"] == "encodec":
                # Set the codec part to evaluation mode
                # model.audio_encoder.codes_enc.to(device)
                model.audio_encoder.eval()

    epoch_loss = AverageMeter()
    start_time = time.time()

    if is_dist_avail_and_initialized():
        dataloader.sampler.set_epoch(epoch)

    accumulation_steps = config.get("gradient_accumulation_steps", 1)
    
    # 初始化性能监控
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    end = time.time()
    
    optimizer.zero_grad()
    
    for batch_id, (audio, text, idx) in tqdm(enumerate(dataloader), total=len(dataloader)):
        # 测量数据加载时间
        data_time.update(time.time() - end)
        
        step = len(dataloader) * (epoch - 1) + batch_id
        scheduler(step)
        if is_main_process() and WB_LOG:
            wandb.log({"lr": optimizer.param_groups[0]["lr"]}, step=step)

        audio = audio.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)

        # 计算损失
        loss = model(audio, text, idx)
        loss = loss / accumulation_steps  # 梯度累积

        # 反向传播
        loss.backward()

        # 梯度累积
        if (batch_id + 1) % accumulation_steps == 0:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("max_grad_norm", 1.0))
            
            # 更新参数
            optimizer.step()
            optimizer.zero_grad()

            # 记录损失
            losses.update(loss.item() * accumulation_steps, audio.size(0))
            
            # 测量批处理时间
            batch_time.update(time.time() - end)
            end = time.time()

            # # 定期记录训练状态
            # if batch_id % config.get("log_interval", 10) == 0 and is_main_process():
            #     logger.info(f'Epoch: [{epoch}][{batch_id}/{len(dataloader)}]\t'
            #               f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #               f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            #               f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
            #               f'LR {optimizer.param_groups[0]["lr"]:.6f}')

        epoch_loss.update(loss.detach().cpu().item() * accumulation_steps)

    elapsed_time = time.time() - start_time

    if is_main_process() and WB_LOG:
        # 记录更多训练指标
        wandb.log({
            "loss": epoch_loss.avg,
            "epoch": epoch,
            "gpu_memory": torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "batch_time": elapsed_time / len(dataloader)
        })

    return {
        "loss": epoch_loss.avg,
        "time": elapsed_time
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="settings/baseline.yaml", type=str,
                        help="Setting files")
    parser.add_argument("-n", "--exp_name", default="exp_name", type=str,
                        help="name of this experiment.")
    parser.add_argument("-l", "--lr", default=5e-5, type=float,
                        help="Learning rate.")
    parser.add_argument("-t", "--model_type", default="cnn", type=str,
                        help="Model type.")
    parser.add_argument("-m", "--model", default="Cnn14", type=str,
                        help="Model name.")
    parser.add_argument("-a", "--max_length", default=30, type=int,
                        help="Max length.")
    parser.add_argument("-s", "--batch_size", default=128, type=int,
                        help="Batch size.")
    parser.add_argument("-b", "--blacklist", default='blacklist_exclude_ub8k_esc50_vggsound.json', type=str,
                        help="Blacklist file.")
    args = parser.parse_args()

    exp_name = args.exp_name

    with open(args.config, "r") as f:
        # config = yaml.safe_load(f)
        yaml = YAML(typ='safe', pure=True)
        config = yaml.load(f)

    # setup distribution mode
    init_distributed_mode(config["dist_args"])
    device = torch.device(config["device"])

    #self_add commented get_rank(), then add it to exp_name
    # setup seed
    seed = config["seed"] + get_rank()
    setup_seed(seed)

    # exp_name = exp_name

    if is_main_process() and WB_LOG:
        wandb.init(
            project="preCLAP",
            name=exp_name,
            config=config,
            group="Mel",
            # group="DAC",
            # group="EnCodec",  # all runs for the experiment in one group
            mode='offline'
        )

    # 优化数据加载和预处理
    clotho_datamodule = AudioCaptionDataModule(config, "Clotho")
        
    # 创建训练数据加载器
    dataloader = clotho_datamodule.train_dataloader(
        is_distributed=is_dist_avail_and_initialized(),
        num_tasks=get_world_size(),
        global_rank=get_rank(),
    )
    
    # 优化数据加载器配置
    if is_main_process():
        logger.info(f"DataLoader configuration:")
        logger.info(f"  - num_workers: {dataloader.num_workers}")
        logger.info(f"  - batch_size: {dataloader.batch_size}")
        logger.info(f"  - pin_memory: {dataloader.pin_memory}")
        logger.info(f"  - prefetch_factor: {dataloader.prefetch_factor}")
    
    if is_main_process():
        logger.info(f"Initialized dataloader with {len(dataloader.dataset)} samples")

    # setup model
    model = ASE(config)
    model = model.to(device)

    # setup optim utils
    optimizer = get_optimizer(model.parameters(),
                              lr=config["optim_args"]["lr"],
                              betas=config["optim_args"]["betas"],
                              eps=config["optim_args"]["eps"],
                              momentum=config["optim_args"]["momentum"],
                              optimizer_name=config["optim_args"]["optimizer_name"])
    # 改进的学习率调度器
    scheduler = cosine_lr(
        optimizer,
        base_lr=config["optim_args"]["lr"],
        warmup_length=config["optim_args"]["warmup_epochs"] * len(dataloader),
        steps=len(dataloader) * config["training"]["epochs"],
        # min_lr=config["optim_args"].get("min_lr", 1e-6),  # 最小学习率
        # cycle_mult=config["optim_args"].get("cycle_mult", 1.0),  # 周期倍增因子
        # restart_decay=config["optim_args"].get("restart_decay", 0.5)  # 重启衰减
    )
    
    # 添加学习率监控
    if is_main_process():
        logger.info(f"Initialized learning rate scheduler with:")
        logger.info(f"  - base_lr: {config['optim_args']['lr']}")
        logger.info(f"  - warmup_epochs: {config['optim_args']['warmup_epochs']}")
        # logger.info(f"  - min_lr: {config['optim_args'].get('min_lr', 1e-6)}")
        logger.info(f"  - total_steps: {len(dataloader) * config['training']['epochs']}")
    start_epoch = 1
    max_epoch = config["training"]["epochs"]

    if config["resume"]:
        cp = torch.load(config.checkpoint, map_location="cpu")
        state_dict = cp["model"]

        optimizer.load_state_dict(cp["optimizer"])
        start_epoch = cp["epoch"] + 1
        model.load_state_dict(state_dict)

    # setup logger
    model_output_dir, log_output_dir = set_logger(exp_name)

    main_logger = logger.bind(indent=1)

    # print training settings
    printer = PrettyPrinter()
    main_logger.info('Training setting:\n'
                     f'{printer.pformat(config)}')

    main_logger.info(f'Total numer of parameters: {sum([i.numel() for i in model.parameters()])}')
    main_logger.info(f'Size of training set: {len(dataloader.dataset)}, size of batches: {len(dataloader)}')

    model_without_ddp = model
    if is_dist_avail_and_initialized():
        # 检查所有进程是否准备好
        dist.barrier()
        
        # 优化分布式训练设置
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[get_rank()],
            output_device=get_rank(),
            # find_unused_parameters=True, #ONLY USED IN UNFREEZE HTSAT MODEL
            gradient_as_bucket_view=True,  # 优化内存使用
            static_graph=True  # 提高训练效率
        )
        model_without_ddp = model.module
        
        # 设置分布式训练参数
        torch.cuda.set_device(get_rank())
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        if is_main_process():
            main_logger.info(f"Initialized DDP with {get_world_size()} processes")
        
        # 再次同步以确保所有进程完成初始化
        dist.barrier()
        if is_main_process():
            main_logger.info("All processes completed DDP initialization")
        
    if is_main_process() and WB_LOG:
        wandb.watch(model)

    clotho_val_loader = clotho_datamodule.val_dataloader(
        is_distributed=is_dist_avail_and_initialized(),
        num_tasks=get_world_size(),
        global_rank=get_rank(),
    )
    
    # clotho_val_loader = clotho_datamodule.val_dataloader()
    
    clotho_test_loader = clotho_datamodule.test_dataloader(
        is_distributed=is_dist_avail_and_initialized(),
        num_tasks=get_world_size(),
        global_rank=get_rank(),
    )
    
    # clotho_test_loader = clotho_datamodule.test_dataloader()
    
    loss_stats = []
    ac_recall_stats = []
    clotho_recall_stats = []
    for epoch in range(start_epoch, max_epoch + 1):
        main_logger.info(f'Training for epoch [{epoch}]')

        train_statics = train(model, dataloader, optimizer, scheduler, device, epoch, config)
        loss = train_statics["loss"]
        elapsed_time = train_statics["time"]
        loss_stats.append(loss)

        main_logger.info(f'Training statistics:\tloss for epoch [{epoch}]: {loss:.3f},'
                         f'\ttime: {elapsed_time:.1f}, lr: {optimizer.param_groups[0]["lr"]:.6f}.')
        if is_dist_avail_and_initialized():
            dist.barrier()
            torch.cuda.empty_cache()
        
        main_logger.info("Validating model...")
        clotho_metrics = validate(model, clotho_val_loader, device, 
                                world_size=get_world_size(), rank=get_rank(),
                                ddp=is_dist_avail_and_initialized()
                                )
        if is_main_process():
            if WB_LOG:
                log_results_wandb(clotho_metrics, 'Clotho', main_logger, test=False)
            else:
                log_results(clotho_metrics, 'Clotho', main_logger, test=False)
        clotho_recall_stats.append(clotho_metrics["t2a"][0] + clotho_metrics["a2t"][0])
        if clotho_recall_stats[-1] >= max(clotho_recall_stats) and is_main_process():
            # 保存完整checkpoint
            checkpoint = {
                "epoch": epoch,
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                # "scheduler": scheduler.state_dict(),
                # "scaler": None,
                "config": config,
                "best_metric": clotho_recall_stats[-1],
                "distributed": is_dist_avail_and_initialized(),
                "world_size": get_world_size() if is_dist_avail_and_initialized() else 1
            }
            
            # 保存最佳checkpoint
            best_path = str(model_output_dir) + "/clotho_best_model.pt"
            torch.save(checkpoint, best_path)
            main_logger.info(f"New best model saved at epoch {epoch} with metric {clotho_recall_stats[-1]:.4f} to {best_path}")
                
            # # 保存额外的元数据
            # meta_path = str(model_output_dir) + "/clotho_metadata.json"
            # with open(meta_path, "w") as f:
            #     json.dump({
            #         "epoch": epoch,
            #         "metric": clotho_recall_stats[-1],
            #         "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            #         "config": config
            #     }, f, indent=2)

    # # 在所有epoch完成后检查并处理最佳模型
    # if is_main_process():
    #     best_model_path = str(model_output_dir) + "/clotho_best_model.pt"
    #     latest_model_path = str(model_output_dir) + "/clotho_latest.pt"
        
    #     # 如果不存在最佳模型，则将最新模型复制为最佳模型
    #     if not os.path.exists(best_model_path):
    #         import shutil
    #         shutil.copy(latest_model_path, best_model_path)
    #         main_logger.info(f"Copied latest model to best model at {best_model_path}")

    main_logger.info('Evaluation start...')
    if is_dist_avail_and_initialized():
        dist.barrier()
    # model_without_ddp.load_state_dict(torch.load(str(model_output_dir) + "/clotho_best_model.pt", weights_only=True)["model"])
    checkpoint = torch.load(str(model_output_dir) + "/clotho_best_model.pt", map_location="cpu")
    model_without_ddp.load_state_dict(checkpoint["model"], strict=False)

    if is_dist_avail_and_initialized():
        dist.barrier()
    # print('let me see if go here') 
    # model_without_ddp.load_state_dict(torch.load(str(model_output_dir) + "/clotho_best_model.pt")["model"])
    
    if is_dist_avail_and_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model_without_ddp,
            find_unused_parameters=False #ONLY USED IN UNFREEZE HTSAT MODEL
        )               
    
    main_logger.info(f"Evaluation best Clotho model... epoch:{torch.load(str(model_output_dir) + '/clotho_best_model.pt', map_location="cpu")['epoch']}")
    # main_logger.info(f"Evaluation best Clotho model... epoch:{torch.load(str(model_output_dir) + '/clotho_best_model.pt')['epoch']}")
    clotho_metrics = validate(model, clotho_test_loader, device, 
                              world_size=get_world_size(), rank=get_rank(), 
                              ddp=is_dist_avail_and_initialized()) # model_without_ddp
    
    # if is_dist_avail_and_initialized():
    #     dist.destroy_process_group()
    
    if is_main_process():
        if WB_LOG:
            log_results_wandb(clotho_metrics, 'Clotho', main_logger, test=True)
            wandb.finish()
        else:
            log_results(clotho_metrics, 'Clotho', main_logger, test=True)
        main_logger.info("Done.")
        # wandb.finish()


'''
1. DistributedSampler： 对于samples的分配上，在sample 总数无法被进程数（GPUs）整除时，如果drop_last=False, 会对每个进程分配的sample数量进行向上求整；如果drop_last=True，则删除余数sample，最终使得每条进程分配到的sample数量还是一样的。另外，由于各个进程上分配的sample包含部分的，某个audio*5的一部分，而其gather的时候，没有按照合理顺序拼接，所以在算t2a, at2时，隔5取audio会有问题；
    a. 实际的drop_last指令在sub datasets里进行，所以sub samples 也需要能尽除 batch size;
    b. 而这对于Clotho数据集的retrieval 计算带来麻烦，目前t2a, at2计算方法不适用；
2. Clotho 数据集需要用 DDP模式的validate，需要在 5 GPUs 资源下使用；
    a. 还是不行，观察到了更进一步的问题。audio数据是按照发牌顺序发，各个gpu轮着发的，而不是发完一个再发下一个。
3. 如果找到原因，修改正确了，comment out datamodule中的(val/test)_dataloader，使用ddp模式；
'''

def validate(model, dataloader, device, world_size=1, rank=0, ddp=False):
    model.eval()
    audio_embeds_all, text_embeds_all = None, None  # 初始化为 None 

    # 使用 DistributedSampler 来确保每个进程处理不同的数据 [测试验证环节，不需要shuffle]
    if ddp and world_size > 1:
        dataloader.sampler.set_epoch(0)  # 设置 epoch 以确保每个进程的数据分布一致
        print(f"Rank {rank} is processing {len(dataloader)} batches")

    with torch.no_grad():
        for batch_idx, (audio, text, idx) in tqdm(enumerate(dataloader), total=len(dataloader)):
            audio = audio.to(device)
            
            if ddp:
                audio_embeds = model.module.encode_audio(audio)
                text_embeds = model.module.encode_text(text)
            else:
                audio_embeds = model.encode_audio(audio)
                text_embeds = model.encode_text(text)
                
            if audio_embeds_all is None:  # 初始化
                audio_embeds_all = audio_embeds
                text_embeds_all = text_embeds
            else:
                audio_embeds_all = torch.cat((audio_embeds_all, audio_embeds), dim=0)
                text_embeds_all = torch.cat((text_embeds_all, text_embeds), dim=0)

    # 处理数据同步
    if world_size > 1:
        # 检查所有进程是否准备好同步
        dist.barrier()
        
        # 计算总数据量
        local_audio_size = audio_embeds_all.size(0)
        local_text_size = text_embeds_all.size(0)
        
        print(f'Rank {rank}: local_audio_size shape: {local_audio_size}')
        print(f'Rank {rank}: local_text_size shape: {local_text_size}')
        
        # 收集所有进程的本地数据大小
        all_audio_sizes = torch.tensor([local_audio_size], dtype=torch.int64, device=device)
        all_text_sizes = torch.tensor([local_text_size], dtype=torch.int64, device=device)
        
        print(f'Rank {rank}: all_audio_sizes shape: {all_audio_sizes}')
        print(f'Rank {rank}: all_text_sizes shape: {all_text_sizes}')
        
        dist.all_reduce(all_audio_sizes, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_text_sizes, op=dist.ReduceOp.SUM)
        
        total_audio_size = all_audio_sizes.item()
        total_text_size = all_text_sizes.item()
        
        print(f'Rank {rank}: total_audio_size shape: {total_audio_size}')
        print(f'Rank {rank}: total_text_size shape: {total_text_size}')
        
        # 分配输出张量
        all_audio_embeds = torch.empty((total_audio_size, *audio_embeds_all.shape[1:]), dtype=audio_embeds_all.dtype, device=device)
        all_text_embeds = torch.empty((total_text_size, *text_embeds_all.shape[1:]), dtype=text_embeds_all.dtype, device=device)
        
        print(f'Rank {rank}: all_audio_embeds shape: {all_audio_embeds.size()}')
        print(f'Rank {rank}: all_text_embeds shape: {all_text_embeds.size()}')
        
        # 使用 all_gather_into_tensor 收集数据
        torch.distributed.all_gather_into_tensor(all_audio_embeds, audio_embeds_all)
        torch.distributed.all_gather_into_tensor(all_text_embeds, text_embeds_all)
        
        # 将结果展平为一个大的张量（实际上已经展平）
        audio_embeds_all = all_audio_embeds
        text_embeds_all = all_text_embeds
    else:
        # 非分布式模式直接使用原始数据
        pass

    print(f'Rank {rank}: audio_embeds_all shape: {audio_embeds_all.shape}')
    print(f'Rank {rank}: text_embeds_all shape: {text_embeds_all.shape}')

    # evaluate text to audio retrieval
    r1, r5, r10, r50, medr, meanr, mAP10 = t2a(audio_embeds_all.cpu(), text_embeds_all.cpu())

    # evaluate audio to text retrieval
    r1_a, r5_a, r10_a, r50_a, medr_a, meanr_a, mAP10_a = a2t(audio_embeds_all.cpu(), text_embeds_all.cpu())
    return {"t2a": [r1, r5, r10, r50, medr, meanr, mAP10],
            "a2t": [r1_a, r5_a, r10_a, r50_a, medr_a, meanr_a, mAP10_a]}


if __name__ == '__main__':
    main()