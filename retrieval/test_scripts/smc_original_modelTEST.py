#!/usr/bin/env python3
# coding: utf-8
# below codes are based and adapted from https://github.com/XinhaoMei/WavCaps/blob/master/retrieval/train.py

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"


# only record main process logs on wandb

import time
from pprint import PrettyPrinter
import wandb
import torch
import argparse
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

WB_LOG = False



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

    exp_name = exp_name

    if is_main_process() and WB_LOG:
        wandb.init(
            project="SMC",
            name=exp_name,
            config=config,
            # group="Mel",
            group="DAC",
            # group="EnCodec",  # all runs for the experiment in one group
            mode='offline'
        )

    # self_add create BASELINE dataloader
    clotho_datamodule = AudioCaptionDataModule(config, "Clotho")
    dataloader = clotho_datamodule.train_dataloader(is_distributed=is_dist_avail_and_initialized(),
                                                    num_tasks=get_world_size(),
                                                    global_rank=get_rank())

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
    scheduler = cosine_lr(optimizer,
                          base_lr=config["optim_args"]["lr"],
                          warmup_length=config["optim_args"]["warmup_epochs"] * len(dataloader),
                          steps=len(dataloader) * config["training"]["epochs"])
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
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            find_unused_parameters=True #ONLY USED IN UNFREEZE HTSAT MODEL
        )                
        model_without_ddp = model.module
        
    if is_main_process() and WB_LOG:
        wandb.watch(model)

    if is_main_process():
        main_logger.info('Evaluation start...')
        clotho_test_loader = clotho_datamodule.test_dataloader()
        model_without_ddp.load_state_dict(torch.load(str(model_output_dir) + "/clotho_best_model.pt")["model"])
        main_logger.info(f"Evaluation best Clotho model... epoch:{torch.load(str(model_output_dir) + '/clotho_best_model.pt')['epoch']}")
        clotho_metrics = validate(model_without_ddp, clotho_test_loader, device) # model_without_ddp
        if WB_LOG:
            log_results_wandb(clotho_metrics, 'Clotho', main_logger, test=True)
            wandb.finish()
        else:
            log_results(clotho_metrics, 'Clotho', main_logger, test=True)
        main_logger.info("Done.")


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    audio_embeds_all, text_embeds_all = [], []
    for batch_idx, (audio, text, idx) in tqdm(enumerate(dataloader), total=len(dataloader)):
        audio = audio.to(device)

        audio_embeds = model.encode_audio(audio)
        text_embeds = model.encode_text(text)

        audio_embeds_all.append(audio_embeds.cpu())
        text_embeds_all.append(text_embeds.cpu())

    audio_embeds_all = torch.cat(audio_embeds_all, dim=0).numpy()
    text_embeds_all = torch.cat(text_embeds_all, dim=0).numpy()

    # evaluate text to audio retrieval
    r1, r5, r10, r50, medr, meanr, mAP10 = t2a(audio_embeds_all, text_embeds_all)

    # evaluate audio to text retrieval
    r1_a, r5_a, r10_a, r50_a, medr_a, meanr_a, mAP10_a = a2t(audio_embeds_all, text_embeds_all)

    return {"t2a": [r1, r5, r10, r50, medr, meanr, mAP10],
            "a2t": [r1_a, r5_a, r10_a, r50_a, medr_a, meanr_a, mAP10_a]}


if __name__ == '__main__':
    main()
