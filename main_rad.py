import argparse
import csv
import os
import logging
import yaml
import numpy as np
import random
import time
import datetime
import json
import math
import shutil
from pathlib import Path
from functools import partial
from sklearn.metrics import roc_auc_score

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler

import pandas as pd

from tensorboardX import SummaryWriter
from transformers import AutoModel,BertConfig,AutoTokenizer

from factory import utils
from scheduler import create_scheduler
from optim import create_optimizer_dual
from engine.train_rad import evaluate_skin_test, valid_on_ICD, train_grad_acc, test_logits

from models.clip_tqn import ModelRes, Text_Encoder_Bert, TQN_Model_fusion, ModelConvNeXt, ModelRes_3D
from models.tokenization_bert import BertTokenizer
from dataset.dataset_entity import ICD_Train_Dataset, Fair_ori_train_dataset, Skin_Train_Dataset, NACC_Train_Dataset
from dataset.test_dataset import ICD_Dataset, Fair_ori_test_dataset, Skin_Test_Dataset, NACC_Test_Dataset


import socket
from io import BytesIO


def seed_torch(seed=42):
    print('=====> Using fixed random seed: ' + str(seed))
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _atomic_torch_save(state, checkpoint_path):
    """Write a checkpoint completely before replacing the previous latest state."""
    checkpoint_path = Path(checkpoint_path)
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + '.tmp')
    torch.save(state, temporary_path)
    os.replace(temporary_path, checkpoint_path)

def _write_epoch_metrics(metrics_dir, record):
    """Append durable history and replace the latest metric snapshot."""
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    history_path = metrics_dir / 'metrics_history.csv'
    write_header = not history_path.exists() or history_path.stat().st_size == 0
    with open(history_path, 'a', newline='', encoding='utf-8') as history_file:
        writer = csv.DictWriter(history_file, fieldnames=record.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(record)
        history_file.flush()
        os.fsync(history_file.fileno())

    latest_path = metrics_dir / 'latest_metrics.json'
    temporary_path = latest_path.with_suffix('.json.tmp')
    with open(temporary_path, 'w', encoding='utf-8') as metrics_file:
        json.dump(record, metrics_file, ensure_ascii=False, indent=2)
        metrics_file.write('\n')
        metrics_file.flush()
        os.fsync(metrics_file.fileno())
    os.replace(temporary_path, latest_path)

def _configure_logging(output_dir):
    logs_dir = Path(output_dir) / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(logs_dir / 'training.log', encoding='utf-8'),
            logging.StreamHandler(),
        ],
        force=True,
    )

def main(args, config):
    if not torch.cuda.is_available():
        raise RuntimeError('RAD training requires a CUDA-enabled PyTorch runtime.')
    torch.cuda.current_device()
    print("Total CUDA devices: ", torch.cuda.device_count()) 
    torch.set_default_tensor_type('torch.FloatTensor')
    
    utils.init_distributed_mode(args)
    
    device = torch.device(args.device)#cuda

    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / 'checkpoints'
    metrics_dir = output_dir / 'metrics'
    tensorboard_dir = output_dir / 'tensorboard'
    for directory in (checkpoints_dir, metrics_dir, output_dir / 'predictions', tensorboard_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if utils.is_main_process():
        _configure_logging(output_dir)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_rank = global_rank
    print('sampler_rank',sampler_rank,'num_tasks',num_tasks)

    train_csv = args.train_csv or config.get('ICD_train_file')
    val_csv = args.val_csv or config.get('ICD_val_file')
    test_csv = args.test_csv or config.get('ICD_test_file')
    image_root = args.image_root or config.get('image_root')
    if not train_csv or not test_csv:
        raise ValueError('Set train/test CSV paths in the YAML configuration or pass --train_csv and --test_csv.')
    if 'skin' in args.dataset and not val_csv:
        raise ValueError('SkinCAP requires a validation CSV. Pass --val_csv.')
    if not image_root:
        raise ValueError('Set image_root in the YAML configuration or pass --image_root.')
    train_num_workers = config.get('num_workers', 2)
    test_num_workers = config.get('test_num_workers', train_num_workers)

    #### Dataset #### 
    print("Creating dataset")
    if 'fair_ori' in args.dataset:
        train_dataset = Fair_ori_train_dataset(train_csv, config['image_res'], image_root)
    elif 'skin' in args.dataset:
        train_dataset = Skin_Train_Dataset(train_csv, config['image_res'], image_root)
    elif 'nacc' in args.dataset:
        train_dataset = NACC_Train_Dataset(train_csv, config['image_res'], image_root)
    else:
        train_dataset = ICD_Train_Dataset(train_csv, config['image_res'], image_root)
    
  
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,num_replicas=num_tasks, rank=sampler_rank, shuffle=True)
    train_dataloader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            num_workers=train_num_workers,
            pin_memory=True,
            sampler=train_sampler, 
            collate_fn=None,
            worker_init_fn=utils.seed_worker,
            drop_last=True,
        )    
    train_dataloader.num_samples = len(train_dataset)
    train_dataloader.num_batches = len(train_dataloader)  

    if 'fair_ori' in args.dataset:
        val_dataset = Fair_ori_test_dataset(val_csv or test_csv, config['image_res'], image_root)
    elif 'skin' in args.dataset:
        val_dataset = Skin_Test_Dataset(val_csv, config['image_res'], image_root)
    elif 'nacc' in args.dataset:
        val_dataset = NACC_Test_Dataset(val_csv or test_csv, config['image_res'], image_root)
    else:
        val_dataset = ICD_Dataset(val_csv or test_csv, config['image_res'], image_root)
    val_sampler = (
        SequentialSampler(val_dataset)
        if 'skin' in args.dataset
        else torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=num_tasks, rank=sampler_rank, shuffle=True,
        )
    )
    val_dataloader = DataLoader(
            val_dataset,
            batch_size=config['test_batch_size'],
            num_workers=test_num_workers,
            pin_memory=True,
            sampler=val_sampler,
            collate_fn=None,
            worker_init_fn=utils.seed_worker,
            drop_last=False,
        )
    val_dataloader.num_samples = len(val_dataset)
    val_dataloader.num_batches = len(val_dataloader)

    skin_label_list = None
    if 'skin' in args.dataset:
        skin_label_list = list(pd.read_csv(train_csv, nrows=0).columns[2:])
        if not skin_label_list:
            raise ValueError(f'No SkinCAP label columns found in {train_csv}.')
        val_label_list = list(pd.read_csv(val_csv, nrows=0).columns[2:])
        if skin_label_list != val_label_list:
            raise ValueError('SkinCAP train and validation label columns must have the same names and order.')

    if 'res' in args.image_encoder_name:
        if 'nacc' in args.dataset:
            resnet_3d_config = {
                'model_type': 'resnet',
                'model_depth': 50,
                'input_W': 96,
                'input_H': 96,
                'input_D': 96,
                'resnet_shortcut': 'B',
                'no_cuda': False,
                'gpu_id': 0,
                'pretrain_path': '',
                'out_feature': args.embed_dim
            }
            image_encoder = ModelRes_3D(resnet_3d_config).cuda()
        else:
            image_encoder = ModelRes(res_base_model=args.image_encoder_name, embed_dim=args.embed_dim).cuda()
    elif 'convnext' in args.image_encoder_name:
        image_encoder = ModelConvNeXt(convnext_base_model=args.image_encoder_name).cuda()
    else:
        raise ValueError('invalid image encoder', args.image_encoder_name)

    if args.bert_model_name:
        tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name,do_lower_case=True, local_files_only=True)
        text_encoder = Text_Encoder_Bert(bert_model_name=args.bert_model_name).cuda()

    model = TQN_Model_fusion(embed_dim=args.embed_dim).cuda()
    model_guideline = TQN_Model_fusion(embed_dim=args.embed_dim).cuda()
    
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer_dual(arg_opt, model, model_guideline, image_encoder, text_encoder)

    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer) 

    checkpoint_path = checkpoints_dir / 'latest.pt'
    best_validation_score = -math.inf
    best_epoch = None
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        checkpoint_name = checkpoint_path.name
    else:
        legacy_checkpoints = [
            path for path in output_dir.glob('checkpoint_*.pt')
            if path.stem.removeprefix('checkpoint_').isdigit()
        ]
        if legacy_checkpoints:
            checkpoint_path = max(legacy_checkpoints, key=lambda path: int(path.stem.removeprefix('checkpoint_')))
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            checkpoint_name = checkpoint_path.name
        else:
            checkpoint = None

    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        model_guideline.load_state_dict(checkpoint['model_guideline'])
        image_encoder.load_state_dict(checkpoint['image_encoder'])
        text_encoder.load_state_dict(checkpoint['text_encoder'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_validation_score = float(checkpoint.get('best_validation_score', -math.inf))
        best_epoch = checkpoint.get('best_epoch')
        logging.info('Resuming training from epoch %s using %s.', start_epoch + 1, checkpoint_name)
    else:
        logging.info('No checkpoint found, starting training from scratch.')

    print("Start training")
    start_time = time.time()
    if utils.is_main_process():
        writer = SummaryWriter(str(tensorboard_dir))


    for epoch in range(start_epoch, max_epoch):
        train_dataloader.sampler.set_epoch(epoch)
        learning_rate = float(optimizer.param_groups[0]['lr'])

        train_stats = train_grad_acc(model, model_guideline, image_encoder, text_encoder, tokenizer, train_dataloader, optimizer, epoch, warmup_steps, device, lr_scheduler, args, config, writer, config['grad_accumulation_steps'], args.guideline_path, skin_label_list)

        train_loss_epoch = float(train_stats.get('loss', 'nan'))
        train_loss_ce_epoch = float(train_stats.get('loss_ce', 'nan'))
        train_loss_clip_epoch = float(train_stats.get('loss_clip', 'nan'))        
        writer.add_scalar('loss/train_loss_epoch', train_loss_epoch, epoch)
        writer.add_scalar('loss/train_loss_ce_epoch', train_loss_ce_epoch, epoch)
        writer.add_scalar('loss/train_loss_clip_epoch', train_loss_clip_epoch, epoch)
        writer.add_scalar('lr/learning_rate', learning_rate, epoch)

        validation_summary = valid_on_ICD(
            model, model_guideline, image_encoder, text_encoder, tokenizer, val_dataloader,
            epoch, device, args, config, args.guideline_path, skin_label_list,
        )

        if utils.is_main_process():
            epoch_number = epoch + 1
            validation_score = float(validation_summary['best_validation_score'])
            is_best = validation_score > best_validation_score
            if is_best:
                best_validation_score = validation_score
                best_epoch = epoch_number
            metrics_record = {
                'epoch': epoch_number,
                'completed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'train_loss': train_loss_epoch,
                'train_loss_ce': train_loss_ce_epoch,
                'train_loss_clip': train_loss_clip_epoch,
                'learning_rate': learning_rate,
                **validation_summary,
                'validation_score': validation_score,
                'best_validation_score': best_validation_score,
                'best_epoch': best_epoch,
            }
            _write_epoch_metrics(metrics_dir, metrics_record)
            save_obj = {
                    'model': model.state_dict(),
                    'model_guideline': model_guideline.state_dict(),
                    'image_encoder': image_encoder.state_dict(),
                    'text_encoder':text_encoder.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'config': config,
                    'epoch': epoch,
                    'metrics': metrics_record,
                    'best_validation_score': best_validation_score,
                    'best_epoch': best_epoch,
                }
            latest_checkpoint = checkpoints_dir / 'latest.pt'
            _atomic_torch_save(save_obj, latest_checkpoint)
            if is_best:
                _atomic_torch_save(save_obj, checkpoints_dir / 'best.pt')
            if epoch_number % 10 == 0:
                shutil.copy2(latest_checkpoint, checkpoints_dir / f'epoch_{epoch_number:03d}.pt')
            writer.flush()
            for handler in logging.getLogger().handlers:
                handler.flush()
            logging.info(
                'Epoch %03d completed and persisted. validation_score=%.6f, best_score=%.6f, lr=%.8g, checkpoint=%s',
                epoch_number,
                validation_score,
                best_validation_score,
                learning_rate,
                latest_checkpoint,
            )

        if epoch + 1 < max_epoch:
            lr_scheduler.step(epoch + 1)
            
            
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print('Evaluating')
    if 'skin' in args.dataset:
        best_checkpoint_path = checkpoints_dir / 'best.pt'
        if not best_checkpoint_path.is_file():
            raise FileNotFoundError(f'Best SkinCAP checkpoint not found: {best_checkpoint_path}')
        best_checkpoint = torch.load(best_checkpoint_path, map_location='cpu')
        model.load_state_dict(best_checkpoint['model'])
        model_guideline.load_state_dict(best_checkpoint['model_guideline'])
        image_encoder.load_state_dict(best_checkpoint['image_encoder'])
        text_encoder.load_state_dict(best_checkpoint['text_encoder'])
        test_dataset = Skin_Test_Dataset(test_csv, config['image_res'], image_root)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config['test_batch_size'],
            num_workers=test_num_workers,
            pin_memory=True,
            sampler=SequentialSampler(test_dataset),
            collate_fn=None,
            worker_init_fn=utils.seed_worker,
            drop_last=False,
        )
        test_dataloader.num_samples = len(test_dataset)
        test_dataloader.num_batches = len(test_dataloader)
        test_label_list = list(pd.read_csv(test_csv, nrows=0).columns[2:])
        if test_label_list != skin_label_list:
            raise ValueError('SkinCAP train and test label columns must have the same names and order.')
        evaluate_skin_test(
            model, model_guideline, image_encoder, text_encoder, tokenizer, test_dataloader,
            device, args, config, args.guideline_path, skin_label_list,
            best_epoch=best_checkpoint.get('best_epoch', best_epoch),
            best_validation_score=best_checkpoint.get('best_validation_score', best_validation_score),
        )
    else:
        test_logits(args, config, max_epoch)
    if utils.is_main_process():
        writer.flush()
        writer.close()



if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    parser = argparse.ArgumentParser()
    parser.add_argument('--momentum', default=False, type=bool)
    parser.add_argument('--dataset', default='icd53')
    parser.add_argument('--config', default='./configs/ICD.yaml')

    parser.add_argument('--class_num', default=1, type=int)
    # Port
    parser.add_argument('--port', default=80, type=int)

    parser.add_argument('--loss_ratio', default=1, type=float)
    parser.add_argument('--contrast_ratio_text', default=0.1, type=float)
    parser.add_argument('--temperature_text', default=0.5, type=float)
    parser.add_argument('--contrast_ratio_vision', default=0.1, type=float)
    parser.add_argument('--temperature_vision', default=2.0, type=float)

    parser.add_argument('--dist_backend', default='nccl')

    parser.add_argument('--output_dir', default='./output_dir/0116_toy')
    parser.add_argument('--train_csv', default='', help='Training CSV path. Overrides ICD_train_file in the YAML configuration.')
    parser.add_argument('--val_csv', default='', help='Validation CSV path. Required for SkinCAP and overrides ICD_val_file in the YAML configuration.')
    parser.add_argument('--test_csv', default='', help='Test CSV path. Overrides ICD_test_file in the YAML configuration.')
    parser.add_argument('--image_root', default='', help='Root directory for relative image paths stored in the CSV files.')
    parser.add_argument('--image_encoder_name', default='resnet50')

    parser.add_argument('--guideline_path', default='')
    parser.add_argument('--bert_model_name', default='')
    parser.add_argument('--max_length', default=512, type=int)
    parser.add_argument('--embed_dim', type=int, default=768, help='embedding dim')
    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--distributed', default=False)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    parser.add_argument('--gpu', default='0', type=str, help='gpu')
    args = parser.parse_args()
    os.environ['MASTER_PORT'] = f'{args.port}'

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))  

    logging.info("Params:")
    params_file = os.path.join(args.output_dir, "params.txt")
    with open(params_file, "w") as f:
        for name in sorted(vars(args)):
            val = getattr(args, name)
            logging.info(f"  {name}: {val}")
            f.write(f"{name}: {val}\n")

    seed_torch(args.seed)
    main(args, config)
