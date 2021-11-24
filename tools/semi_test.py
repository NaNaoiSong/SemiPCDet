import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_semi_test_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils


class DistStudent(nn.Module):
    def __init__(self, student):
        super().__init__()
        self.onepass = student

    def forward(self, ld_batch, ud_batch):
        return self.onepass(ld_batch), self.onepass(ud_batch)

class DistTeacher(nn.Module):
    def __init__(self, teacher):
        super().__init__()
        self.onepass = teacher

    def forward(self, ld_batch, ud_batch):
        if ld_batch is not None:
            return self.onepass(ld_batch), self.onepass(ud_batch)
        else:
            return None, self.onepass(ud_batch)


def parse_config():
    group_tag = 'semi_once_models'
    model_tag = 'mean_teacher_second'
    ext_tag = 'default'
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='../output/%s/%s/%s/%s.yaml' % (group_tag, model_tag, ext_tag, model_tag), help='specify the config for training')
    parser.add_argument('--batch_size', type=int, default=2, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--target_dir', type=str, default=ext_tag, help='target directory for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=12345, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=5, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = group_tag

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id=-1, dist_test=False):
    # load checkpoint
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test)
    model.cuda()

    # start evaluation
    eval_utils.eval_one_epoch(
        cfg, model, test_loader, epoch_id, logger, dist_test=dist_test,
        result_dir=eval_output_dir, save_to_file=args.save_to_file
    )


def get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args):
    ckpt_list = glob.glob(os.path.join(ckpt_dir, '*checkpoint_epoch_*.pth'))
    ckpt_list.sort(key=os.path.getmtime)
    evaluated_ckpt_list = [float(x.strip()) for x in open(ckpt_record_file, 'r').readlines()]

    for cur_ckpt in ckpt_list:
        num_list = re.findall('checkpoint_epoch_(.*).pth', cur_ckpt)
        if num_list.__len__() == 0:
            continue

        epoch_id = num_list[-1]
        if 'optim' in epoch_id:
            continue
        if float(epoch_id) not in evaluated_ckpt_list and int(float(epoch_id)) >= args.start_epoch:
            return epoch_id, cur_ckpt
    return -1, None


def repeat_eval_ckpt(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=False):
    # evaluated ckpt record
    ckpt_record_file = eval_output_dir / ('eval_list_%s.txt' % cfg.DATA_CONFIG.DATA_SPLIT['test'])
    with open(ckpt_record_file, 'a'):
        pass

    # tensorboard log
    if cfg.LOCAL_RANK == 0:
        tb_log = SummaryWriter(log_dir=str(eval_output_dir / ('tensorboard_%s' % cfg.DATA_CONFIG.DATA_SPLIT['test'])))
    total_time = 0
    first_eval = True

    while True:
        # check whether there is checkpoint which is not evaluated
        cur_epoch_id, cur_ckpt = get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args)
        if cur_epoch_id == -1 or int(float(cur_epoch_id)) < args.start_epoch:
            wait_second = 30
            if cfg.LOCAL_RANK == 0:
                print('Wait %s seconds for next check (progress: %.1f / %d minutes): %s \r'
                      % (wait_second, total_time * 1.0 / 60, args.max_waiting_mins, ckpt_dir), end='', flush=True)
            time.sleep(wait_second)
            total_time += 30
            if total_time > args.max_waiting_mins * 60 and (first_eval is False):
                break
            continue

        total_time = 0
        first_eval = False

        model.load_params_from_file(filename=cur_ckpt, logger=logger, to_cpu=dist_test)
        model.cuda()

        # start evaluation
        cur_result_dir = eval_output_dir / ('epoch_%s' % cur_epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
        tb_dict = eval_utils.eval_one_epoch(
            cfg, model, test_loader, cur_epoch_id, logger, dist_test=dist_test,
            result_dir=cur_result_dir, save_to_file=args.save_to_file
        )

        # record this epoch which has been evaluated
        with open(ckpt_record_file, 'a') as f:
            print('%s' % cur_epoch_id, file=f)
        logger.info('Epoch %s has been evaluated' % cur_epoch_id)


def main():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_test = False
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.target_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    ssl_ckpt_dir = output_dir / 'ssl_ckpt'
    eval_ssl_dir = output_dir / 'eval'
    eval_ssl_dir.mkdir(parents=True, exist_ok=True)
    log_file = eval_ssl_dir / ('log_eval_%sdata.txt' % cfg.DATA_CONFIG.DATA_SPLIT['test'])
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    test_dataset, test_dataloader, test_sampler = build_semi_test_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=cfg.OPTIMIZATION.TEST.BATCH_SIZE_PER_GPU,
        dist=dist_test,
        root_path=cfg.DATA_CONFIG.DATA_PATH,
        workers=args.workers,
        logger=logger
    )

    teacher_model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_dataset)
    student_model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_dataset)
    teacher_model.cuda()
    student_model.cuda()

    if dist_test:
        student_model = DistStudent(student_model) # add wrapper for dist training
        student_model = nn.parallel.DistributedDataParallel(student_model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
        # teacher doesn't need dist train
        teacher_model = DistTeacher(teacher_model)
        teacher_model = nn.parallel.DistributedDataParallel(teacher_model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])

    logger.info('**********************Start evaluation for student model %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.target_dir))
    eval_ssl_dir = output_dir / 'eval' / 'eval_with_student_model'
    eval_ssl_dir.mkdir(parents=True, exist_ok=True)
    if dist_test:
        student_model.module.onepass.set_model_type('origin') # ret filtered boxes
    else:
        student_model.set_model_type('origin')
    repeat_eval_ckpt(
        model = student_model.module.onepass if dist_test else student_model,
        test_loader = test_dataloader,
        args = args,
        eval_output_dir = eval_ssl_dir,
        logger = logger,
        ckpt_dir = ssl_ckpt_dir / 'student',
        dist_test=dist_test
    )
    logger.info('**********************End evaluation for student model %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.target_dir))

    logger.info('**********************Start evaluation for teacher model %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.target_dir))
    eval_ssl_dir = output_dir / 'eval' / 'eval_with_teacher_model'
    eval_ssl_dir.mkdir(parents=True, exist_ok=True)
    if dist_test:
        teacher_model.module.onepass.set_model_type('origin') # ret filtered boxes
    else:
        teacher_model.set_model_type('origin')
    for t_param in teacher_model.parameters(): # Add this to avoid errors
        t_param.requires_grad = True
    repeat_eval_ckpt(
        model = teacher_model.module.onepass if dist_test else teacher_model,
        test_loader = test_dataloader,
        args = args,
        eval_output_dir = eval_ssl_dir,
        logger = logger,
        ckpt_dir = ssl_ckpt_dir / 'teacher',
        dist_test=dist_test
    )
    logger.info('**********************End evaluation for teacher model %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.target_dir))

if __name__ == '__main__':
    main()
