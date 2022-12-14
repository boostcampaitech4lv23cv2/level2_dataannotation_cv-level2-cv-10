# 변경가능
import os
import ast
import argparse
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST

from utils.seed import seed_everything
import wandb

# def increment_path(model_dir, exp_name, exist_ok=False):
#     """ Automatically increment path, i.e. trained_models/exp --> trained_models/exp0, trained_models/exp1 etc.
#     Args:
#         exist_ok (bool): whether increment path (increment if False).
#     """
#     path = osp.join(model_dir, exp_name)
#     path = Path(path)
#     if (path.exists() and exist_ok) or (not path.exists()):
#         return str(exp_name)
#     else:
#         dirs = glob.glob(f"{path}*")
#         matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
#         i = [int(m.groups()[0]) for m in matches if m]
#         n = max(i) + 1 if i else 2
#         return f"{exp_name}{n}"


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '../input/data/Upstage'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=5)

    # 추가
    parser.add_argument('--exp_name', type=str, default='15+17+19+aug_v2+batch24')
    parser.add_argument('--seed', type=int, default=214)

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, seed, exp_name):

    # fix seed
    seed_everything(seed)
    
    # wandb
    wandb.login()
    # exp_name = increment_path(model_dir, exp_name)
    config = args.__dict__
    config['exp_name'] = exp_name
    wandb.init(project='data_ann', entity='godkym', name=exp_name, config=config)

    # wandb.define_metric('Recall', summary='max')
    # wandb.define_metric('Hansumean', summary='max')
    # wandb.define_metric('Precision', summary='max')

    dataset = SceneTextDataset(data_dir, split='train_151719_ver', image_size=image_size, crop_size=input_size)
    dataset = EASTDataset(dataset)
    # num_batches = math.ceil(len(dataset) / batch_size)
    # train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    ######################################################################################################################
    # parameter추가 할 것: Weighted, total_data_num
    Weighted = Weighted[:len(data_dir)]
    print("data_dir: ", data_dir)
    str_w = ''
    total_data_num = total
    all_Weight = []
    for i in range(len(data_dir)):
        W = [Weighted[i]] * each_dataset_len[i]
        all_Weight.extend(W)
        str_w += ('[' + str(W[0]) + ', ..., '+ str(W[-1]) + ']\t')
    print("Weight: ", str_w)
    Sampler = WeightedRandomSampler(W, total_data_num)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=Sampler)
    num_batches = math.ceil(total_data_num / batch_size)
    ######################################################################################################################
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    # model.load_state_dict(torch.load('pths/latest_151719.pth'))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)

    model.train()
    for epoch in range(max_epoch):
        epoch_loss, epoch_start = 0, time.time()
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.update(1)
                val_dict = {
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }
                # wandb logging
                wandb.log({
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                })
                pbar.set_postfix(val_dict)

                wandb.log({
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                })
        scheduler.step()

        #TODO: validation

        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))

        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, 'latest.pth')
            torch.save(model.state_dict(), ckpt_fpath)

    wandb.finish()

def main(args):
    do_training(**args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    main(args)

# 변경 불가
# model.py
# loss.py
# east_dataset.py
# detect.py