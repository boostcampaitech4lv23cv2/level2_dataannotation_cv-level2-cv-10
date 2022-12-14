# 변경가능
import os
import ast
import argparse
import os.path as osp
import numpy as np
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
from deteval import calc_deteval_metrics
from detect import get_bboxes

def arg_as_datadir_list(s):
    v=ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list " %(s))
    v = ['/opt/ml/input/data/' + dataset for dataset in v]
    return v

def arg_as_num_list(s):
    v=ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list " %(s))
    return v

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--seed', type=int, default=214)
    parser.add_argument('--val_interval', type=int, default=5)

    # 설정
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--data_dir', type=arg_as_datadir_list, default="['ICDAR17_Korean']")
    parser.add_argument('--val_data_dir', type=arg_as_datadir_list, default="['ICDAR17_Korean']")
    parser.add_argument('--total', type=int)
    parser.add_argument('--Weighted', type=arg_as_num_list, default="[1, 1, 1, 1, 1, 1]")
    parser.add_argument('--load_pre', type=str)

    

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_training(data_dir, model_dir, val_data_dir, val_interval, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, seed, exp_name, Weighted, total, load_pre=None):
    # fix seed
    seed_everything(seed)
    
    # wandb
    wandb.login()
    # exp_name = increment_path(model_dir, exp_name)
    config = args.__dict__
    config['exp_name'] = exp_name
    wandb.init(project='data_ann', entity='godkym', name=exp_name, config=config)

    dataset = SceneTextDataset(data_dir, split='train', image_size=image_size, crop_size=input_size)
    each_dataset_len = dataset.get_each_dataset_len()
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
    if load_pre:
        model.load_state_dict(torch.load(load_pre))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)

    model.train()
    max_hmean = 0
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

        scheduler.step()

        #TODO: validation

        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))

        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, 'latest.pth')
            torch.save(model.state_dict(), ckpt_fpath)

        if epoch % val_interval != 0:
            continue
        #########################################  validation  ############################################################################################################
        val_dataset = SceneTextDataset(val_data_dir, split='train', image_size=image_size, crop_size=input_size)
        val_dataset = EASTDataset(val_dataset)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        gt_bboxes, pred_bboxes, transcriptions = [], [], []
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            
            for img, gt_score_map, gt_geo_map, _ in val_loader:
                orig_sizes = []
                pred_bbox, gt_bbox, transcription = [], [], []

                for image in img:
                    orig_sizes.append(image.shape[:2]) 

                pred_score_map, pred_geo_map = model.forward(img.to(device))
                pred_score_map, pred_geo_map = pred_score_map.cpu().numpy(), pred_geo_map.cpu().numpy()
                gt_score_map, gt_geo_map = gt_score_map.cpu().numpy(), gt_geo_map.cpu().numpy()

                for pred_score, pred_geo, gt_score, gt_geo, orig_size in zip(pred_score_map, pred_geo_map, gt_score_map, gt_geo_map, orig_sizes):
                    pred_bx = get_bboxes(pred_score, pred_geo)
                    gt_bx = get_bboxes(gt_score, gt_geo)

                    if pred_bx is None:
                        pred_bx = np.zeros((0, 4, 2), dtype=np.float32)
                    else:
                        pred_bx = pred_bx[:, :8].reshape(-1, 4, 2)
                        pred_bx *= max(orig_size) / input_size

                    if gt_bx is None:
                        gt_bx = np.zeros((0, 4, 2), dtype=np.float32)
                        trans = []
                    else:
                        gt_bx = gt_bx[:, :8].reshape(-1, 4, 2)
                        gt_bx *= max(orig_size) / input_size
                        trans = ['null' for _ in range(gt_bx.shape[0])]

                    pred_bbox.append(pred_bx)
                    gt_bbox.append(gt_bx)
                    transcription.append(trans)

                pred_bboxes.extend(pred_bbox)
                gt_bboxes.extend(gt_bbox)
                transcriptions.extend(transcription)
            
            pred_bboxes_dict, gt_bboxes_dict, transcriptions_dict = dict(), dict(), dict()
            img_len = len(val_dataset)
            for img_num in range(img_len):
                pred_bboxes_dict[f'{img_num}'] = pred_bboxes[img_num]
                gt_bboxes_dict[f'{img_num}'] = gt_bboxes[img_num]
                transcriptions_dict[f'{img_num}'] = transcriptions[img_num]
            
            deteval_metrics = calc_deteval_metrics(pred_bboxes_dict, gt_bboxes_dict, transcriptions_dict)['total']
            # fix
            print(f"[Val] Precision: {deteval_metrics['precision']:.4f} | Recall: {deteval_metrics['recall']:.4f} | Hamsumean: {deteval_metrics['hmean']:.4f}")

            if deteval_metrics['hmean'] > max_hmean:
                max_hmean = deteval_metrics['hmean']
                if not osp.exists(model_dir):
                    os.makedirs(model_dir)

                print('best model changed!')
                ckpt_fpath = osp.join(model_dir, 'best.pth')
                torch.save(model.state_dict(), ckpt_fpath)
            # wandb logging
            wandb.log({
                'Precision': deteval_metrics['precision'], 'Recall': deteval_metrics['recall'], 'Hansumean': deteval_metrics['hmean']
            })
        ###################################################################################################################################################################

    wandb.finish()

def main(args):
    do_training(**args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    main(args)

# 변경 가능한 부분
# batch_size
# lr_rate, epoch
# lr_rate_scheduling
# optimizer
# data augmentation
# input_data

# 1. eda

# 2.det eval.py가 어디서 사용됨?
# deteval적용해서 loss값 알아보기
# deteval적용해서 save best

# 3. detect.py는 뭐하는데 쓰임?

# 4. 실험을 통해 변경 가능한 부분에서 최댓값 알아보기

# 5. (confusion matrix를 통해) 어떤 부분에 대해서 학습이 부족한지? 알아보기