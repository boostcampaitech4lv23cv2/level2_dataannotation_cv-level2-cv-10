# 변경가능
import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset, SceneTextDataset2
from model import EAST

from utils.seed import seed_everything
import wandb
from deteval import calc_deteval_metrics


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
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--save_interval', type=int, default=5)

    # 추가
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--seed', type=int, default=214)

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, exp_name, seed):
     # fix seed
    seed_everything(seed)
    
    # # wandb
    # wandb.login()
    # config = args.__dict__
    # config['exp_name'] = exp_name
    # wandb.init(project='data_ann', entity='godkym', name=exp_name, config=config)

    train_dataset = SceneTextDataset(train_data_dir, split='train', image_size=image_size, crop_size=input_size)
    train_dataset = EASTDataset(train_dataset)
    valid_dataset = SceneTextDataset(valid_data_dir, split='val', image_size=image_size, crop_size=input_size)
    valid_dataset = EASTDataset(valid_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)

    max_hmean = 0
    for epoch in range(max_epoch):      
        epoch_loss, epoch_start = 0, time.time()
        gt_bboxes, pred_bboxes = [], []

        # train
        model.train()
        num_batches = math.ceil(len(train_dataset) / batch_size)
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.update(100)
                val_dict = {
                        'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'], 'IoU loss': extra_info['iou_loss']
                        }
                # wandb logging
                # wandb.log({
                #     'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                #     'IoU loss': extra_info['iou_loss']
                # })
                pbar.set_postfix(val_dict)

        print('[Train] Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))

        # validation
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
                
            orig_sizes = []
            pred_bbox, gt_bbox = [], []
            
            for img, gt_score_map, gt_geo_map, roi_mask in val_loader:
                pred_score_map, pred_geo_map = model.forward(img.to(device))
                pred_score_map, pred_geo_map = pred_score_map.cpu().numpy(), pred_geo_map.cpu().numpy()
                gt_score_map, gt_geo_map = gt_score_map.cpu().numpy(), gt_geo_map.cpu().numpy()

                for pred_score, pred_geo, gt_score, gt_geo, orig_size in zip(pred_score_map, pred_geo_map, gt_score_map, gt_geo_map, orig_sizes):
                    gt_bbox_angle = get_bboxes(gt_score, gt_geo)
                    pred_bbox_angle = get_bboxes(pred_score, pred_geo)

                    if gt_bbox_angle is None:
                        gt_bbox_angle = np.zeros((0, 4, 2), dtype=np.float32)
                    else:
                        gt_bbox_angle = gt_bbox_angle[:, :8].reshape(-1, 4, 2)
                        gt_bbox_angle *= max(orig_size) / input_size

                    if pred_bbox_angle is None:
                        pred_bbox_angle = np.zeros((0, 4, 2), dtype=np.float32)
                    else:
                        pred_bbox_angle = pred_bbox_angle[:, :8].reshape(-1, 4, 2)
                        pred_bbox_angle *= max(orig_size) / input_size

                    pred_bbox.append(pred_bbox_angle)
                    gt_bbox.append(gt_bbox_angle)

                pred_bboxes.extend(pred_bbox)
                gt_bboxes.extend(gt_bbox)
            
            img_len = len(valid_dataset)
            pred_bboxes_dict, gt_bboxes_dict = dict(), dict()
            for img_num in range(img_len):
                pred_bboxes_dict[img_num] = pred_bboxes[img_num]
                gt_bboxes_dict[img_num] = gt_bboxes[img_num]
            
            deteval_metrics = calc_deteval_metrics(pred_bboxes_dict, gt_bboxes_dict)['total']
            print(f'[Val] Precision: {deteval_metrics['precision']} | Recall: {deteval_metrics['recall']} | Hansumean: {deteval_metrics['hmean']}')

            if deteval_metrics[hmean] > max_hmean:
                max_hmean = deteval_metrics[hmean]
                if not osp.exists(model_dir):
                    os.makedirs(model_dir)
                print('best model changed!')
                ckpt_fpath = osp.join(model_dir, 'best.pth')
                torch.save(model.state_dict(), ckpt_fpath)
            # wandb logging
            # wandb.log({
            #     'Precision': deteval_metrics['precision'], 'Recall': deteval_metrics['recall'], 'Hansumean': deteval_metrics['hmean']
            # })

        # scheduler
        scheduler.step()

        # checkpoint
        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, 'latest.pth')
            torch.save(model.state_dict(), ckpt_fpath)

    # wandb.finish()


def main(args):
    do_training(**args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    main(args)