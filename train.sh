python train.py \
--exp_name 'Up_Valid_15171919ArT_train' \
--load_pre /opt/ml/input/data/_ArchivePth/Pre06296_ICDAR15171919ArT_weight1312_total2500_100Epoch.pth \
--data_dir "['ICDAR15_Korean', 'ICDAR17_Korean', 'ICDAR19_ArT', 'ICDAR19_Korean']" \
--total 3000 \
--Weighted "[0.2, 0.2, 0.3, 0.3]" \
--max_epoch 150 \
--batch_size 16 \
--learning_rate 2e-4
# 실행방법
# 1. sh train.sh
# 2. ./train.sh