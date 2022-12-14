python train.py \
--exp_name '19ArT_Test' \
--load_pre /opt/ml/input/data/_ArchivePth/Pre06296_ICDAR15171919ArT_weight1312_total2500_.pth \
--data_dir "['ICDAR15_Korean', 'ICDAR17_Korean', 'ICDAR19_ArT', 'ICDAR19_Korean']" \
--total 2500 \
--Weighted "[1, 3, 1, 2]" \
--max_epoch 100 \
--batch_size 16

# 실행방법
# 1. sh train.sh
# 2. ./train.sh