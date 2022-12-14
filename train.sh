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

# 잘 되는지 확인하는 방법
# 1. 17korean, 19korean으로 total 500 weighted [1, 2]로 2시간동안 돌려서 잘 나오는지 확인하기