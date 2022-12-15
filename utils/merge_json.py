import json
import os

file1_dir = '/opt/ml/input/data/ICDAR_15_17_19/ufo/train_15.json'
file2_dir = '/opt/ml/input/data/ICDAR_15_17_19/ufo/train_17.json'
file3_dir = '/opt/ml/input/data/ICDAR_15_17_19/ufo/train_19.json'
save_dir = '/opt/ml/input/data/ICDAR_15_17_19/ufo/'

with open(file1_dir) as f:
    anno1 = json.load(f)

with open(file2_dir) as f:
    anno2 = json.load(f)

with open(file3_dir) as f:
    anno3 = json.load(f)
    
annos = [anno1 ,anno2, anno3]


def merge_file(annos):
    total = dict(images=dict())

    for anno in annos:
        images = anno['images']
        for img in images:
            total['images'][img] = images[img]


total = merge_file(annos)
with open(os.path.join(save_dir,'train.json'), 'w') as f:
    train = json.dump(total, f, indent=4)