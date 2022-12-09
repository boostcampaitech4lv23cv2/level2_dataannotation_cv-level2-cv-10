from asyncio import new_event_loop
from pathlib import Path
import numpy as np

import os.path as osp
import json
from copy import deepcopy


root = '/opt/ml/input/data/Upstage/ufo/'  
with Path(osp.join(root, 'train.json')).open(encoding='utf8') as handle:
        ann = json.load(handle)

anno = deepcopy(ann)
new_dict = {'images':{}}

for img_name in anno['images']:
    if not len(anno['images'][img_name]['words']):
        continue
    for idx in anno['images'][img_name]['words']:
        box = anno['images'][img_name]['words'][idx]['points']
        if len(box) > 4 and len(box) % 2 == 0:
            box_array = np.array(box)
            min_x, max_x = box_array[:,0].min(), box_array[:,0].max()
            min_y, max_y = box_array[:,1].min(), box_array[:,1].max()
            anno['images'][img_name]['words'][idx]['points'] = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]

    new_dict['images'][img_name] = anno['images'][img_name]

with open(osp.join(root, 'train_new.json'), 'w') as f:
    json_string = json.dump(new_dict, f, indent=2)