from asyncio import new_event_loop
from pathlib import Path
import numpy as np

import os.path as osp
import json
from copy import deepcopy


root = '/opt/ml/input/data/ICDAR19_ArT/ufo/'  
with Path(osp.join(root, 'train.json')).open(encoding='utf8') as handle:
        ann = json.load(handle)

anno = deepcopy(ann)
new_dict = {'images':{}}

num = 0
for img_name in anno['images']:
    if not len(anno['images'][img_name]['words']):
        continue
    for idx in anno['images'][img_name]['words']:
        box = anno['images'][img_name]['words'][idx]['points']
        if len(box) != 4:
            num += 1
            box_array = np.array(box)
            min_x, max_x = int(box_array[:,0].min()), int(box_array[:,0].max())
            min_y, max_y = int(box_array[:,1].min()), int(box_array[:,1].max())
            anno['images'][img_name]['words'][idx]['points'] = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
            

    new_dict['images'][img_name] = anno['images'][img_name]

print(num)
with open(osp.join(root, 'train_new.json'), 'w') as f:
    json_string = json.dump(new_dict, f, indent=4)