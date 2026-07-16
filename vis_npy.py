import os
import numpy as np

folder = "yangnet/forest_fire_size_classification_dataset_detail/train/bigscene_bigfire/scene_vec"

count = {
    "near": 0,
    "mid": 0,
    "far": 0
}

for f in os.listdir(folder):
    data = np.load(os.path.join(folder, f))

    if (data == [1,0,0]).all():
        count["near"] += 1
    elif (data == [0,1,0]).all():
        count["mid"] += 1
    elif (data == [0,0,1]).all():
        count["far"] += 1

print(count)