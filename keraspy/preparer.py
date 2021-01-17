import shutil
import os
from pathlib import Path

curdir = Path(__file__).resolve().parent

person_count = 5
pic_count = 8
format = '.jpg'

for j in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
    cur = 1
    for i in range(1, person_count + 1):
        for k in range(1, pic_count + 1):
            path = curdir.joinpath(f"dataset2/{ord(j) - ord('A') + 1}")
            try:
                os.makedirs(str(path))
            except Exception:
                pass
            p_from = str(curdir.joinpath(f"gestures_dataset/HGM-4/HGM-1.0/Below_CAM/{j}/P{i}_00{k}{format}"))
            p_to = str(path.joinpath(f"{cur}{format}"))
            try:
                shutil.copy(p_from, p_to)
            except Exception:
                continue
            cur += 1
