import shutil
import os
from pathlib import Path

curdir = Path(__file__).resolve().parent

for i in range(1, 4 + 1):
    for j in range(1, 11 + 1):
        for k in range(1, 30 + 1):
            path = curdir.joinpath(f"dataset/{j}")
            try:
                os.makedirs(str(path))
            except Exception:
                pass
            p_from = str(curdir.joinpath(f"senz3d_dataset/acquisitions/S{i}/G{j}/{k}-color.png"))
            p_to = str(path.joinpath(f"{(i - 1) * 30 + k}.png"))
            shutil.copy(p_from, p_to)
