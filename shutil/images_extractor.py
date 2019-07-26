import csv
import shutil
import os

data_root = '/home/weizhaoyu/wzy/ori/dataset/'
target_root = '/home/weizhaoyu/wzy/ori/position/'
# train_root = '/home/weizhaoyu/wzy/APTOS2019/rawdata/train/'
# val_root = '/home/weizhaoyu/wzy/APTOS2019/rawdata/val/'
with open('/home/weizhaoyu/wzy/ori/影像评级明细（含部位）.csv',"rt", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    rows= [row for row in reader]
    rows = rows[1:]
    for row in rows:
        if os.path.exists(target_root+row[7]):
            full_path = data_root + row[1]
            try:
                shutil.move(full_path,target_root + row[7] +'/')
            except(FileNotFoundError):
                pass
        else:
            os.makedirs(data_root+row[7])
            full_path = data_root + row[1]
            try:
                shutil.move(full_path,target_root + row[7] +'/')
            except(FileNotFoundError):
                pass
