import glob
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--train_dir",required=True, help="path to folder containing train images")
parser.add_argument("--val_dir",required=True, help="path to folder containing val images")
a = parser.parse_args()
cnt=0
for sub_dir in os.listdir(a.train_dir):
    input_paths=glob.glob(os.path.join(a.train_dir,[sub_dir,"*.jpg"]))
    if len(input_paths)==0:
        input_paths = glob.glob(os.path.join(a.train_dir, [sub_dir, "*.png"]))
    if len(input_paths)==0:
        raise Exception("train_dir contains no image files")
    for files in input_paths:
        p,n=os.path.split(files)
        nn=n.split('.')
        name=nn[0]
        rename=name+str(cnt)+'.'+nn[1]
    cnt+=1
cnt=0
for sub_dir in os.listdir(a.val_dir):
    input_paths=glob.glob(os.path.join(a.train_dir,[sub_dir,"*.jpg"]))
    if len(input_paths)==0:
        input_paths = glob.glob(os.path.join(a.train_dir, [sub_dir, "*.png"]))
    if len(input_paths)==0:
        raise Exception("val_dir contains no image files")
    for files in input_paths:
        p,n=os.path.split(files)
        nn=n.split('.')
        name=nn[0]
        rename=name+str(cnt)+'.'+nn[1]
    cnt+=1