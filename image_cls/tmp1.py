import os
import sys
import glob
import shutil
root1='/media/vision/43c620be-e7c3-4af9-9cf6-c791ef2ed83e/zzq/research/PycharmProjects/symgan/train-data/rafd/pe-cascade-gp/test/target/'
root2='/media/vision/43c620be-e7c3-4af9-9cf6-c791ef2ed83e/zzq/research/PycharmProjects/symgan/train-data/rafd/pe-cascade-gp/test/output/'
for files in glob.glob(root1+"*.png"):
    p,n=os.path.split(files)
    nn=n.split('.')
    name=nn[0]
    name_=name.split('-')
    final_name=root1+'/'+name_[0]+'.png'
    shutil.move(files,final_name)

for files in glob.glob(root2+"*.png"):
    p,n=os.path.split(files)
    nn=n.split('.')
    name=nn[0]
    name_=name.split('-')
    final_name=root2+'/'+name_[0]+'.png'
    shutil.move(files,final_name)