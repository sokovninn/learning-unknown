import os
import numpy as np

trained_model = './data/yolact/weights/weights_yolact_kuka_14/crow_plus_base_56_330000.pth'
score_threshold = '0.15'
top_k = '15'
max_images='1000'
image_source_cams=[1,1,1,1,1,1,1,1]

command = 'python eval.py --trained_model='+trained_model+' --score_threshold='+score_threshold+' --top_k='+top_k+' --max_images='+max_images+' --dataset=kuka_env_pybullet_dataset_test_cam'
for cam in np.nonzero(image_source_cams)[0]:
    cam=str(cam)
    print('testing on camera '+cam+':')    
    os.system(command+cam)