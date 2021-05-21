## A benchmark measuring speed for sequential calls to YOLACT inference (eval.py, InfTool.raw_inference())
# vs. using a batch mode.
from inference_tool import InfTool

import cv2
import numpy as np
import os
import imageio
import glob

import timeit

DATASET='./data/yolact/datasets'

IMGS=os.path.join(DATASET, "banana_360")
COUNT=69
BATCH=1

if __name__ == '__main__':
  cnn = InfTool(weights='./data/yolact/weights/weights_yolact_23/crow_base_7_133333.pth',
                top_k=15, score_threshold=0.51,
                config='/home/nikita/crow_vision_yolact/data/yolact/weights/weights_yolact_23/config_train.obj')

  #prepare data for work
  yolact_semantic_dir = os.path.join(DATASET, "yolact_semantic")
  os.makedirs(yolact_semantic_dir, exist_ok=True)
  files = glob.glob(os.path.join(yolact_semantic_dir,'./*'))
  for f in files:
    os.remove(f)

  images = []
  images_numpy = []
  names = os.listdir(IMGS)
  names.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  video=cv2.VideoWriter('banana_habitat.mp4', fourcc, 30,(640,480))

  for name in names:
      img = cv2.imread(os.path.join(IMGS, name))
      video.write(img)
      img_numpy = cnn.label_image(img)
      # cv2.imshow('retina_cam_{}'.format(name),img)
      # cv2.waitKey(1000)
      cv2.imshow('img_yolact', img_numpy)
      cv2.imwrite(os.path.join(yolact_semantic_dir, name),
                    img_numpy, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
      images_numpy.append(img_numpy)
      cv2.waitKey(1)
      images.append(img)

  cv2.destroyAllWindows()
  video.release()

  # gif_path = "spring_yolact_test.gif"
  # imageio.mimsave(gif_path, [np.array(cv2.cvtColor(img, cv2.COLOR_RGB2BGR)) for i, img in enumerate(images_numpy) if i%2 == 0], fps=5)
  # print("Record saved to " + gif_path)
