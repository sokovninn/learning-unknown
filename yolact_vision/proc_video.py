from inference_tool import InfTool

from collections import Counter
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
import glob
import pathlib
import torch.nn.functional as F
import torch
import json

from scipy.stats import entropy

from data.config import LU_COCO_CLASSES

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Processing video using YOLACT')
    parser.add_argument('--weights_folder',
                        default='data/weights', type=str,
                        help='Folder with weights')
    parser.add_argument('--trained_model',
                        default='yolact_base_54_800000.pth', type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument('--config',
                        default='yolact_base_config', type=str,
                        help='Model config')
    parser.add_argument('--top_k', default=15, type=int,
                        help='Further restrict the number of predictions to parse')
    parser.add_argument('--video', default=None, type=str,
                        help='A path to a video to evaluate on. Passing in a number will use that index webcam.')
    parser.add_argument('--score_threshold', default=0.15, type=float,
                        help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')

    args = parser.parse_args(argv)

    return args

if __name__ == '__main__':
    args = parse_args()

    cnn = InfTool(weights=os.path.join(args.weights_folder, args.trained_model), config=args.config,
                  top_k=args.top_k, score_threshold=args.score_threshold)

    is_webcam = args.video.isdigit()
    if is_webcam:
        vid = cv2.VideoCapture(int(args.video))
    else:
        vid = cv2.VideoCapture(args.video)

    if not vid.isOpened():
        print('Could not open video "%s"' % args.video)
        exit(-1)

    object_name = os.path.splitext(pathlib.PurePath(args.video).name)[0]
    images_dir = "video_frames_" + object_name
    os.makedirs(images_dir, exist_ok=True)
    files = glob.glob(os.path.join(images_dir,'./*'))
    for f in files:
      os.remove(f)


    success,image = vid.read()
    image_center = np.array([image.shape[0] / 2, image.shape[1] / 2])
    counter = Counter()
    distance_threshold = 500 #should be smaller for low resolution frames
    count = 0
    detected_frames = 0
    max_count = 10000
    highest_class_scores = defaultdict(list)
    smax_vectors = None
    saved_frames = [0, 100, 200]
    saved_images = None
    while success and count < max_count:

        classes, class_names, scores, boxes, masks, centroids, smax = cnn.raw_inference(image)
        if smax.shape[0] != 0:
            if smax.shape[1] != 0:
                detected_frames +=1
            if smax_vectors is None:
                smax_vectors = smax.detach().cpu().numpy()
            else:
                print(smax_vectors.shape, smax.shape, smax)
                print(class_names)
                smax_vectors = np.concatenate((smax_vectors, smax.detach().cpu().numpy()), axis=1)
        for i in range(len(scores)):
            highest_class_scores[class_names[i]].append(scores[i])

        #labeled_image = cnn.label_image(image)
        #cv2.imwrite(os.path.join(images_dir, "frame%d.jpg" % count), labeled_image)     # save frame as JPEG file
        print("frame%d done" % count)
        success, image = vid.read()
        if count in saved_frames:
            x = int(image.shape[1]/2)
            y = int(image.shape[0]/2)
            cropped_image = cv2.resize(image[:,x-y:x+y, :], (0,0), fx=0.5, fy=0.5)
            if saved_images is None:
                saved_images = cropped_image
            else:
                saved_images = cv2.vconcat([saved_images, cropped_image])
        count += 1

    cv2.imwrite(os.path.join(images_dir,object_name + '_frames.png'), saved_images)
    #del counter["none"]
    #del highest_class_scores["none"]

    labels = list(highest_class_scores.keys())
    sums_scores = np.array([round(np.sum(scores), 2) for class_name, scores in highest_class_scores.items()])
    norm_sums_scores = sums_scores / np.sum(sums_scores)
    print("Sums results: {}".format(norm_sums_scores))
    print("Labels: {}".format(labels))
    maxsum = np.max(norm_sums_scores)

    total_mean = np.mean(smax_vectors, 1)
    print(smax_vectors.shape)
    print("Total mean {}{}".format(total_mean.shape, np.sum(total_mean)))
    entr = entropy(total_mean.squeeze()) / np.log(len(total_mean))
    max_class = np.argmax(total_mean) - 1
    if max_class == -1:
        final_class = "background"
    else:
        final_class = LU_COCO_CLASSES[max_class]
    print("Class: {}".format(final_class))
    print("Entropy: {}".format(entr))


    with open(os.path.join(images_dir,'detections.json'), 'w', encoding='utf-8') as f:
        json.dump({"args": vars(args),
        "object_name": object_name,
        "num_frames": count,
        "detected_frames": detected_frames,
        "entropy": float(round(entr, 2)),
        "maxsum": round(float(maxsum), 2),
        "max_class": final_class,
        "detections": smax_vectors.tolist(),
         }, f, ensure_ascii=False, indent=4)

    print("Episode finished after {} steps.".format(count))

    print("Video processing succeeded")
