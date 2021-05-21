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

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Processing video using YOLACT')
    parser.add_argument('--trained_model',
                        default='weights/yolact_plus_resnet50_54_800000.pth', type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
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

    cnn = InfTool(weights=args.trained_model, config="yolact_plus_resnet50_config",
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
    max_count = 10000
    highest_class_scores = defaultdict(list)
    while success and count < max_count:
        highest_class_name = "none"
        highest_score = 1
        classes, class_names, scores, boxes, masks, centroids = cnn.raw_inference(image)
        image_info = zip(classes, class_names, scores, boxes, masks, centroids)
        # Filter "noise classes"
        image_info = filter((lambda a: a[1] != "dining table" and a[1] != "bed"), image_info)
        # Filter classes far from the center
        image_info = list(filter((lambda a: np.linalg.norm(a[5] - image_center) < distance_threshold), image_info))
        if image_info:
            # Find class with the highest score
            highest_class_info = max(image_info, key=lambda x: x[2])
            highest_class_name = highest_class_info[1]
            highest_score = highest_class_info[2]
        counter[highest_class_name] += 1
        highest_class_scores[highest_class_name].append(highest_score)
        #labeled_image = cnn.label_image(image)
        #cv2.imwrite(os.path.join(images_dir, "frame%d.jpg" % count), labeled_image)     # save frame as JPEG file
        print("frame%d done" % count)
        success, image = vid.read()
        count += 1

    del counter["none"]
    del highest_class_scores["none"]
    print(counter)
    labels = counter.keys()
    indexes = np.arange(len(labels))
    values = np.fromiter(counter.values(), dtype=float)
    norm_values = values/values.sum()
    mean_scores = {class_name: round(np.mean(scores), 2) for class_name, scores in highest_class_scores.items()}
    labels_scores = list(map(lambda l: l + "\n({:0.2f})".format(mean_scores[l]), labels))
    t1 = torch.Tensor(torch.from_numpy(norm_values).cuda().float())
    print(mean_scores.values)
    t2 = torch.Tensor(list(mean_scores.values()))
    print(F.softmax(t1 * t2))
    #print(torch.log(F.softmax(t1 * t2)))
    print(t1*t2/torch.sum(t1*t2))
    plt.figure(figsize=(16,9))
    plt.title("Relative frequencies of given classes to the {} per {} frames".format(object_name, count), pad=15, weight="bold")
    plt.xlabel('Given class and mean score', labelpad=15, color='#333333')
    plt.ylabel('Relative frequency', labelpad=15, color='#333333')
    #plt.xticks(rotation='vertical')
    plt.bar(labels_scores, norm_values)
    #plt.xticks(indexes + 0.5, labels)
    plt.savefig(os.path.join(images_dir, object_name + "_classes_bar.png"))

    print("Video processing succeeded")
