import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import cv2
import envs.spring_env
import numpy as np
from PIL import Image
from habitat_sim.utils.common import d3_40_colors_rgb
from collections import defaultdict, Counter
import quaternion
from habitat.utils.visualizations import maps
import imageio
import torch
import torch.nn.functional as F
import os
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import argparse
import json

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('../../yolact')
from inference_tool import InfTool


FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
FINISH="f"
LOOK_UP_KEY="u"
LOOK_DOWN_KEY="j"

#cnn = None

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


def show_observations(cnn, observations, memory_labels={}, map=None, memory=None):
    labeled_image = cnn.label_image(transform_rgb_bgr(observations["rgb"]))
    #cv2.imwrite("yolact_habitat_example.png", labeled_image)
    # Show labels from memory
    for pos, text in memory_labels.items():
        cv2.putText(labeled_image, text, pos, cv2.FONT_HERSHEY_DUPLEX, 1, [255, 255, 255], 1)

    if not map is None:
        labeled_image = np.hstack((labeled_image, map))
        print(labeled_image.shape)
        cv2.putText(labeled_image, "Memory:", (650, 270), cv2.FONT_HERSHEY_DUPLEX, 0.6, [0, 0, 0], 1)
        for i, (key, value) in enumerate(memory.items()):
            cv2.putText(labeled_image, "{}: {}".format(key, value), (650, 270+(i+1)*15), cv2.FONT_HERSHEY_DUPLEX, 0.4, [0, 0, 0], 1)
    cv2.imshow("Labeled image", labeled_image)

def fill_map(
    topdown_map_info: Dict[str, Any], output_height: int, memory, env
):
    r"""Given the output of the TopDownMap measure, colorizes the map, draws the agent,
    and fits to a desired output height

    :param topdown_map_info: The output of the TopDownMap measure
    :param output_height: The desired output height
    """
    top_down_map = topdown_map_info["map"]
    top_down_map = maps.colorize_topdown_map(
        top_down_map, topdown_map_info["fog_of_war_mask"]
    )
    map_agent_pos = topdown_map_info["agent_map_coord"]
    top_down_map = maps.draw_agent(
        image=top_down_map,
        agent_center_coord=map_agent_pos,
        agent_rotation=topdown_map_info["agent_angle"],
        agent_radius_px=min(top_down_map.shape[0:2]) / 32,
    )

    for object_pos, label in memory.items():
        map_x, map_y = maps.to_grid(
            object_pos[2],
            object_pos[0],
            top_down_map.shape[0:2],
            sim=env.sim,
        )
        print(map_x, map_y)
        color = (0, 0, 0) if label == "unknown" else (0, 255, 0)
        top_down_map = cv2.circle(top_down_map, (map_y,map_x), radius=10, color=color, thickness=-1)

    if top_down_map.shape[0] > top_down_map.shape[1]:
        top_down_map = np.rot90(top_down_map, 1)

    # scale top down map to align with rgb view
    old_h, old_w, _ = top_down_map.shape
    top_down_height = output_height
    top_down_width = int(float(top_down_height) / old_h * old_w)
    # cv2 resize (dsize is width first)
    top_down_map = cv2.resize(
        top_down_map,
        (top_down_width, top_down_height),
        interpolation=cv2.INTER_CUBIC,
    )

    return top_down_map

def compute_metrics(memory, labels):
    print(memory)
    print(labels)
    TP = 0
    TPU = 0
    FP = 0
    FN = 0
    object_positions = np.array(list(labels.keys()))
    object_labels = list(labels.values())

    true_set = set([v for k,v in labels.items()])
    memory_set = set([v for k,v in memory.items()])
    predicted_set = list(true_set) + list(memory_set.difference(true_set))

    true_set = list(true_set)
    predicted_set = list(predicted_set)
    confusion_matrix = np.zeros((len(true_set) + 1, len(predicted_set) + 1))

    for pred_pos, pred in memory.items():
        distances = np.linalg.norm(np.array(pred_pos) - object_positions, axis=1)
        min_idx = np.argmin(distances)
        label = labels[tuple(object_positions[min_idx])]
        if distances[min_idx] < 0.7:
            if pred == label:
                print(label)
                if label in object_labels:
                    object_labels.remove(label)
                if label != "unknown":
                    TP += 1
                else:
                    TPU += 1
                confusion_matrix[true_set.index(label)][predicted_set.index(label)] += 1
            else:
                confusion_matrix[true_set.index(label)][predicted_set.index(pred)] += 1
        else:
            confusion_matrix[-1][predicted_set.index(pred)] += 1

    for l in object_labels:
        confusion_matrix[true_set.index(l)][-1] += 1

    print(confusion_matrix)

    df_cm = pd.DataFrame(confusion_matrix, index = true_set + ["no object"],
                  columns = predicted_set + ["not detected"])
    plt.figure(figsize = (10,7))
    plt.title("Classification", fontsize=15)
    sn.heatmap(df_cm, annot=True, square=True)
    plt.xlabel("Predicted labels", fontweight='bold')
    plt.ylabel("True labels", fontweight='bold')
    plt.subplots_adjust(bottom=0.2)
    plt.savefig("classification_matrix.png")

    detected_no_objects = np.sum(confusion_matrix[-1, :])
    undetected_unknown = confusion_matrix[true_set.index("unknown")][-1]
    undetected_known = np.sum(confusion_matrix[:, -1]) - undetected_unknown

    detection_matrix = np.array([[float('inf'), detected_no_objects],
                                 [undetected_unknown, TPU],
                                 [undetected_known, TP]])
    detection_matrix_normalized = detection_matrix # / np.sum(detection_matrix.flatten()[1:])
    df_dm = pd.DataFrame(detection_matrix_normalized, index = ["No object", "Unknown", "Known"],
                  columns = ["Not detected", "Detected"])
    plt.figure(figsize = (7,10))
    plt.title("Detection", fontsize=15)
    sn.heatmap(df_dm, annot=True, cmap="Blues", linecolor="black", linewidths = 1,
                square=True, vmax=np.unique(detection_matrix_normalized)[-2], cbar=False)
    plt.savefig("detection_matrix.png")



    print("sets", true_set, predicted_set)
    # for pred_pos, pred in memory.items():
    #     for label_pos, label in labels.items():
    #         if np.linalg.norm(np.array(pred_pos) - np.array(label_pos)) < 0.5 and pred == label:
    #             if label != "unknown":
    #                 TP += 1
    #             else:
    #                 TPU += 1
    if TP != 0:
        FP = len(memory) - TP - TPU
        FN = len(labels) - TP - TPU
        print(TP, TPU, FP, FN, len(memory))
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1_score = 2 * precision * recall / (precision + recall)
    else:
        precision = 0
        recall = 0
        F1_score = 0

    print("Precision: {}\nRecall: {}\nF1_score: {}".format(precision, recall, F1_score))

    return precision, recall, F1_score

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--manual", type=int, default=1, help="True: manual creation, False: automatic creation using actions from file")
    parser.add_argument("-a", "--actions_filename", type=str, default='actions.json', help="Filename of the file with actions for automatic dataset creation")
    parser.add_argument("-d", "--dataset_folder", type=str, default='data/spring_dataset', help="Path to the folder with dataset")
    parser.add_argument("-v", "--video", type=str, default=None, help="If given then video is captured with specified name")
    parser.add_argument("-t", "--softmax_threshold", type=float, default=0.5, help="Softmax threshold")

    args = parser.parse_args()
    return args

def example():


    args = get_arguments()
    actions = []
    if not args.manual:
        with open(args.actions_filename) as f:
            actions = json.load(f)['sequence']
    #global cnn
    cnn = InfTool(weights="/home/nikita/yolact/weights/yolact_plus_resnet50_54_800000.pth", config="yolact_plus_resnet50_config",
                  top_k=15, score_threshold=0.3)
    objects = [["armchair", [0.64, 0.3, -2.69]],
           ["bed", [-5.4, 0.3, 0.7]],
           ["car", [-1.44, 0.3, -0.8]],
           ["chainsaw",  [1.2, 0.3, 0.2]],
           ["chair", [-4.64, 0.3, -2.69]],
           ["cleaner", [-1, 0.5, 1.4]],
           ["couch",  [-3.64, 0.3, -0.59]],
           ["tob",  [-3.64, 1, 1.4]],
           ["plant",  [1.2, 1, 1.4]],
           ["speaker",  [-0.3, 0.4, -1.5]],
           ["table",  [1, 0.3, -1.2]],
           ["giraffe",  [-4.9, 1, -1]],]

    true_labels = {tuple(objects[0][1]): 'unknown',
              tuple(objects[1][1]): 'bed',
              tuple(objects[2][1]): 'car',
              tuple(objects[3][1]): 'unknown',
              tuple(objects[4][1]): 'chair',
              tuple(objects[5][1]): 'unknown',
              tuple(objects[6][1]): 'couch',
              tuple(objects[7][1]): 'unknown',
              tuple(objects[8][1]): 'potted plant',
              tuple(objects[9][1]): 'unknown',
              tuple(objects[10][1]): 'table',
              tuple(objects[11][1]): 'giraffe',
              }




    # memory =  {(-1.23, 0.37, -0.54): 'car', (-2.8, 0.52, -0.28): 'unknown', (0.75, 0.63, -1.0): 'unknown', (1.13, 0.32, 0.01): 'unknown', (1.14, 0.9, 1.22): 'potted plant', (-0.66, 0.74, 0.99): 'unknown', (-1.09, 0.17, -1.33): 'clock', (-0.18, 0.87, -1.26): 'unknown', (-3.47, 0.34, -0.49): 'couch', (-5.01, 0.32, 0.52): 'unknown', (-3.95, 0.33, -0.48): 'couch', (-3.43, 0.4, 1.37): 'unknown'}



    compute_metrics(memory, true_labels)

    config = habitat.get_config("configs/tasks/pointnav.yaml")
    env = envs.spring_env.SpringEnv(
        config=config,
        objects = objects
    )

    print("Environment creation successful")
    observations = env.reset()

    print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
        observations["pointgoal_with_gps_compass"][0],
        observations["pointgoal_with_gps_compass"][1]))
    show_observations(cnn, observations)

    print("Agent stepping around inside environment.")


    objects_labels = defaultdict(lambda : defaultdict(list))
    memory = {}
    distance_threshold = 0.3
    softmax_threshold = args.softmax_threshold
    limit = 30
    W = config.SIMULATOR.RGB_SENSOR.WIDTH
    H = config.SIMULATOR.RGB_SENSOR.HEIGHT
    hfov = float(90) * np.pi / 180.
    vfov = 2. * np.arctan(np.tan(hfov / 2) * H / W)
    max_depth = config.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
    count_steps = 0
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    while not env.episode_over:
        action = None
        if args.manual:
            keystroke = cv2.waitKey(0)

            if keystroke == ord(FORWARD_KEY):
                action = HabitatSimActions.MOVE_FORWARD
                print("action: FORWARD")
            elif keystroke == ord(LEFT_KEY):
                action = HabitatSimActions.TURN_LEFT
                print("action: LEFT")
            elif keystroke == ord(RIGHT_KEY):
                action = HabitatSimActions.TURN_RIGHT
                print("action: RIGHT")
            elif keystroke == ord(FINISH):
                action = HabitatSimActions.STOP
                print("action: FINISH")
            elif keystroke == ord(LOOK_UP_KEY):
                action = HabitatSimActions.LOOK_UP
                print("action: LOOK_UP")
            elif keystroke == ord(LOOK_DOWN_KEY):
                action = HabitatSimActions.LOOK_DOWN
                print("action: LOOK_DOWN")
            else:
                print("INVALID KEY")
                continue
            actions.append(action)
        else:
            action = actions[count_steps]

        observations = env.step(action)
        top_down_map_info = env.get_metrics()["top_down_map"]
        top_down_map = fill_map(top_down_map_info, H, memory, env)
        count_steps += 1


        K = np.array([
            [1 / np.tan(hfov / 2.), 0., 0., 0.],
            [0., 1 / np.tan(vfov / 2.), 0., 0.],
            [0., 0.,  -1, 0], # -1 to swap floor/ceil distances
            [0., 0., 0, 1]])
        R = quaternion.as_rotation_matrix(observations["agent_state"].sensor_states["rgb"].rotation)
        cam_pos = observations["agent_state"].sensor_states["rgb"].position
        T = -R.T @ cam_pos
        C = np.eye(4)
        C[0:3,0:3] = R.T
        C[0:3,3] = T.squeeze()
        depth = (observations['depth']*max_depth).squeeze()

        classes, class_names, scores, boxes, masks, centroids = cnn.raw_inference(observations['rgb'])
        memory_labels = {}
        #print(depth.shape)
        #Reproject all
        xs, ys = np.meshgrid(np.linspace(-1,1,W), np.linspace(1,-1,H))
        xs = xs.reshape(H,W)
        ys = ys.reshape(H,W)
        xys = np.vstack((xs * depth , ys * depth, depth, np.ones(depth.shape))).reshape(4,-1)
        xyz = (np.linalg.inv(C) @ np.linalg.inv(K) @ xys)[:3]
        #print("Reprojection shape: {}".format(xyz.shape))
        #print(xyz[:,0])
        for key, value in memory.items():
            print(np.linalg.norm(xyz-[[key[2]], [key[1]], [key[0]]], axis=0).shape)
            closest_pos = np.argmin(np.linalg.norm(xyz-[[key[0]], [key[1]], [key[2]]], axis=0))
            closest_point = xyz[:, np.argmin(np.linalg.norm(xyz-[[key[0]], [key[1]], [key[2]]], axis=0))]
            if np.linalg.norm(closest_point - np.array(key)) < distance_threshold:
                memory_labels[(closest_pos % W, closest_pos // W )] = memory[key]



        #memory_labels = {}
        for i in range(len(centroids)):
            c = centroids[i].astype('int32')

            if c.shape[0] == 1:
                c  = c.squeeze()
            print(c)
            d = depth[c[1]][c[0]]
            pixel_coordinate = np.array([(c[0]/W - 0.5)*d*2, -(c[1]/H - 0.5) * d * 2, d, 1]).reshape(4,-1)
            xyz = (np.linalg.inv(C) @ np.linalg.inv(K) @ pixel_coordinate)

            xyz = tuple(np.round(xyz[:3].T.squeeze(), 2))

            for key, value in objects_labels.items():
                if np.linalg.norm(np.array(key) - np.array(xyz)) < distance_threshold:
                    xyz = key
                    break
            # if xyz in memory:
            #     memory_labels[tuple(c)] = memory[xyz]
            objects_labels[xyz][class_names[i]].append(scores[i])

        # Test of softmax
        for key, value in objects_labels.items():
            # Comment for classification based on the whole history
            # if key in memory:
            #     continue
            counter = Counter()
            for class_name, scores in value.items():
                counter[class_name] += len(scores)
            if sum(counter.values()) > limit:
                print("Object with {} observations found on position {}!".format(limit, key))
                labels = counter.keys()
                indexes = np.arange(len(labels))
                values = np.fromiter(counter.values(), dtype=float)
                norm_values = values/values.sum()
                mean_scores = {class_name: round(np.mean(scores), 2) for class_name, scores in value.items()}
                labels_scores = list(map(lambda l: l + "\n({:0.2f})".format(mean_scores[l]), labels))
                t1 = torch.Tensor(torch.from_numpy(norm_values).cuda().float())
                print("Mean scores: {}".format(list(mean_scores.values())))
                t2 = torch.Tensor(list(mean_scores.values()))
                smax = F.softmax(t1 * t2)
                print("Softmax results: {}".format(smax))
                print("Labels: {}".format(labels))
                if torch.max(smax) > softmax_threshold:
                    memory[key] = list(labels)[torch.argmax(smax)]
                else:
                    memory[key] = "unknown"

        #print(objects_labels)
        print("Memory: {}".format(memory))

        show_observations(cnn, observations, memory_labels, map=top_down_map, memory=memory)

    precision, recall, F1_score = compute_metrics(memory, true_labels)
    print("Episode finished after {} steps.".format(count_steps))

    if (
        action == HabitatSimActions.STOP
        and observations["pointgoal_with_gps_compass"][0] < 0.2
    ):
        print("you successfully navigated to destination point")
    else:
        print("your navigation was unsuccessful")

    metrics = {}
    with open("metrics.json", 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    metrics[str(softmax_threshold)] = [precision, recall, F1_score]
    with open("metrics.json", 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    if args.manual:
        data = {'sequence': actions}
        with open('actions.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    example()
