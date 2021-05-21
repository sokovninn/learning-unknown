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
from evaluate_results import compute_metrics
from datetime import datetime

from scipy.stats import entropy
import scipy

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('../../crow_vision_yolact')
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

COCO_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
            'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush']


def show_observations(cnn, observations, memory_labels={}, map=None, memory=None, out=None):
    labeled_image = cnn.label_image(cv2.cvtColor(observations["rgb"], cv2.COLOR_BGR2RGB))
    #labeled_image = cnn.label_image(transform_rgb_bgr(observations["rgb"]))
    #cv2.imwrite("yolact_habitat_example.png", labeled_image)
    # Show labels from memory
    for pos, text in memory_labels.items():
        cv2.putText(labeled_image, text, pos, cv2.FONT_HERSHEY_DUPLEX, 1, [255, 255, 255], 1)

    if not map is None:
        labeled_image = np.hstack((labeled_image, map))
        if out:
            out.write(labeled_image)
        print(labeled_image.shape)
        #cv2.putText(labeled_image, "Memory:", (650, 270), cv2.FONT_HERSHEY_DUPLEX, 0.6, [0, 0, 0], 1)
        #for i, (key, value) in enumerate(memory.items()):
            #cv2.putText(labeled_image, "{}: {}".format(key, value), (650, 270+(i+1)*15), cv2.FONT_HERSHEY_DUPLEX, 0.4, [0, 0, 0], 1)
    cv2.imshow("Labeled image", labeled_image)

def fill_map(
    topdown_map_info: Dict[str, Any], output_height: int, memory, objects, env, show_labels=True
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

    if show_labels:
        cv2.rectangle(top_down_map, (20, top_down_map.shape[0] // 2 + 35), (top_down_map.shape[1] - 20, top_down_map.shape[0] - 10), [0,0,0], 3)
        cv2.rectangle(top_down_map, (20, top_down_map.shape[0] // 2 + 35), (top_down_map.shape[1] - 20, top_down_map.shape[0] - 10), [255,255,255], -1)

    for i, (object_pos, label) in enumerate(objects.items()):
        map_x, map_y = maps.to_grid(
            object_pos[2],
            object_pos[0],
            top_down_map.shape[0:2],
            sim=env.sim,
        )
        print(map_x, map_y)
        color = (0, 0, 0) if label == "unknown" else (0, 255, 0)
        top_down_map = cv2.circle(top_down_map, (map_y,map_x), radius=12, color=[255,255,255], thickness=3)
        if show_labels:
            cv2.putText(top_down_map, chr(ord('a')+i), (map_y + 20,map_x + 20), cv2.FONT_HERSHEY_DUPLEX, 1, [255, 255, 255], 2)
            cv2.putText(top_down_map, "{}: {}".format(chr(ord('a')+i), label),
                (30, top_down_map.shape[0] // 2 +  35 * (i + 2)), cv2.FONT_HERSHEY_DUPLEX, 1.3, [0, 0, 0], 2)


    for i, (object_pos, label) in enumerate(memory.items()):
        map_x, map_y = maps.to_grid(
            object_pos[2],
            object_pos[0],
            top_down_map.shape[0:2],
            sim=env.sim,
        )
        print(map_x, map_y)
        color = (0, 0, 0) if label == "unknown" else (0, 255, 0)
        top_down_map = cv2.circle(top_down_map, (map_y,map_x), radius=10, color=[0,0,0], thickness=-1)
        top_down_map = cv2.circle(top_down_map, (map_y,map_x), radius=7, color=color, thickness=-1)
        if show_labels:
            cv2.putText(top_down_map, str(i), (map_y + 15,map_x - 15), cv2.FONT_HERSHEY_DUPLEX, 1, [0,0,0], 2)
            cv2.putText(top_down_map, "{}: {}".format(str(i), label),
                (top_down_map.shape[1] // 2, top_down_map.shape[0] // 2 +  35 * (i + 2)), cv2.FONT_HERSHEY_DUPLEX, 1.3, [0, 0, 0], 2)

        #top_down_map = cv2.circle(top_down_map, (map_y,map_x), radius=11, color=(255, 255, 255), thickness=2)

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

def update_memory(clusters_max, clusters_entropy, use_entropy=False, unknown_threshold=0.8,
                  seen_threshold=30, seen_ratio=0.15):
    memory = {}
    for center, class_dets in clusters_max.items():
        total_obsrvations = 0
        seen = len(class_dets["seen"])
        for class_name, scores in class_dets.items():
            total_obsrvations += len(scores)
        if seen > seen_threshold and (total_obsrvations - seen) / total_obsrvations > seen_ratio:
            print(total_obsrvations, seen, (total_obsrvations - seen) / total_obsrvations)
            print("Object with {} observations found on position {}!".format(seen_threshold, center))
            if use_entropy:
                total_mean = np.mean(np.array(clusters_entropy[center]), 0)
                print("Total mean {}{}".format(total_mean.shape, np.sum(total_mean)))
                entr = entropy(total_mean.squeeze()) / np.log(len(total_mean))
                max_class_id = np.argmax(total_mean) - 1
                if max_class_id == -1:
                    max_class = "background"
                else:
                    max_class = COCO_CLASSES[max_class_id]
                print("Class: {}".format(max_class))
                print("Entropy: {}".format(entr))
                if entr < unknown_threshold and max_class != "background":
                    print(np.argmax(total_mean))
                    memory[center] = max_class
                else:
                    memory[center] = "unknown"
            else:
                labels = list(class_dets.keys())
                labels.remove("seen")
                sums_scores = np.array([round(np.sum(scores), 2) for class_name, scores in class_dets.items() if class_name !="seen"])
                norm_sums_scores = sums_scores / np.sum(sums_scores)
                print("Sums results: {}".format(norm_sums_scores))
                print("Labels: {}".format(labels))
                if np.max(norm_sums_scores) > unknown_threshold:
                    memory[center] = labels[np.argmax(norm_sums_scores)]
                else:
                    memory[center] = "unknown"
    return memory


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-a", "--actions_filename", type=str, default=None, help="Filename of the file with actions for automatic dataset creation")
    parser.add_argument("-d", "--dataset_folder", type=str, default='data/spring_dataset', help="Path to the folder with dataset")
    parser.add_argument("-v", "--video", type=str, default=None, help="If given then video is captured with specified name")
    parser.add_argument("-t", "--unknown_threshold", type=float, default=0.8, help="Unknown threshold")
    parser.add_argument("-e", "--use_entropy", type=int, default=0, help="Whether to use entropy or softmax thresholding")
    parser.add_argument("-y", "--yolact_threshold", type=float, default=0.3, help="Yolact threshold")
    parser.add_argument("-c", "--cluster_radius", type=float, default=0.5, help="Cluster radius")
    parser.add_argument("-n", "--seen_threshold", type=float, default=50, help="Number of observations to make first classification")
    parser.add_argument("-sr", "--seen_ratio", type=float, default=0.15, help="Ratio detections/observations to consider an object")
    parser.add_argument("-w", "--model", type=str, default="yolact_base_54_800000.pth", help="Model weights")
    parser.add_argument("-mc", "--model_config", type=str, default="yolact_base_config", help="Model weights config")
    parser.add_argument("-em", "--experiment_name", type=str, default="", help="Name of the experiment")
    parser.add_argument("-cd", "--clusters_detections", type=str, default=None, help="Filename of the ready clusters detections")
    parser.add_argument("-m", "--merge_on", type=int, default=0, help="Enable cluster merge or not")
    parser.add_argument("-sl", "--show_labels", type=int, default=1, help="Whether to show object labels on topdown map or not")


    args = parser.parse_args()
    return args

def main():


    args = get_arguments()
    if "lu" in args.model:
        global COCO_CLASSES
        COCO_CLASSES += ["dino"]

    if args.clusters_detections:
        with open(args.clusters_detections, 'r', encoding='utf-8') as f:
            clusters_detections = json.load(f)
        clusters_max = {eval(key): value for key, value in clusters_detections["clusters_max"].items()}
        clusters_entropy = {eval(key): value for key, value in clusters_detections["clusters_entropy"].items()}
        memory = update_memory(clusters_max, clusters_entropy, args.use_entropy,
                               args.unknown_threshold, args.seen_threshold)
        count_steps = clusters_detections["episode_length"]
        true_labels = clusters_detections["true_labels"]
    else:
        actions = []
        if args.actions_filename:
            with open(args.actions_filename) as f:
                actions = json.load(f)['sequence']
        #global cnn
        cnn = InfTool(weights=os.path.join("/home/nikita/crow_vision_yolact/data/weights", args.model), config=args.model_config,
                      top_k=15, score_threshold=args.yolact_threshold)
        objects = [["ukulele", [0.6, 1.3, -0.8]],
               ["backpack", [-0.6, 1.2, -0.8]],
               ["extension", [-1.8, 1.0, -0.8]],
               ["book",  [-4.5, 1.0, -0.8]],
               ["dino", [-5, 1.2, 0]],
               ["skateboard", [0.6, 1.03, 0.8]],
               ["computer",  [-0.6, 1.15, 0.8]],
               ["keyboard",  [-1.8, 1.0, 0.8]],
               ["shoe",  [-3.0, 1.08, 0.8]],
               ["laptop",  [-4.5, 1.13, 0.8]],]

        true_labels = {tuple(objects[0][1]): 'unknown',
                  tuple(objects[1][1]): 'backpack',
                  tuple(objects[2][1]): 'unknown',
                  tuple(objects[3][1]): 'book',
                  tuple(objects[4][1]): 'unknown',
                  tuple(objects[5][1]): 'skateboard',
                  tuple(objects[6][1]): 'unknown',
                  tuple(objects[7][1]): 'keyboard',
                  tuple(objects[8][1]): 'unknown',
                  tuple(objects[9][1]): 'laptop',
                  }


        #memory =  {(-1.23, 0.37, -0.54): 'car', (-2.8, 0.52, -0.28): 'unknown', (0.75, 0.63, -1.0): 'unknown', (1.13, 0.32, 0.01): 'unknown', (1.14, 0.9, 1.22): 'potted plant', (-0.66, 0.74, 0.99): 'unknown', (-1.09, 0.17, -1.33): 'clock', (-0.18, 0.87, -1.26): 'unknown', (-3.47, 0.34, -0.49): 'couch', (-5.01, 0.32, 0.52): 'unknown', (-3.95, 0.33, -0.48): 'couch', (-3.43, 0.4, 1.37): 'unknown'}
        #compute_metrics(memory, true_labels)
        config = habitat.get_config("configs/tasks/lu_trial.yaml")
        env = envs.spring_env.SpringEnv(
            config=config,
            objects = objects
        )
        env.set_test_scene()

        print("Environment creation successful")
        observations = env.reset()

        show_observations(cnn, observations)

        print("Agent stepping around inside environment.")


        clusters_max = defaultdict(lambda : defaultdict(list))
        clusters_entropy = defaultdict(list)
        memory = {}
        cluster_radius = args.cluster_radius
        W = config.SIMULATOR.RGB_SENSOR.WIDTH
        H = config.SIMULATOR.RGB_SENSOR.HEIGHT
        hfov = float(90) * np.pi / 180.
        vfov = 2. * np.arctan(np.tan(hfov / 2) * H / W)
        max_depth = config.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
        count_steps = 0


        out=None
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        while not env.episode_over:
            action = None
            if not args.actions_filename:
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
            top_down_map = fill_map(top_down_map_info, H, memory, true_labels, env, args.show_labels)

            if args.video and not out:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(args.experiment_name + "_" + args.video, fourcc, 30.0, (W + top_down_map.shape[1], H))
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

            classes, class_names, scores, boxes, masks, centroids, smax = cnn.raw_inference(cv2.cvtColor(observations["rgb"], cv2.COLOR_BGR2RGB))
            #print("Scores: {}".format(scores))
            #print("Smax: {}".format(torch.max(smax, 0)[0]))
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
                if np.linalg.norm(closest_point - np.array(key)) < cluster_radius:
                    memory_labels[(closest_pos % W, closest_pos // W )] = memory[key]

            for key, value in clusters_max.items():
                print(np.linalg.norm(xyz-[[key[2]], [key[1]], [key[0]]], axis=0).shape)
                closest_pos = np.argmin(np.linalg.norm(xyz-[[key[0]], [key[1]], [key[2]]], axis=0))
                closest_point = xyz[:, np.argmin(np.linalg.norm(xyz-[[key[0]], [key[1]], [key[2]]], axis=0))]
                if np.linalg.norm(closest_point - np.array(key)) < cluster_radius / 2 \
                    and np.linalg.norm(closest_point - cam_pos) < 2:
                    #memory_labels[(closest_pos % W, closest_pos // W )] = memory[key]
                    clusters_max[key]["seen"].append(1)



            #memory_labels = {}
            for i in range(len(centroids)):
                # if class_names[i] == "dining table":
                #     continue
                c = centroids[i].astype('int32')

                if c.shape[0] == 1:
                    c  = c.squeeze()
                print(c)
                d = depth[c[1]][c[0]]
                pixel_coordinate = np.array([(c[0]/W - 0.5)*d*2, -(c[1]/H - 0.5) * d * 2, d, 1]).reshape(4,-1)
                xyz = (np.linalg.inv(C) @ np.linalg.inv(K) @ pixel_coordinate)

                xyz = tuple(np.round(xyz[:3].T.squeeze(), 3))
                if xyz[1] < 1:
                    continue

                assigned_to_key = None
                for key, value in clusters_max.items():
                    if np.linalg.norm(np.array(key) - np.array(xyz)) < cluster_radius:
                        #print("Old key: {}".format(key))
                        new_key = np.array(key) + (np.array(xyz) - np.array(key)) / sum([len(v) for k, v in clusters_max.items()])
                        #print("New key: {}".format(new_key))
                        xyz = tuple(np.round(new_key, 3))
                        assigned_to_key = key
                        # xyz = key
                        break
                # if xyz in memory:
                #     memory_labels[tuple(c)] = memory[xyz]
                if assigned_to_key:
                    clusters_max[xyz] = clusters_max.pop(assigned_to_key)
                    clusters_entropy[xyz] = clusters_entropy.pop(assigned_to_key)
                #print("smax : {}".format(torch.sum(smax[:, i])))
                #print(smax[:, i].cpu().numpy())
                clusters_max[xyz][class_names[i]].append(scores[i].tolist())
                clusters_entropy[xyz].append(smax[:, i].detach().cpu().numpy().tolist())


            if args.merge_on:
                cluster_centres = np.array(list(clusters_max.keys()))
                num_centres = len(cluster_centres)
                pdistances = scipy.spatial.distance.pdist(cluster_centres)
                if np.any(pdistances < args.cluster_radius):
                    for i in range(num_centres):
                        flag = False
                        for j in range(i+1, num_centres):
                            entry_id = num_centres * i + j - ((i + 2) * (i + 1)) // 2
                            if pdistances[entry_id] < args.cluster_radius:
                                first_cluster_max = clusters_max.pop(tuple(cluster_centres[i]))
                                second_cluster_max = clusters_max.pop(tuple(cluster_centres[j]))
                                clusters_max[tuple((cluster_centres[i] + cluster_centres[j])/2)] = defaultdict(list, {**first_cluster_max, **second_cluster_max})
                                first_cluster_ent = clusters_entropy.pop(tuple(cluster_centres[i]))
                                second_cluster_ent = clusters_entropy.pop(tuple(cluster_centres[j]))
                                print("MERGE!!!")
                                clusters_entropy[tuple((cluster_centres[i] + cluster_centres[j])/2)] = first_cluster_ent + second_cluster_ent
                                flag = True
                                break
                        if flag:
                            break

            #print("Distances: {} {}".format(scipy.spatial.distance.pdist(cluster_centres), cluster_centres))

            # Test of softmax
            memory = update_memory(clusters_max, clusters_entropy, args.use_entropy,
                                   args.unknown_threshold, args.seen_threshold, args.seen_ratio)

            #print(clusters_max)
            print("Memory: {}".format(memory))


            show_observations(cnn, observations, memory_labels, map=top_down_map, memory=memory, out=out)

        #precision, recall, F1_score = compute_metrics(memory, true_labels)
        #print(clusters_max, clusters_entropy)
        if out:
            out.release()
        cv2.imwrite(args.experiment_name + "_top_down_map.png", top_down_map)
        with open(args.experiment_name + "_clusters_detections.json", 'w', encoding='utf-8') as f:
            json.dump({"args": vars(args),
             "episode_length": count_steps,
             "true_labels": {str(key): value for key, value in true_labels.items()},
             "clusters_max": {str(key): value for key, value in clusters_max.items()},
             "clusters_entropy": {str(key): value for key, value in clusters_entropy.items()},
             }, f, ensure_ascii=False, indent=4)
        if not args.actions_filename:
            data = {'sequence': actions}
            with open(args.experiment_name + '_actions.json', 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        print("Episode finished after {} steps.".format(count_steps))


    with open(args.experiment_name + "_detections.json", 'w', encoding='utf-8') as f:
        json.dump({"args": vars(args),
         "episode_length": count_steps,
         "memory": {str(key): value for key, value in memory.items()},
         "true_labels": {str(key): value for key, value in true_labels.items()},}, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
