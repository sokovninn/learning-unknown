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


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("test.mp4", fourcc, 30.0, (1262, 480))

def show_observations(cnn, observations, memory_labels={}, map=None, memory=None):
    #print(type(observations["rgb"]))
    #cv2.imshow("image", observations["rgb"])
    #time.sleep(3)
    labeled_image = cnn.label_image(cv2.cvtColor(observations["rgb"], cv2.COLOR_BGR2RGB))
    # Show labels from memory
    for pos, text in memory_labels.items():
        cv2.putText(labeled_image, text, pos, cv2.FONT_HERSHEY_DUPLEX, 1, [255, 255, 255], 1)

    #out.write(labeled_image)
    if not map is None:
        labeled_image = np.hstack((labeled_image, map))
        print(labeled_image.shape)
        cv2.putText(labeled_image, "Memory:", (650, 270), cv2.FONT_HERSHEY_DUPLEX, 0.6, [0, 0, 0], 1)
        for i, (key, value) in enumerate(memory.items()):
            cv2.putText(labeled_image, "{}: {}".format(key, value), (650, 270+(i+1)*15), cv2.FONT_HERSHEY_DUPLEX, 0.4, [0, 0, 0], 1)
    out.write(labeled_image)
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

def example():
    #global cnn
    cnn = InfTool(weights="/home/nikita/crow_vision_yolact/data/weights/lu_base_44_240000.pth", config="lu_base_config",
                  top_k=15, score_threshold=0.15)
    objects = [["ukulele", [0.6, 1.4, -0.8]],
           ["backpack", [-0.6, 1.2, -0.8]],
           ["extension", [-1.8, 1.0, -0.8]],
           ["book",  [-4.5, 1.0, -0.8]],
           ["dino", [-5, 1.2, 0]],
           ["skateboard", [0.6, 1.03, 0.8]],
           ["computer",  [-0.6, 1.15, 0.8]],
           ["keyboard",  [-1.8, 1.0, 0.8]],
           ["shoe",  [-3.0, 1.08, 0.8]],
           ["laptop",  [-4.5, 1.13, 0.8]],]
    env = envs.spring_env.SpringEnv(
        config=habitat.get_config("configs/tasks/pointnav.yaml"),
        objects = objects
    )
    env.add_object("objects/curtain", [1.55, 1.5, 0.2])
    env.add_object("objects/stand", [0.6, 0, -0.8])
    env.add_object("objects/stand", [-0.6, 0, -0.8])
    env.add_object("objects/stand", [-1.8, 0, -0.8])

    env.add_object("objects/stand", [-4.5, 0, -0.8])

    env.add_object("objects/stand", [0.6, 0, 0.8])
    env.add_object("objects/stand", [-0.6, 0, 0.8])
    env.add_object("objects/stand", [-1.8, 0, 0.8])
    env.add_object("objects/stand", [-3.0, 0, 0.8])
    env.add_object("objects/stand", [-4.5, 0, 0.8])

    env.add_object("objects/stand", [-5.0, 0, 0])


    print("Environment creation successful")
    observations = env.reset()


    # top_down_map = maps.get_topdown_map_from_sim(
    #         env.sim, map_resolution=1024
    #     )


    print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
        observations["pointgoal_with_gps_compass"][0],
        observations["pointgoal_with_gps_compass"][1]))
    show_observations(cnn, observations)

    print("Agent stepping around inside environment.")

    objects_labels = defaultdict(lambda : defaultdict(list))
    memory = {}
    distance_threshold = 0.1
    count_steps = 0
    while not env.episode_over:
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

        observations = env.step(action)
        top_down_map_info = env.get_metrics()["top_down_map"]
        top_down_map = fill_map(top_down_map_info, 480, memory, env)
        #cv2.imshow("TOP_DOWN_MAP", top_down_map)
        count_steps += 1

        print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
            observations["pointgoal_with_gps_compass"][0],
            observations["pointgoal_with_gps_compass"][1]))

        W = 640
        H = 480
        hfov = float(90) * np.pi / 180.
        vfov = 2. * np.arctan(np.tan(hfov / 2) * H / W)
        max_depth = 10

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

        classes, class_names, scores, boxes, masks, centroids, smax = cnn.raw_inference(observations['rgb'])
        memory_labels = {}
        #print(depth.shape)
        #Reproject all
        xs, ys = np.meshgrid(np.linspace(-1,1,W), np.linspace(1,-1,H))
        xs = xs.reshape(H,W)
        ys = ys.reshape(H,W)
        xys = np.vstack((xs * depth , ys * depth, depth, np.ones(depth.shape))).reshape(4,-1)
        xyz = (np.linalg.inv(C) @ np.linalg.inv(K) @ xys)[:3]
        print("Reprojection shape: {}".format(xyz.shape))
        print(xyz[:,0])
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

            #env.add_object("objects/banana", xyz[:3].T.squeeze())
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
            if key in memory:
                continue
            counter = Counter()
            limit = 30
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
                if torch.max(smax) > 0.4:
                    memory[key] = list(labels)[torch.argmax(smax)]
                else:
                    memory[key] = "unknown"
                #env.add_object("objects/banana", key)

        #print(objects_labels)
        print("Memory: {}".format(memory))

        show_observations(cnn, observations, memory_labels, map=top_down_map, memory=memory)

    out.release()
    cv2.destroyAllWindows()
    print("Episode finished after {} steps.".format(count_steps))

    if (
        action == HabitatSimActions.STOP
        and observations["pointgoal_with_gps_compass"][0] < 0.2
    ):
        print("you successfully navigated to destination point")
    else:
        print("your navigation was unsuccessful")


if __name__ == "__main__":
    example()
