import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import cv2
import envs.spring_env
import numpy as np
from PIL import Image
from habitat_sim.utils.common import d3_40_colors_rgb
import os, glob
import json
import argparse
import quaternion

from scipy.spatial.transform import Rotation

import sims.actions


FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
FINISH="f"
LOOK_UP_KEY="u"
LOOK_DOWN_KEY="j"
STRAFE_LEFT_KEY="n"
STRAFE_RIGHT_KEY="m"


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def extract_semantic_image(observations):
    semantic_img = Image.new("P", (observations['semantic'].shape[1], observations['semantic'].shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((observations['semantic'].flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    semantic_img = np.array(semantic_img)
    print(observations["agent_state"])
    return semantic_img


def show_observations(observations):
    print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
        observations["pointgoal_with_gps_compass"][0],
        observations["pointgoal_with_gps_compass"][1]))

    numpy_horizontal1 = np.hstack((transform_rgb_bgr(observations["rgb"]), transform_rgb_bgr(observations["rgb1"])))
    semantic_img = extract_semantic_image(observations)
    depth_image = np.stack((observations['depth'], observations['depth'], observations['depth']), axis=2)[:, :, :, 0]
    depth_image = (255*depth_image).astype(np.uint8)
    numpy_horizontal2 = np.hstack((semantic_img, depth_image))
    numpy_vertical = np.vstack((numpy_horizontal1, numpy_horizontal2))
    cv2.imshow("RGB+RGB1+SEMANTIC+DEPTH", numpy_vertical)

def get_action(keystroke):
    action = None
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
    elif keystroke == ord(STRAFE_LEFT_KEY):
        action = HabitatSimActions.STRAFE_LEFT
        print("action: STRAFE_LEFT")
    elif keystroke == ord(STRAFE_RIGHT_KEY):
        action = HabitatSimActions.STRAFE_RIGHT
        print("action: STRAFE_RIGHT")
    else:
        print("INVALID KEY")
    return action


def add_strafe_actions(config):

    HabitatSimActions.extend_action_space("STRAFE_LEFT")
    HabitatSimActions.extend_action_space("STRAFE_RIGHT")

    config.defrost()

    config.TASK.POSSIBLE_ACTIONS = config.TASK.POSSIBLE_ACTIONS + [
        "STRAFE_LEFT",
        "STRAFE_RIGHT",
    ]
    config.TASK.ACTIONS.STRAFE_LEFT = habitat.config.Config()
    config.TASK.ACTIONS.STRAFE_LEFT.TYPE = "StrafeLeft"
    config.TASK.ACTIONS.STRAFE_RIGHT = habitat.config.Config()
    config.TASK.ACTIONS.STRAFE_RIGHT.TYPE = "StrafeRight"
    config.SIMULATOR.ACTION_SPACE_CONFIG = "NoNoiseStrafe"
    config.freeze()

def prepare_directories(dataset_folder):
    folders = {}
    folders["rgb_left_folder"] = os.path.join(dataset_folder, "RGB_left")
    os.makedirs(folders["rgb_left_folder"], exist_ok=True)
    folders["rgb_right_folder"] = os.path.join(dataset_folder, "RGB_right")
    os.makedirs(folders["rgb_right_folder"], exist_ok=True)
    folders["semantic_folder"] = os.path.join(dataset_folder, "semantic")
    os.makedirs(folders["semantic_folder"], exist_ok=True)
    folders["depth_folder"] = os.path.join(dataset_folder, "depth")
    os.makedirs(folders["depth_folder"], exist_ok=True)
    return folders

def cleanup_files(dataset_folder):
    files = glob.glob(os.path.join(dataset_folder,'./*/*'))
    agent_states_path = os.path.join(dataset_folder, 'agent_states.json')
    if os.path.isfile(agent_states_path):
        files.append(agent_states_path)
    for f in files:
        os.remove(f)

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--manual", type=int, default=1, help="True: manual creation, False: automatic creation using actions from file")
    parser.add_argument("-a", "--actions_filename", type=str, default='actions.json', help="Filename of the file with actions for automatic dataset creation")
    parser.add_argument("-d", "--dataset_folder", type=str, default='data/spring_dataset', help="Path to the folder with dataset")
    parser.add_argument("-v", "--video", type=str, default=None, help="If given then video is captured with specified name")


    args = parser.parse_args()
    return args

# main
if __name__ == "__main__":

    args = get_arguments()


    config = habitat.get_config(config_paths="configs/tasks/pointnav.yaml")
    add_strafe_actions(config)

    W = config.SIMULATOR.RGB_SENSOR.WIDTH
    H = config.SIMULATOR.RGB_SENSOR.HEIGHT

    if args.video:
        cap = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(os.path.join(args.dataset_folder, args.video), fourcc, 30.0, (W,H))

    ###############################################
    hfov = float(config.SIMULATOR.DEPTH_SENSOR.HFOV) * np.pi / 180.
    vfov = 2. * np.arctan(np.tan(hfov / 2) * H / W)
    max_depth = config.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
    ###############################################


    actions = []
    agent_states = {}
    if not args.manual:
        with open(os.path.join(args.dataset_folder, args.actions_filename)) as f:
            actions = json.load(f)['sequence']

    folders = prepare_directories(args.dataset_folder)
    cleanup_files(args.dataset_folder)

    objects = []
    # objects = [["trex", [-2.64, 0.70, 1.59]],
    #        ["bird", [-2.64, 0.2, 2.59]],
    #        ["bucket", [-3.64, 0.2, 1.59]],
    #        ["plane", [-3.64, 0.2, 2.59]],
    #        ["banana",  [7.31 , 0.05, -0.28]]]

    # initialize env
    env = envs.spring_env.SpringEnv(
        config=config, objects=objects
    )
    print("Environment creation successful")
    observations = env.reset()
    show_observations(observations)

    print("Agent stepping around inside environment.")
    count_steps = 0
    agent_states[count_steps] = {"position": observations["agent_state"].position.tolist(),
                                 "rotation": np.asarray(observations["agent_state"].rotation,
                                                        dtype=np.quaternion).view((np.double, 4)).tolist()}
    while not env.episode_over:
        action = None
        if args.manual:
            keystroke = cv2.waitKey(0)
            action = get_action(keystroke)
            if action is None:
                continue
            actions.append(action)
        else:
            action = actions[count_steps]

        observations = env.step(action)
        count_steps += 1

        show_observations(observations)
        agent_states[count_steps] = {"position": observations["agent_state"].position.tolist(),
                                     "rotation": np.asarray(observations["agent_state"].rotation,
                                                            dtype=np.quaternion).view((np.double, 4)).tolist()}
        rgb_left_image = cv2.cvtColor(observations['rgb'], cv2.COLOR_RGB2BGR)
        rgb_right_image = cv2.cvtColor(observations['rgb'], cv2.COLOR_RGB2BGR)
        semantic_image = extract_semantic_image(observations)
        depth_image = np.stack((observations['depth'], observations['depth'], observations['depth']), axis=2)[:, :, :, 0]
        depth_image = (255*depth_image).astype(np.uint8)
        if args.video:
            out.write(rgb_left_image)
        cv2.imwrite(os.path.join(folders["rgb_left_folder"], "{}.jpg".format(count_steps)),
                    rgb_left_image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        cv2.imwrite(os.path.join(folders["rgb_right_folder"], "{}.jpg".format(count_steps)),
                    rgb_right_image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        cv2.imwrite(os.path.join(folders["semantic_folder"], "{}.jpg".format(count_steps)),
                    semantic_image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        cv2.imwrite(os.path.join(folders["depth_folder"], "{}.jpg".format(count_steps)),
                    depth_image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

        K = np.array([
            [1 / np.tan(hfov / 2.), 0., 0., 0.],
            [0., 1 / np.tan(vfov / 2.), 0., 0.],
            [0., 0.,  -1, 0], # -1 to swap floor/ceil distances
            [0., 0., 0, 1]])

        #quaternion = [observations["agent_state"].sensor_states["rgb"].rotation.w] + observations["agent_state"].sensor_states["rgb"].rotation.item().vec.tolist()

        R = quaternion.as_rotation_matrix(observations["agent_state"].sensor_states["rgb"].rotation)#Rotation.from_quat(quaternion).as_matrix()
        cam_pos = observations["agent_state"].sensor_states["rgb"].position
        T = -R.T @ cam_pos
        C = np.eye(4)
        C[0:3,0:3] = R.T
        C[0:3,3] = T.squeeze()
        d = depth_image[depth_image.shape[0]//2][depth_image.shape[1]//2][0]/255*max_depth
        pixel_coordinate = np.array([0, 0, d, 1]).reshape(4,-1)

        xyz = np.linalg.inv(C) @ np.linalg.inv(K) @ pixel_coordinate
        env.add_object("objects/banana", xyz[:3].T.squeeze())

        # d = depth_image[depth_image.shape[0]//4][depth_image.shape[1]//2][0]/255*max_depth
        # pixel_coordinate = np.array([0, 0.5*d, d, 1]).reshape(4,-1)
        #
        # xyz = np.linalg.inv(C) @ np.linalg.inv(K) @ pixel_coordinate
        # env.add_object("objects/banana", xyz[:3].T.squeeze())

        print(observations["agent_state"].sensor_states["rgb"].position)
        print("XYZ: {}".format(xyz[:3].T.squeeze()))


    # end

    if args.video:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    with open(os.path.join(args.dataset_folder, "agent_states.json"), 'w', encoding='utf-8') as f:
        json.dump(agent_states, f, ensure_ascii=False, indent=4)

    if args.manual:
        data = {'sequence': actions}
        with open(os.path.join(args.dataset_folder, args.actions_filename), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    print('DATASET WITH LENGTH {} FINISHED'.format(count_steps))
