import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import cv2
import envs.spring_env
import numpy as np
from PIL import Image
from habitat_sim.utils.common import d3_40_colors_rgb
import json
import os


FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
FINISH="f"
LOOK_UP_KEY="u"
LOOK_DOWN_KEY="j"



def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def show_observations(observations):
    cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))
    print(observations["agent_state"])

def main():
    manual = 0
    objects = [
               #["ukulele", [0.5, 0.4, 0.0]],
               #["backpack", [0.5, 0.3, 0.0]],
               #["extension", [0.5, 0.1, 0.0]],
               #["book",  [0.5, 0.1, 0.0]],
               #["dino", [0.5, 0.3, 0.0]],
               ["skateboard", [0.5, 0.13, 0.0]],
               #["computer",  [0.5, 0.25, 0.0]],
               #["keyboard",  [0.5, 0.1, 0.0]],
               #["shoe",  [0.5, 0.18, 0.0]],
               #["laptop",  [0.5, 0.23, 0.0]],

    ]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join("lu_dataset_habitat", "{}_habitat.mp4".format(objects[0][0])), fourcc, 30.0, (1280, 720))
    env = envs.spring_env.SpringEnv(
        config=habitat.get_config("configs/tasks/object360.yaml"),
        objects = objects,
    )
    #-door, height, y

    print("Environment creation successful")
    observations = env.reset()
    if manual:
        show_observations(observations)

    print("Agent stepping around inside environment.")

    if not manual:
        with open('actions360.json') as f:
            actions = json.load(f)['sequence']
    count_steps = 0
    while not env.episode_over:
        action = None
        if manual:
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
        out.write(transform_rgb_bgr(observations["rgb"]))
        count_steps += 1
        print("Step: {}".format(count_steps))

        if manual:
            show_observations(observations)


    print("Episode finished after {} steps.".format(count_steps))
    out.release()


if __name__ == "__main__":
    main()
