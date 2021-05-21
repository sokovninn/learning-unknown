import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import cv2
import envs.spring_env
import numpy as np
from PIL import Image
from habitat_sim.utils.common import d3_40_colors_rgb


FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
FINISH="f"
LOOK_UP_KEY="u"
LOOK_DOWN_KEY="j"


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def show_observations(observations):
    numpy_horizontal1 = np.hstack((transform_rgb_bgr(observations["rgb"]), transform_rgb_bgr(observations["rgb1"])))

    semantic_img = Image.new("P", (observations['semantic'].shape[1], observations['semantic'].shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((observations['semantic'].flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    semantic_img = np.array(semantic_img)
    print(semantic_img.shape)
    print(observations['depth'].shape)
    depth_image = np.stack((observations['depth'], observations['depth'], observations['depth']), axis=2)[:, :, :, 0]
    depth_image = (255*depth_image).astype(np.uint8)
    print(depth_image.shape)
    numpy_horizontal2 = np.hstack((semantic_img, depth_image))
    numpy_vertical = np.vstack((numpy_horizontal1, numpy_horizontal2))
    cv2.imshow("RGB+RGB1+SEMANTIC+DEPTH", numpy_vertical)
    print(observations["agent_state"])



def example():
    objects = []
    env = envs.spring_env.SpringEnv(
        config=habitat.get_config("configs/tasks/pointnav.yaml"),
        objects = objects,
    )

    print("Environment creation successful")
    observations = env.reset()
    print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
        observations["pointgoal_with_gps_compass"][0],
        observations["pointgoal_with_gps_compass"][1]))
    show_observations(observations)

    print("Agent stepping around inside environment.")

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
        count_steps += 1

        print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
            observations["pointgoal_with_gps_compass"][0],
            observations["pointgoal_with_gps_compass"][1]))
        show_observations(observations)


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
