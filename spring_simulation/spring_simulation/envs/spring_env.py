import numpy as np
from PIL import Image
from habitat_sim.utils.common import d3_40_colors_rgb
from habitat.config import Config
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type, Union
from habitat.core.dataset import Dataset, Episode, EpisodeIterator
import os
import magnum as mn
import habitat
import random
import matplotlib.pyplot as plt


class SpringEnv(habitat.Env):

    def __init__(
        self, config: Config, dataset: Optional[Dataset] = None, objects = []
    ) -> None:
        super(SpringEnv, self).__init__(config, dataset)
        self._sim.sim_config.sim_cfg.enable_physics = True

        for object in objects:
            self.add_object(os.path.join("objects", object[0]), object[1])

    def add_object(self, object_path, position):
        obj_template_manager = self._sim.get_object_template_manager()
        print(os.getcwd())
        template_id = obj_template_manager.load_object_configs(
            str(os.path.join("data", object_path))
        )[0]
        # object_template = obj_template_manager.get_object_template(template_id)
        #template_id.set_scale(np.array([0.005, 0.005, 0.005]))
        id_1 = self._sim.add_object(template_id)
        # rotation= mn.Quaternion.rotation(
        #     mn.Rad(1.47), np.array([-1.0, 0, 0])
        # )
        # self._sim.set_rotation(rotation, id_1)
        self._sim.set_translation(np.array(position), id_1)
        self._sim.set_object_semantic_id(id_1 + 1, id_1)

    def reset(self):
        observations = super(SpringEnv, self).reset()
        observations["agent_state"] = self._sim.get_agent_state(0)
        return observations

    def step(self, action):
        observations = super(SpringEnv, self).step(action)
        observations["agent_state"] = self._sim.get_agent_state(0)
        return observations

    def set_test_scene(self):
        self.add_object("objects/curtain", [1.55, 1.5, 0.2])
        self.add_object("objects/stand", [0.6, 0, -0.8])
        self.add_object("objects/stand", [-0.6, 0, -0.8])
        self.add_object("objects/stand", [-1.8, 0, -0.8])

        self.add_object("objects/stand", [-4.5, 0, -0.8])

        self.add_object("objects/stand", [0.6, 0, 0.8])
        self.add_object("objects/stand", [-0.6, 0, 0.8])
        self.add_object("objects/stand", [-1.8, 0, 0.8])
        self.add_object("objects/stand", [-3.0, 0, 0.8])
        self.add_object("objects/stand", [-4.5, 0, 0.8])

        self.add_object("objects/stand", [-5.0, 0, 0])
