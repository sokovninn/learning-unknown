#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Optional, Union

import numpy as np
from gym import spaces
import os

import habitat_sim
from habitat.core.dataset import Episode
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import (
    AgentState,
    Config,
    DepthSensor,
    Observations,
    RGBSensor,
    SemanticSensor,
    SensorSuite,
    ShortestPathPoint,
    Simulator,
)
from habitat.core.spaces import Space

from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.sims.habitat_simulator.habitat_simulator import overwrite_config, check_sim_obs

RGBSENSOR_DIMENSION = 3


@registry.register_sensor
class HabitatSimRGBSensor1(RGBSensor):
    sim_sensor_type: habitat_sim.SensorType

    def __init__(self, config):
        self.sim_sensor_type = habitat_sim.SensorType.COLOR
        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.config.HEIGHT, self.config.WIDTH, RGBSENSOR_DIMENSION),
            dtype=np.uint8,
        )

    def get_observation(self, sim_obs):
        obs = sim_obs.get(self.uuid, None)
        check_sim_obs(obs, self)

        # remove alpha channel
        obs = obs[:, :, :RGBSENSOR_DIMENSION]
        return obs

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "rgb1"



@registry.register_simulator(name="SpringSim-v0")
class SpringSim(HabitatSim):
    r"""Simulator wrapper over habitat-sim

    habitat-sim repo: https://github.com/facebookresearch/habitat-sim

    Args:
        config: configuration for initializing the simulator.
    """

    def create_sim_config(
        self, _sensor_suite: SensorSuite
    ) -> habitat_sim.Configuration:
        sim_config = super(SpringSim, self).create_sim_config(_sensor_suite)
        sim_config.sim_cfg.enable_physics = True
        for sim_sensor_cfg in sim_config.agents[0].sensor_specifications:
            for sensor in _sensor_suite.sensors.values():
                if sim_sensor_cfg.uuid == sensor.uuid:
                    sim_sensor_cfg.position = sensor.config.POSITION
                    sim_sensor_cfg.orientation = sensor.config.ORIENTATION
        # sim_config.physics_config_file = "./data/default.phys_scene_config.json"

        return sim_config
