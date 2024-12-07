# @title Imports
from dataclasses import dataclass
import datetime, pytz
import enum
import functools
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from plot_utils import *
import time
from typing import Final, Sequence
from typing import Optional
from typing import Union, cast
os.environ['WRAPT_DISABLE_EXTENSIONS'] = 'true'
from absl import logging
import gin
import gin
from matplotlib import patches
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import reverb
import mediapy as media
from IPython.display import clear_output
import sys
sys.path.append("/burg/home/ssa2206/sbsim_dual_control/smart_control/notebooks/")

from smart_control.environment import environment

from smart_control.proto import smart_control_building_pb2, smart_control_normalization_pb2
from smart_control.reward import electricity_energy_cost, natural_gas_energy_cost, setpoint_energy_carbon_reward, setpoint_energy_carbon_regret
from smart_control.simulator import randomized_arrival_departure_occupancy, rejection_simulator_building
from smart_control.simulator import simulator_building, step_function_occupancy, stochastic_convection_simulator
from smart_control.utils import bounded_action_normalizer, building_renderer, controller_reader
from smart_control.utils import controller_writer, conversion_utils, observation_normalizer, reader_lib
from smart_control.utils import writer_lib, histogram_reducer, environment_utils


# @title Set local runtime configurations


def logging_info(*args):
    logging.info(*args)
    print(*args)

data_path = "/burg/home/ssa2206/sbsim_dual_control/smart_control/configs/resources/sb1/" #@param {type:"string"}
metrics_path = "/burg/home/ssa2206/sbsim_dual_control/metrics" #@param {type:"string"}
output_data_path = '/burg/home/ssa2206/sbsim_dual_control/output' #@param {type:"string"}
root_dir = "/burg/home/ssa2206/sbsim_dual_control/root" #@param {type:"string"}


@gin.configurable
def get_histogram_reducer():
    reader = controller_reader.ProtoReader(data_path)

    hr = histogram_reducer.HistogramReducer(
        histogram_parameters_tuples=histogram_parameters_tuples,
        reader=reader,
        normalize_reduce=True,
        )
    return hr

def remap_filepath(filepath) -> str:
    return str(filepath)


# @title Plotting Utities
reward_shift = 0
reward_scale = 1.0
person_productivity_hour = 300.0

KELVIN_TO_CELSIUS = 273.15

def render_env(env: environment.Environment):
    """Renders the environment."""
    building_layout = env.building._simulator._building._floor_plan

    # create a renderer
    renderer = building_renderer.BuildingRenderer(building_layout, 1)

    # get the current temps to render
    # this also is not ideal, since the temps are not fully exposed.
    # V Ideally this should be a publicly accessable field
    temps = env.building._simulator._building.temp

    input_q = env.building._simulator._building.input_q

    # render
    vmin = 285
    vmax = 305
    image = renderer.render(temps, cmap='bwr', vmin=vmin, vmax=vmax, colorbar=False, 
                            input_q=input_q, diff_range=0.5, diff_size=1,).convert('RGB')
    media.show_image(image, title='Environment %s' % env.current_simulation_timestamp)


# @title Utils for importing the environment.

def load_environment(gin_config_file: str):
    """Returns an Environment from a config file."""
    # Global definition is required by Gin library to instantiate Environment.
    global environment  # pylint: disable=global-variable-not-assigned
    with gin.unlock_config():
        gin.parse_config_file(gin_config_file)
        return environment.Environment()  # pylint: disable=no-value-for-parameter


def get_latest_episode_reader(metrics_path: str,) -> controller_reader.ProtoReader:
    episode_infos = controller_reader.get_episode_data(metrics_path).sort_index()
    selected_episode = episode_infos.index[-1]
    episode_path = os.path.join(metrics_path, selected_episode)
    reader = controller_reader.ProtoReader(episode_path)
    return reader

@gin.configurable
def get_histogram_path():
    return data_path


@gin.configurable
def get_reset_temp_values():
    reset_temps_filepath = remap_filepath(
      os.path.join(data_path, "reset_temps.npy")
    )
    return np.load(reset_temps_filepath)


@gin.configurable
def get_zone_path():
    return remap_filepath(
      os.path.join(data_path, "double_resolution_zone_1_2.npy")
    )


@gin.configurable
def get_metrics_path():
    return os.path.join(metrics_path, "metrics")

@gin.configurable
def get_weather_path():
    return remap_filepath(os.path.join(
        data_path, "local_weather_moffett_field_20230701_20231122.csv"
    ))

histogram_parameters_tuples = (
        ('zone_air_temperature_sensor',(285., 286., 287., 288, 289., 290., 291., 292., 293., 294., 295., 296., 297., 298., 299., 300.,301,302,303)),
        ('supply_air_damper_percentage_command',(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)),
        ('supply_air_flowrate_setpoint',( 0., 0.05, .1, .2, .3, .4, .5,  .7,  .9)),
    )

def load_envs(data_path=data_path, metrics_path=None):
        
    time_zone = 'US/Pacific'
    collect_scenario_config = os.path.join(data_path, "sim_config.gin")
    print(collect_scenario_config)
    eval_scenario_config = os.path.join(data_path, "sim_config.gin")
    print(eval_scenario_config)

    collect_env = load_environment(collect_scenario_config)

    # For efficency, set metrics_path to None
    collect_env._occupancy_normalization_constant = 125.0

    eval_env = load_environment(eval_scenario_config)
    # eval_env._label += "_eval"
    eval_env._occupancy_normalization_constant = 125.0
    
    initial_collect_env = load_environment(eval_scenario_config)
    initial_collect_env._occupancy_normalization_constant = 125.0

    collect_env._metrics_path = None
    eval_env._metrics_path = metrics_path
    initial_collect_env._metrics_path = metrics_path

    return(eval_env, collect_env, initial_collect_env)
