import tensorflow as tf
from tf_agents.agents.sac import sac_agent, tanh_normal_projection_network
from tf_agents.drivers import py_driver
from tf_agents.keras_layers import inner_reshape
from tf_agents.metrics import py_metrics
from tf_agents.networks import nest_map, sequential
from tf_agents.policies import greedy_policy, py_tf_eager_policy, random_py_policy, tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer, reverb_utils
from tf_agents.specs import tensor_spec
from tf_agents.train import actor, learner, triggers
from tf_agents.train.utils import spec_utils, train_utils
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory as trajectory_lib
from tf_agents.trajectories import trajectory
from tf_agents.typing import types

from dataclasses import dataclass
import datetime, pytz
import enum
import functools
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
from typing import Final, Sequence
from typing import Optional
from typing import Union, cast
os.environ['WRAPT_DISABLE_EXTENSIONS'] = 'true'
from absl import logging
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

data_path = "/burg/home/ssa2206/sbsim_dual_control/smart_control/configs/resources/sb1/" #@param {type:"string"}
metrics_path = "/burg/home/ssa2206/sbsim_dual_control/metrics" #@param {type:"string"}
output_data_path = '/burg/home/ssa2206/sbsim_dual_contarol/output' #@param {type:"string"}
root_dir = "/burg/home/ssa2206/sbsim_dual_control/root" #@param {type:"string"}

time_zone = 'US/Pacific'

# @title Define Observers
class RenderAndPlotObserver:
    """Renders and plots the environment."""
    def __init__(self, render_interval_steps: int = 10, environment=None,):
        self._counter = 0
        self._render_interval_steps = render_interval_steps
        self._environment = environment
        self._cumulative_reward = 0.0
        self._start_time = None
        if self._environment is not None:
            self._num_timesteps_in_episode = (self._environment._num_timesteps_in_episode)
            self._environment._end_timestamp

    def __call__(self, trajectory: trajectory_lib.Trajectory) -> None:
        reward = trajectory.reward
        self._cumulative_reward += reward
        self._counter += 1
        if self._start_time is None:
            self._start_time = pd.Timestamp.now()

        if self._counter % self._render_interval_steps == 0 and self._environment:
            execution_time = pd.Timestamp.now() - self._start_time
            mean_execution_time = execution_time.total_seconds() / self._counter
            clear_output(wait=True)
            if self._environment._metrics_path is not None:
                reader = get_latest_episode_reader(self._environment._metrics_path)
                plot_timeseries_charts(reader, time_zone)

            render_env(self._environment)

class PrintStatusObserver:
    """Prints status information."""

    def __init__(self, status_interval_steps: int = 1, environment=None, replay_buffer=None):
        self._counter = 0
        self._status_interval_steps = status_interval_steps
        self._environment = environment
        self._cumulative_reward = 0.0
        self._replay_buffer = replay_buffer

        self._start_time = None
        if self._environment is not None:
            self._num_timesteps_in_episode = (
                    self._environment._num_timesteps_in_episode
            )
            self._environment._end_timestamp

    def __call__(self, trajectory: trajectory_lib.Trajectory) -> None:

        reward = trajectory.reward
        self._cumulative_reward += reward
        self._counter += 1
        if self._start_time is None:
            self._start_time = pd.Timestamp.now()

        if self._counter % self._status_interval_steps == 0 and self._environment:
            execution_time = pd.Timestamp.now() - self._start_time
            mean_execution_time = execution_time.total_seconds() / self._counter

            sim_time = self._environment.current_simulation_timestamp.tz_convert(
                    time_zone
            )
            percent_complete = int(
                    100.0 * (self._counter / self._num_timesteps_in_episode)
            )

            if self._replay_buffer is not None:
                rb_size = self._replay_buffer.num_frames()
                rb_string = " Replay Buffer Size: %d" % rb_size
            else:
                rb_string = ""

            print(
                    "Step %5d of %5d (%3d%%) Sim Time: %s Reward: %2.2f Cumulative"
                    " Reward: %8.2f Execution Time: %s Mean Execution Time: %3.2fs %s"
                    % (
                            self._environment._step_count,
                            self._num_timesteps_in_episode,
                            percent_complete,
                            sim_time.strftime("%Y-%m-%d %H:%M"),
                            reward,
                            self._cumulative_reward,
                            execution_time,
                            mean_execution_time,
                            rb_string,
                    )
            )


# @title Define an Agent Learner

experience_dataset_fn = lambda: dataset

saved_model_dir = os.path.join(root_dir, learner.POLICY_SAVED_MODEL_DIR)
print('Policies will be saved to saved_model_dir: %s' % saved_model_dir)
env_step_metric = py_metrics.EnvironmentSteps()

def get_learners(agent, train_step, policy_save_interval, triggers):
    learning_triggers = [
        triggers.PolicySavedModelTrigger(
            saved_model_dir,
            agent,
            train_step,
            interval=policy_save_interval,
            metadata_metrics={triggers.ENV_STEP_METADATA_KEY: env_step_metric},
        ),
        triggers.StepPerSecondLogTrigger(train_step, interval=10),
    ]

    agent_learner = learner.Learner(
        root_dir,
        train_step,
        agent,
        experience_dataset_fn,
        triggers=learning_triggers,
        strategy=None,
        summary_interval=learner_summary_interval,
    )

    return(learning_triggers, agent_learner)







### RULES BASED CONTROL SETUP

# @title Utils for RBC

# We're concerned with controlling Heatpumps/ACs and Hot Water Systems (HWS).
class DeviceType(enum.Enum):
    AC = 0
    HWS = 1


SetpointName = str    # Identify the setpoint
# Setpoint value.
SetpointValue = Union[float, int, bool]


@dataclass
class ScheduleEvent:
    start_time: pd.Timedelta
    device: DeviceType
    setpoint_name: SetpointName
    setpoint_value: SetpointValue


# A schedule is a list of times and setpoints for a device.
Schedule = list[ScheduleEvent]
ActionSequence = list[tuple[DeviceType, SetpointName]]



# A schedule is a list of times and setpoints for a device.
Schedule = list[ScheduleEvent]
ActionSequence = list[tuple[DeviceType, SetpointName]]

action_sequence = [
    (DeviceType.HWS, 'supply_water_setpoint'),
    (DeviceType.AC, 'supply_air_heating_temperature_setpoint'),
]


def to_rad(sin_theta: float, cos_theta: float) -> float:
    """Converts a sin and cos theta to radians to extract the time."""

    if sin_theta >= 0 and cos_theta >= 0:
        return np.arccos(cos_theta)
    elif sin_theta >= 0 and cos_theta < 0:
        return np.pi - np.arcsin(sin_theta)
    elif sin_theta < 0 and cos_theta < 0:
        return np.pi - np.arcsin(sin_theta)
    else:
        return 2 * np.pi - np.arccos(cos_theta)

    return np.arccos(cos_theta) + rad_offset


def to_dow(sin_theta: float, cos_theta: float) -> float:
    """Converts a sin and cos theta to days to extract day of week."""
    theta = to_rad(sin_theta, cos_theta)
    return np.floor(7 * theta / 2 / np.pi)


def to_hod(sin_theta: float, cos_theta: float) -> float:
    """Converts a sin and cos theta to hours to extract hour of day."""
    theta = to_rad(sin_theta, cos_theta)
    return np.floor(24 * theta / 2 / np.pi)


def find_schedule_action(schedule: Schedule, device: DeviceType, 
                         setpoint_name: SetpointName, timestamp: pd.Timedelta,) -> SetpointValue:
    """Finds the action for a schedule event for a time and schedule."""

    # Get all the schedule events for the device and the setpoint, and turn it
    # into a series.
    device_schedule_dict = {}
    for schedule_event in schedule:
        if (schedule_event.device == device and schedule_event.setpoint_name == setpoint_name):
            device_schedule_dict[schedule_event.start_time] = (schedule_event.setpoint_value)
    device_schedule = pd.Series(device_schedule_dict)

    # Get the indexes of the schedule events that fall before the timestamp.

    device_schedule_indexes = device_schedule.index[device_schedule.index <= timestamp]

    # If are no events preceedding the time, then choose the last
    # (assuming it wraps around).
    if device_schedule_indexes.empty:
        return device_schedule.loc[device_schedule.index[-1]]
    return device_schedule.loc[device_schedule_indexes[-1]]

weekday_schedule_events = [
    ScheduleEvent(
        pd.Timedelta(6, unit='hour'),
        DeviceType.AC,
        'supply_air_heating_temperature_setpoint',
        292.0,
    ),
    ScheduleEvent(
        pd.Timedelta(19, unit='hour'),
        DeviceType.AC,
        'supply_air_heating_temperature_setpoint',
        285.0,
    ),
    ScheduleEvent(
        pd.Timedelta(6, unit='hour'),
        DeviceType.HWS,
        'supply_water_setpoint',
        350.0,
    ),
    ScheduleEvent(
        pd.Timedelta(19, unit='hour'),
        DeviceType.HWS,
        'supply_water_setpoint',
        315.0,
    ),
]



weekend_holiday_schedule_events = [
    ScheduleEvent(
        pd.Timedelta(6, unit='hour'),
        DeviceType.AC,
        'supply_air_heating_temperature_setpoint',
        285.0,
    ),
    ScheduleEvent(
        pd.Timedelta(19, unit='hour'),
        DeviceType.AC,
        'supply_air_heating_temperature_setpoint',
        285.0,
    ),
    ScheduleEvent(
        pd.Timedelta(6, unit='hour'),
        DeviceType.HWS,
        'supply_water_setpoint',
        315.0,
    ),
    ScheduleEvent(
        pd.Timedelta(19, unit='hour'),
        DeviceType.HWS,
        'supply_water_setpoint',
        315.0,
    ),
]

action_sequence = [
    (DeviceType.HWS, 'supply_water_setpoint'),
    (DeviceType.AC, 'supply_air_heating_temperature_setpoint'),
]


# @title Define a schedule policy

class SchedulePolicy(tf_policy.TFPolicy):
    """TF Policy implementation of the Schedule policy."""
    def __init__(self, time_step_spec, action_spec: types.NestedTensorSpec,
      dow_sin_index: int, dow_cos_index: int, hod_sin_index: int, hod_cos_index: int, 
      action_normalizers,
      local_start_time: str = pd.Timestamp,
      policy_state_spec: types.NestedTensorSpec = (),
      info_spec: types.NestedTensorSpec = (),
      training: bool = False,
      name: Optional[str] = None,
      action_sequence: ActionSequence = action_sequence,
      weekend_holiday_schedule_events: Schedule = weekend_holiday_schedule_events,
      weekday_schedule_events: Schedule = weekday_schedule_events,
                 
    ):
        self.weekday_schedule_events = weekday_schedule_events
        self.weekend_holiday_schedule_events = weekend_holiday_schedule_events
        self.dow_sin_index = dow_sin_index
        self.dow_cos_index = dow_cos_index
        self.hod_sin_index = hod_sin_index
        self.hod_cos_index = hod_cos_index
        self.action_sequence = action_sequence
        self.action_normalizers = action_normalizers
        self.local_start_time = local_start_time
        self.norm_mean = 0.0
        self.norm_std = 1.0

        policy_state_spec = ()

        super().__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            policy_state_spec=policy_state_spec,
            info_spec=info_spec,
            clip=False,
            observation_and_action_constraint_splitter=None,
            name=name,
        )

    def _normalize_action_map(
      self, action_map: dict[tuple[DeviceType, SetpointName], SetpointValue]
    ) -> dict[tuple[DeviceType, SetpointName], SetpointValue]:

        normalized_action_map = {}

        for k, v in action_map.items(): # k = device_name, v = val (315K)
            for normalizer_k, normalizer in self.action_normalizers.items(): # _k is name, normalizer is function
                # applies the right normalizer to the correct setpoint / action step
                # normalizer brings from native range to desired range, i.e. [-1, 1]
                if normalizer_k.endswith(k[1]):
                    normed_v = normalizer.agent_value(v)
                    normalized_action_map[k] = normed_v
        return normalized_action_map

    def _get_action(self, time_step) -> dict[tuple[DeviceType, SetpointName], SetpointValue]:

        observation = time_step.observation
        action_spec = cast(tensor_spec.BoundedTensorSpec, self.action_spec)
        dow_sin = (observation[self.dow_sin_index] * self.norm_std) + self.norm_mean
        dow_cos = (observation[self.dow_cos_index] * self.norm_std) + self.norm_mean
        hod_sin = (observation[self.hod_sin_index] * self.norm_std) + self.norm_mean
        hod_cos = (observation[self.hod_cos_index] * self.norm_std) + self.norm_mean

        dow = to_dow(dow_sin, dow_cos)
        hod = to_hod(hod_sin, hod_cos)

        timestamp = (
            pd.Timedelta(hod, unit='hour') + self.local_start_time.utcoffset()
        )

        if dow < 5:  # weekday
            action_map = {(tup[0], tup[1]): 
                        find_schedule_action(self.weekday_schedule_events, tup[0], tup[1], timestamp)
                        for tup in action_sequence}

        else:  # Weekend
            action_map = {
              (tup[0], tup[1]): find_schedule_action(
                  self.weekend_holiday_schedule_events, tup[0], tup[1], timestamp
              )
              for tup in action_sequence
            }

        return action_map

    def _action(self, time_step, policy_state=None, seed=None):
        del seed
        action_map = self._get_action(time_step)
        normalized_action_map = self._normalize_action_map(action_map)

        action = np.array([normalized_action_map[device_setpoint] 
                           for device_setpoint in action_sequence],dtype=np.float32,)

        t_action = tf.convert_to_tensor(action)
        
        return policy_step.PolicyStep(t_action, (), ())