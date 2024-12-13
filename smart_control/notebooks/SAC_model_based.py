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
sys.path.append("/burg/home/ssa2206/sbsim_dual_control/smart_control/notebooks/")
from smart_control.environment import environment

from smart_control.proto import smart_control_building_pb2, smart_control_normalization_pb2
from smart_control.reward import electricity_energy_cost, natural_gas_energy_cost, setpoint_energy_carbon_reward, setpoint_energy_carbon_regret

from smart_control.simulator import randomized_arrival_departure_occupancy, rejection_simulator_building
from smart_control.simulator import simulator_building, step_function_occupancy, stochastic_convection_simulator

from smart_control.utils import bounded_action_normalizer, building_renderer, controller_reader
from smart_control.utils import controller_writer, conversion_utils, observation_normalizer, reader_lib
from smart_control.utils import writer_lib, histogram_reducer, environment_utils

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
from tf_agents.trajectories import policy_step, transition, StepType
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory as trajectory_lib
from tf_agents.trajectories import trajectory
from tf_agents.typing import types

#from run_utils import RenderAndPlotObserver, PrintStatusObserver, get_learners
from env_loader import load_envs, get_latest_episode_reader, render_env, remap_filepath
#from env_loader import *

def logging_info(*args):
    logging.info(*args)
    print(*args)


RUN_ID = "SAC_model_based"

data_path = "/burg/home/ssa2206/sbsim_dual_control/smart_control/configs/resources/sb1/" #@param {type:"string"}
metrics_path = f"/burg/home/ssa2206/sbsim_dual_control/metrics/{RUN_ID}" #@param {type:"string"}
output_data_path = f'/burg/home/ssa2206/sbsim_dual_control/output/{RUN_ID}' #@param {type:"string"}
root_dir = "/burg/home/ssa2206/sbsim_dual_control/root" #@param {type:"string"}
time_zone = 'US/Pacific'

eval_env, collect_env, initial_collect_env = load_envs(data_path, metrics_path=metrics_path)


def get_trajectory(time_step, current_action: policy_step.PolicyStep):
    """Get the trajectory for the current action and time step."""
    observation = time_step.observation
    action = current_action.action
    policy_info = ()
    reward = time_step.reward
    discount = time_step.discount

    if time_step.is_first():
        return(trajectory.first(observation, action, policy_info, reward, discount))
    elif time_step.is_last():
        return(trajectory.last(observation, action, policy_info, reward, discount))
    else:
        return(trajectory.mid(observation, action, policy_info, reward, discount))

def compute_avg_return(environment, policy, num_episodes=1, time_zone: str = "US/Pacific", 
                       render_interval_steps: int = 24,trajectory_observers=None,):
    """Computes the average return of the policy on the environment.
    Args:
    environment: environment.Environment
    policy: policy.Policy
    num_episodes: total number of eposides to run.
    time_zone: time zone of the environment
    render_interval_steps: Number of steps to take between rendering.
    trajectory_observers: list of trajectory observers for use in rendering.
    """
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        t0 = time.time()
        epoch = t0
        step_id = 0
        execution_times = []
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)

            if trajectory_observers is not None:
                traj = get_trajectory(time_step, action_step)
                for observer in trajectory_observers:
                    observer(traj)

            episode_return += time_step.reward
            t1 = time.time()
            dt = t1 - t0
            episode_seconds = t1 - epoch
            execution_times.append(dt)
            sim_time = environment.current_simulation_timestamp.tz_convert(time_zone)

            print("Step %5d Sim Time: %s, Reward: %8.2f, Return: %8.2f, Mean Step Time:"
                  " %8.2f s, Episode Time: %8.2f s" % (step_id, sim_time.strftime("%Y-%m-%d %H:%M"),
                                                       time_step.reward, episode_return, 
                                                       np.mean(execution_times), episode_seconds,)
                 )
            if (step_id > 0) and (step_id % render_interval_steps == 0):
                if environment._metrics_path:
                    clear_output(wait=True)
                    reader = get_latest_episode_reader(environment._metrics_path)
                    plot_timeseries_charts(reader, time_zone, save_path=metrics_path)
                render_env(environment)

            t0 = t1
            step_id += 1
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return


class DeviceType(enum.Enum):
    AC = 0
    HWS = 1

SetpointName = str 
SetpointValue = Union[float, int, bool]
#hod_cos_index = collect_env._field_names.index('hod_cos_000')
#hod_sin_index = collect_env._field_names.index('hod_sin_000')
#dow_cos_index = collect_env._field_names.index('dow_cos_000')
#dow_sin_index = collect_env._field_names.index('dow_sin_000')


# @title Utilities to configure networks for the RL Agent.
dense = functools.partial(
    tf.keras.layers.Dense,
    activation=tf.keras.activations.relu,
    kernel_initializer='glorot_uniform',
)

def create_fc_network(layer_units):
    return sequential.Sequential([dense(num_units) for num_units in layer_units])

def create_identity_layer():
    return tf.keras.layers.Lambda(lambda x: x)

def create_sequential_critic_network(obs_fc_layer_units, action_fc_layer_units, joint_fc_layer_units):
    """Create a sequential critic network."""
    # Split the inputs into observations and actions.
    def split_inputs(inputs):
        return {'observation': inputs[0], 'action': inputs[1]}

    # Create an observation network.
    obs_network = (
        create_fc_network(obs_fc_layer_units) if obs_fc_layer_units else create_identity_layer()
    )

    # Create an action network.
    action_network = (
        create_fc_network(action_fc_layer_units) if action_fc_layer_units else create_identity_layer()
    )

    # Create a joint network.
    joint_network = (
        create_fc_network(joint_fc_layer_units) if joint_fc_layer_units else create_identity_layer()
    )

    # Final layer.
    value_layer = tf.keras.layers.Dense(1, kernel_initializer='glorot_uniform')

    return sequential.Sequential(
        [
            tf.keras.layers.Lambda(split_inputs),
            nest_map.NestMap({'observation': obs_network, 'action': action_network}),
            nest_map.NestFlatten(),
            tf.keras.layers.Concatenate(),
            joint_network,
            value_layer,
            inner_reshape.InnerReshape(current_shape=[1], new_shape=[]),
        ],
        name='sequential_critic',
    )

class _TanhNormalProjectionNetworkWrapper(
        tanh_normal_projection_network.TanhNormalProjectionNetwork
):
    """Wrapper to pass predefined `outer_rank` to underlying projection net."""

    def __init__(self, sample_spec, predefined_outer_rank=1):
        super(_TanhNormalProjectionNetworkWrapper, self).__init__(sample_spec)
        self.predefined_outer_rank = predefined_outer_rank

    def call(self, inputs, network_state=(), **kwargs):
        kwargs['outer_rank'] = self.predefined_outer_rank
        if 'step_type' in kwargs:
            del kwargs['step_type']
        return super(_TanhNormalProjectionNetworkWrapper, self).call(inputs, **kwargs)

def create_sequential_actor_network(actor_fc_layers, action_tensor_spec):
    """Create a sequential actor network."""

    def tile_as_nest(non_nested_output):
        return tf.nest.map_structure(
                lambda _: non_nested_output, action_tensor_spec
        )

    return sequential.Sequential(
            [dense(num_units) for num_units in actor_fc_layers]
            + [tf.keras.layers.Lambda(tile_as_nest)]
            + [nest_map.NestMap(tf.nest.map_structure(_TanhNormalProjectionNetworkWrapper, 
                                                      action_tensor_spec))])

# @title Set the RL Agent's parameters

# Actor network fully connected layers.
actor_fc_layers = (128, 128)
# Critic network observation fully connected layers.
critic_obs_fc_layers = (128, 64)
# Critic network action fully connected layers.
critic_action_fc_layers = (128, 64)
# Critic network joint fully connected layers.
critic_joint_fc_layers = (128, 64)

batch_size = 256
actor_learning_rate = 3e-4
critic_learning_rate = 3e-4
alpha_learning_rate = 3e-4
gamma = 0.99
target_update_tau= 0.005
target_update_period= 1
reward_scale_factor = 1.0

# Replay params
replay_capacity = 1000000
debug_summaries = True
summarize_grads_and_vars = True

# @title Construct the SAC agent

_, action_tensor_spec, time_step_tensor_spec = spec_utils.get_tensor_specs(collect_env)

actor_net = create_sequential_actor_network(
    actor_fc_layers=actor_fc_layers, action_tensor_spec=action_tensor_spec
)

critic_net = create_sequential_critic_network(
    obs_fc_layer_units=critic_obs_fc_layers,
    action_fc_layer_units=critic_action_fc_layers,
    joint_fc_layer_units=critic_joint_fc_layers,
)

train_step = train_utils.create_train_step()

agent = sac_agent.SacAgent(
    time_step_tensor_spec,
    action_tensor_spec,
    actor_network=actor_net,
    critic_network=critic_net,
    actor_optimizer=tf.keras.optimizers.Adam(learning_rate=actor_learning_rate),
    critic_optimizer=tf.keras.optimizers.Adam(learning_rate=critic_learning_rate),
    alpha_optimizer=tf.keras.optimizers.Adam(learning_rate=alpha_learning_rate),
    target_update_tau=target_update_tau,
    target_update_period=target_update_period,
    td_errors_loss_fn=tf.math.squared_difference,
    gamma=gamma,
    reward_scale_factor=reward_scale_factor,
    gradient_clipping=None,
    debug_summaries=debug_summaries,
    summarize_grads_and_vars=summarize_grads_and_vars,
    train_step_counter=train_step,
)
agent.initialize()

# @title Access the eval and collect policies
eval_policy = agent.policy
collect_policy = agent.collect_policy

policy_save_interval = 1 # Save the policy after every learning step.
learner_summary_interval = 1 # Produce a summary of the critic, actor, and alpha losses after every gradient update step.

# @title Set up the replay buffer
replay_capacity = 50000
table_name = 'uniform_table'
table = reverb.Table(
    table_name,
    max_size=replay_capacity,
    sampler=reverb.selectors.Uniform(),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1),
)

reverb_checkpoint_dir = output_data_path + "/reverb_checkpoint"
reverb_port = None
print('reverb_checkpoint_dir=%s' %reverb_checkpoint_dir)
reverb_checkpointer = reverb.platform.checkpointers_lib.DefaultCheckpointer(
    path=reverb_checkpoint_dir
)
reverb_server = reverb.Server(
    [table], port=reverb_port, checkpointer=reverb_checkpointer
)
logging_info('reverb_server_port=%d' % reverb_server.port)
reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
    agent.collect_data_spec,
    sequence_length=2,
    table_name=table_name,
    local_server=reverb_server,
)
rb_observer = reverb_utils.ReverbAddTrajectoryObserver(reverb_replay.py_client, table_name, sequence_length=2, stride_length=1)
print('num_frames in replay buffer=%d' %reverb_replay.num_frames())

# @title Make a TF Dataset
# Dataset generates trajectories with shape [Bx2x...]
dataset = reverb_replay.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2).prefetch(50)

# @title Define Observers
class RenderAndPlotObserver:
    """Renders and plots the environment."""
    def __init__(self, render_interval_steps: int = 10, environment=None, save_dir=metrics_path):
        self._counter = 0
        self._render_interval_steps = render_interval_steps
        self._environment = environment
        self._cumulative_reward = 0.0
        self._start_time = None
        self._save_dir = save_dir
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

            sim_time = self._environment.current_simulation_timestamp.tz_convert(time_zone)
            percent_complete = int(100.0 * (self._counter / self._num_timesteps_in_episode))

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

initial_collect_render_plot_observer = RenderAndPlotObserver(
    render_interval_steps=144, environment=initial_collect_env
)
initial_collect_print_status_observer = PrintStatusObserver(
    status_interval_steps=1,
    environment=initial_collect_env,
    replay_buffer=reverb_replay,
)
collect_render_plot_observer = RenderAndPlotObserver(
    render_interval_steps=144, environment=collect_env
)
collect_print_status_observer = PrintStatusObserver(
    status_interval_steps=1,
    environment=collect_env,
    replay_buffer=reverb_replay,
)
eval_render_plot_observer = RenderAndPlotObserver(
    render_interval_steps=144, environment=eval_env
)
eval_print_status_observer = PrintStatusObserver(
    status_interval_steps=1, environment=eval_env, replay_buffer=reverb_replay
)

experience_dataset_fn = lambda: dataset

saved_model_dir = os.path.join(root_dir, learner.POLICY_SAVED_MODEL_DIR)
print('Policies will be saved to saved_model_dir: %s' % saved_model_dir)
env_step_metric = py_metrics.EnvironmentSteps()


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

collect_steps_per_treining_iteration = collect_env._num_timesteps_in_episode


# @title Define a TF-Agents Actor for collect and eval
tf_collect_policy = agent.collect_policy
collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
    tf_collect_policy, use_tf_function=True
)
collect_actor = actor.Actor(
    collect_env,
    collect_policy,
    train_step,
    steps_per_run=collect_steps_per_treining_iteration,
    metrics=actor.collect_metrics(1),
    summary_dir=os.path.join(root_dir, learner.TRAIN_DIR),
    summary_interval=1,
    observers=[
        rb_observer,
        env_step_metric,
        collect_print_status_observer,
        collect_render_plot_observer,
    ],
)

tf_greedy_policy = greedy_policy.GreedyPolicy(agent.policy)
eval_greedy_policy = py_tf_eager_policy.PyTFEagerPolicy(
    tf_greedy_policy, use_tf_function=True
)

eval_actor = actor.Actor(
    eval_env,
    eval_greedy_policy,
    train_step,
    episodes_per_run=1,
    metrics=actor.eval_metrics(1),
    summary_dir=os.path.join(root_dir, 'eval'),
    summary_interval=2,
    observers=[rb_observer, eval_print_status_observer, eval_render_plot_observer],
)


class KoopmanRLS(tf.keras.Model):
    def __init__(self, state_dim, action_dim, lambda_f=0.90, initial_variance=1.0):
        """
        Recursive Least Squares Linear World Model
        
        Args:
        - state_dim: Dimensionality of state vector
        - action_dim: Dimensionality of action vector
        - forgetting_factor: Controls how quickly old observations are discounted
        - initial_variance: Initial diagonal value for inverse covariance matrix
        """
        super(KoopmanRLS, self).__init__()
        
        # RLS parameters
        self.state_dim, self.action_dim, self.total_input_dim = state_dim, action_dim, state_dim + action_dim # 53, 2, 55
        self.lambda_factor, self.initial_variance = lambda_f, initial_variance
        self.momentum = 0.4
        self.stability_threshold = 0.99
        # Model parameters
        self.theta = tf.Variable(tf.random.normal([self.total_input_dim, self.state_dim]), trainable=False, dtype=tf.float32)
        self.P = tf.Variable(initial_variance * tf.eye(self.total_input_dim), trainable=False, dtype=tf.float32)

        # Metrics
        self.prediction_errors = []
        self.parameter_uncertainties = []
        self.debug_logs = dict(gains=[], denominators=[], inputs=[], theta=[], P=[])

        # Initialize running statistics
        self.mean = tf.Variable(tf.zeros([self.total_input_dim]), trainable=False, dtype=tf.float32)
        self.std = tf.Variable(tf.ones([self.total_input_dim]), trainable=False, dtype=tf.float32)
        self.first_input = True
    
    def project_dynamics(self):
        """Project dynamics matrix to have singular values <= threshold"""
        s, U, Vh = tf.linalg.svd(self.theta)
        s_clipped = tf.clip_by_value(s, -self.stability_threshold, self.stability_threshold)
        self.theta.assign(tf.matmul(U, tf.matmul(tf.linalg.diag(s_clipped), Vh)))

    
    @property
    def A(self):
        """Extract A matrix from theta"""
        return self.theta[:self.state_dim]

    @property
    def B(self):
        """Extract B matrix from theta"""
        return self.theta[self.state_dim:]
    
    def call(self, states, actions):
        """
        Predict next states using current A and B matrices
        Args:
        - states: Current states (batch_size, state_dim)
        - actions: Actions taken (batch_size, action_dim)
        """
        # normalize inputs, apply theta, denormalize
        inputs = tf.concat([states, actions], axis=-1)
        normalized_inputs = (inputs - self.mean) / self.std
        output = (normalized_inputs @ self.theta) * self.std[:self.state_dim] + self.mean[:self.state_dim]
        return output
    
    def reset(self):
        """Reset model parameters and statistics."""
        self.theta.assign(tf.random.normal([self.total_input_dim, self.state_dim]))
        self.P.assign(self.initial_variance * tf.eye(self.total_input_dim))
        self.mean.assign(tf.zeros([self.total_input_dim]))
        self.variance.assign(tf.ones([self.total_input_dim]))
        self.prediction_errors = []
        self.parameter_uncertainties = []
    
    def update_running_stats(self, inputs):
        new_mean = tf.reduce_mean(inputs, axis=0)
        new_std = tf.sqrt(tf.reduce_mean(tf.square(inputs - new_mean), axis=0))
        if(not self.first_input):
            new_mean = self.momentum * self.mean + (1 - self.momentum) * new_mean
            new_std = self.momentum * self.std + (1 - self.momentum) * new_std
        else:
            self.first_input = False
        self.mean.assign(new_mean)
        self.std.assign(new_std + 1e-5)
        return (inputs - self.mean) / self.std 
    
    def update(self, states, actions, next_states, debug=False):
        """
        Vectorized RLS update for batch inputs
        
        Args:
        - states: Batch of current states [batch_size, state_dim]
        - actions: Batch of actions [batch_size, action_dim]
        - next_states: Batch of next states [batch_size, state_dim]
        """
        inputs = tf.concat([states, actions], axis=-1) # (batch_size, total_input_dim)
        inputs = self.update_running_stats(inputs)
        predictions = inputs @ self.theta # (batch_size, total_input_dim) * (total_input_dim, state_dim) = (batch_size, state_dim)
        # Normalize predictions before computing innovation
        normalized_next_states = (next_states - self.mean[:self.state_dim]) / self.std[:self.state_dim]
        innovation = normalized_next_states - predictions # (prediction error) [batch_size, state_dim]
        
        P_X = self.P @ tf.transpose(inputs) # (total_input_dim, total_input_dim) * (total_input_dim, batch_size) = (total_input_dim, batch_size)
        # Denominator term (λ + X @ P @ X^T) [batch_size, batch_size]
        R = inputs @ P_X # (batch_size, total_input_dim) * (total_input_dim, batch_size) = (batch_size, batch_size)
        denominator = self.lambda_factor * tf.eye(batch_size) + R # (batch_size, batch_size)
        kalman_gain = P_X @ tf.linalg.inv(denominator) # (total_input_dim * batch_size) * (batch_size, batch_size)
        self.theta.assign_add(kalman_gain @ innovation)
        
        P_new = (self.P - (kalman_gain @ inputs @ self.P)) / self.lambda_factor
        self.P.assign(P_new)         # Update precision matrix
        
        # Store metrics

        if debug:
            self.prediction_errors.append(tf.reduce_mean(tf.square(innovation)).numpy())
            self.parameter_uncertainties.append(tf.linalg.trace(self.P).numpy())
            self.debug_logs['gains'].append(kalman_gain.numpy()     )
            self.debug_logs['denominators'].append(denominator.numpy())
            self.debug_logs['inputs'].append(inputs.numpy())
            self.debug_logs['theta'].append(self.theta.numpy())
            self.debug_logs['P'].append(self.P.numpy())
        return innovation

    def save_weights(self, filepath):
        """Save model parameters."""
        weights = {
            'theta': self.theta.numpy(),
            'P': self.P.numpy(),
            'mean': self.mean.numpy(),
            'variance': self.variance.numpy()
        }
        np.save(filepath, weights)

    def load_weights(self, filepath):
        """Load model parameters."""
        weights = np.load(filepath, allow_pickle=True).item()
        self.theta.assign(weights['theta'])
        self.P.assign(weights['P'])
        self.mean.assign(weights['mean'])
        self.variance.assign(weights['variance'])

class LinearWorldModel(tf.keras.Model):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(LinearWorldModel, self).__init__()
        # RNN for state transitions
        self.rnn = tf.keras.layers.LSTM(hidden_dim, return_sequences=True, return_state=True)
        # Networks for predicting next state and reward
        self.state_predictor = KoopmanRLS(state_dim, action_dim, lambda_f=0.95, initial_variance=1.0)
        self.reward_predictor = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    def call(self, states, actions):
        next_states = self.state_predictor(states, actions)  # Direct prediction of next state
        # Reshape inputs to (batch_size, timesteps=1, features) and concat along feature dimension
        x = tf.concat([tf.expand_dims(states, axis=1), tf.expand_dims(actions, axis=1)], axis=-1)
        rnn_out, hidden_state, cell_state = self.rnn(x)
        rewards = self.reward_predictor(rnn_out[:, 0, :])
        return next_states, rewards, (hidden_state, cell_state)
    
    def train_step(self, states, actions, next_states, rewards):
        self.state_predictor.update(states, actions, next_states, debug=True)
        self.state_predictor.project_dynamics()
        with tf.GradientTape() as tape:
            # Get predictions (remove time dimension for single step prediction)
            next_state_preds, pred_rewards, info = self(states, actions)
            reward_loss = tf.reduce_mean(tf.square(pred_rewards - tf.expand_dims(rewards, -1)))
            grads = tape.gradient(reward_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
    
    def train(self, replay_buffer, batch_size=256, training_steps=64):
        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=batch_size,
            num_steps=2
        ).take(training_steps)
        #all_states = []
        for step, (experience_batch, sample_info) in enumerate(dataset):
            states, actions, rewards = experience_batch.observation, experience_batch.action, experience_batch.reward
            at, atp1 = actions[:,0,:], actions[:, 1, :]
            st, stp1 = states[:,0,:], states[:, 1, :]
            #all_states.append(st.numpy())
            rt = rewards[:, 0]  # Select first timestep rewards
            self.train_step(st, at, stp1, rt)
#return(all_states)
           
def generate_rollouts(world_model, initial_state, initial_reward, initial_discount, policy, rollout_length):
    """Generate model-based rollouts with uncertainty estimation."""
    # Convert initial state to tensor and add batch dimension
    current_state = tf.expand_dims(tf.convert_to_tensor(initial_state, dtype=tf.float32), 0)
    current_reward = initial_reward
    current_discount = initial_discount
    generated_experience = []

    time_step = transition(np.array([current_state[0]], dtype=np.float32), reward=current_reward, discount=current_discount)
    action_step = policy.action(time_step)
    current_action = action_step.action

    next_state, reward, (hidden_state, cell_state) = world_model(current_state, current_action)
    #state_mag = [mag(next_state)]
    for i in range(rollout_length):
        time_step = transition(np.array([next_state[0]], dtype=np.float32), reward=reward, discount=current_discount)
        next_action = policy.action(time_step).action # Get action from policy

        generated_experience.append(
            dict(state=current_state.numpy(), action=current_action, reward=reward,
            next_state= next_state.numpy(), discount=current_discount.reshape(1,1))
        )

        # Update current state
        current_state, current_reward = next_state, reward
        next_state, reward, (hidden_state, cell_state) = world_model(current_state, current_action)
        #next_state = safe_normalize_state(next_state, current_state, current_action)
        #state_mag.append(mag(next_state))
    return generated_experience, #state_mag

def generate_rollouts(world_model, initial_state, initial_reward, initial_discount, policy, rollout_length):
    """Generate model-based rollouts with uncertainty estimation."""
    # Convert initial state to tensor and add batch dimension
    current_state = tf.expand_dims(tf.convert_to_tensor(initial_state, dtype=tf.float32), 0)
    current_reward = initial_reward
    current_discount = initial_discount
    generated_experience = []

    time_step = transition(np.array([current_state[0]], dtype=np.float32), reward=current_reward, discount=current_discount)
    action_step = policy.action(time_step)
    current_action = action_step.action

    next_state, reward, (hidden_state, cell_state) = world_model(current_state, current_action)

    for i in range(rollout_length):
        if(np.isnan(reward).any()):
            print(f"nan at step {i}")
            
        else:
            time_step = transition(np.array([next_state[0]], dtype=np.float32), reward=reward, discount=current_discount)
            next_action = policy.action(time_step).action # Get action from policy

            generated_experience.append(
                dict(state=current_state.numpy(), action=current_action, reward=reward,
                next_state= next_state.numpy(), discount=current_discount.reshape(1,1))
            )

            # Update current state
            current_state, current_reward = next_state, reward
            next_state, reward, (hidden_state, cell_state) = world_model(
                current_state, current_action
            )
    return generated_experience

def add_to_replay_buffer(rb_observer, experience_data):
    """
    Adds experience data to the Reverb replay buffer.
    
    Args:
        rb_observer: ReverbAddTrajectoryObserver instance
        experience_data: Dict containing state, action, reward, next_state, discount
    """
    state, action, reward, next_state, discount = experience_data.values()
    # print(f"State shape: {state.shape}")
    # print(f"Action shape: {action.shape}")
    # print(f"Next state shape: {next_state.shape}")
    # print(f"Reward shape: {reward.shape}")
 
    state = state.reshape(-1) # Remove any extra dimensions
    action = tf.squeeze(action)
    next_state = tf.squeeze(next_state)
    reward = tf.squeeze(reward)
    discount = tf.squeeze(discount)
    # print(f"State shape: {state.shape}")
    # print(f"Action shape: {action.shape}")
    # print(f"Next state shape: {next_state.shape}")
    # print(f"Reward shape: {reward.shape}")
    
    # Create a Trajectory object with the required fields
    traj = trajectory.Trajectory(
        step_type=tf.constant(StepType.MID, dtype=tf.int32),
        observation=tf.constant(state, dtype=tf.float32), # Current observation/state
        action=tf.constant(action, dtype=tf.float32),
        policy_info=(),
        next_step_type=tf.constant(StepType.MID, dtype=tf.int32),
        reward=tf.constant(reward, dtype=tf.float32),
        discount=tf.constant(discount, dtype=tf.float32)
    )
    
    # Add trajectory to replay buffer
    rb_observer(traj)




# @title Execute the training loop

num_training_iterations = 10
num_gradient_updates_per_iter = 100
rollout_length = int(4032/16)

# Collect the performance results with teh untrained model.
eval_actor.run_and_log()
logging_info('Training.')

# @title Modified Training Loop with World Model
# Initialize world model
state_dim = collect_env.observation_spec().shape[0]
action_dim = collect_env.action_spec().shape[0]
world_model = LinearWorldModel(state_dim, action_dim)


# log_dir = root_dir + '/train'
# with tf.summary.create_file_writer(log_dir).as_default() as writer:   


for iter in range(num_training_iterations):
    print('Training iteration: ', iter)

    # Collect real experiences
    collect_actor.run()
    world_model.train(reverb_replay)
    # Generate synthetic experiences
    initial_state, initial_reward, initial_discount = ts.observation, ts.reward, ts.discount
    synthetic_rollouts = generate_rollouts(world_model, initial_state, 
                                            initial_reward, initial_discount, collect_policy, rollout_length)
    #rollout_length *= 2
    for exp in synthetic_rollouts:
        add_to_replay_buffer(rb_observer, exp)


    # Train agent with both real and synthetic data
    loss_info = agent_learner.run(iterations=num_gradient_updates_per_iter)

    logging_info(
        'Actor Loss: %6.2f, Critic Loss: %6.2f, Alpha Loss: %6.2f '
        % (
            loss_info.extra.actor_loss.numpy(),
            loss_info.extra.critic_loss.numpy(),
            loss_info.extra.alpha_loss.numpy(),
        )
    )

    # Evaluate
    eval_actor.run_and_log()

rb_observer.close()
reverb_server.stop()