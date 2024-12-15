import numpy as np
import tensorflow as tf
from tf_agents.trajectories import trajectory
from tf_agents.trajectories import policy_step, StepType

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
        batch_size = inputs.shape[0]
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
    def __init__(self, state_dim, action_dim, hidden_dim=128, stabilize_dynamics=True):
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
        self.stabilize_dynamics = stabilize_dynamics

    def call(self, states, actions):
        next_states = self.state_predictor(states, actions)  # Direct prediction of next state
        # Reshape inputs to (batch_size, timesteps=1, features) and concat along feature dimension
        x = tf.concat([tf.expand_dims(states, axis=1), tf.expand_dims(actions, axis=1)], axis=-1)
        rnn_out, hidden_state, cell_state = self.rnn(x)
        rewards = self.reward_predictor(rnn_out[:, 0, :])
        return next_states, rewards, (hidden_state, cell_state)
    
    def train_step(self, states, actions, next_states, rewards):
        self.state_predictor.update(states, actions, next_states, debug=True)
        if(self.stabilize_dynamics):
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
        all_states = []
        for step, (experience_batch, sample_info) in enumerate(dataset):
            states, actions, rewards = experience_batch.observation, experience_batch.action, experience_batch.reward
            at, atp1 = actions[:,0,:], actions[:, 1, :]
            st, stp1 = states[:,0,:], states[:, 1, :]
            rt = rewards[:, 0]  # Select first timestep rewards
            self.train_step(st, at, stp1, rt)
            all_states.append(st.numpy())
        return(all_states)

    
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
            print("state is nan")
            break
        else:
            time_step = transition(np.array([next_state[0]], dtype=np.float32), reward=reward, discount=current_discount)
            next_action = policy.action(time_step).action # Get action from policy

            generated_experience.append(
                dict(state=current_state.numpy(), action=current_action, reward=reward,
                next_state= next_state.numpy(), discount=current_discount.reshape(1,1))
            )

            current_state, current_reward = next_state, reward
            next_state, reward, (hidden_state, cell_state) = world_model(current_state, current_action)

    return generated_experience



def add_to_replay_buffer(rb_observer, experience_data):
    """
    Adds experience data to the Reverb replay buffer.
    
    Args:
        rb_observer: ReverbAddTrajectoryObserver instance
        experience_data: Dict containing state, action, reward, next_state, discount
    """
    state, action, reward, next_state, discount = experience_data.values()
 
    state = state.reshape(-1) # Remove any extra dimensions
    action = tf.squeeze(action)
    next_state = tf.squeeze(next_state)
    reward = tf.squeeze(reward)
    discount = tf.squeeze(discount)
    traj = trajectory.Trajectory(
        step_type=tf.constant(StepType.MID, dtype=tf.int32),
        observation=tf.constant(state, dtype=tf.float32), # Current observation/state
        action=tf.constant(action, dtype=tf.float32),
        policy_info=(),
        next_step_type=tf.constant(StepType.MID, dtype=tf.int32),
        reward=tf.constant(reward, dtype=tf.float32),
        discount=tf.constant(discount, dtype=tf.float32)
    )
    rb_observer(traj)


 