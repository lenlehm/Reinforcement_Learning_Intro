import gym
import sys
import logging
import numpy as np
import tensorflow as tf
from scipy.stats import beta
import matplotlib.pyplot as plt
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko

## TODO: Beta Distribution to get limited output
## TODO: Refactor the Train method in several functions
class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits):
        # sample a random categorical action from given logits
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

class BetaDistribution(tf.keras.Model):
    def call(self, alpha, beta, x, maxi, mini)
        y = ( ( (maxi - mini) * (x - mini) ) / (maxi - mini) ) + mini
        return np.random.beta(a=alpha, b=beta) #beta.pdf(logits, alpha, beta)

class Model(tf.keras.Model):
    def __init__(self, num_actions):
        #super().__init__('mlp_policy')
        super(Model, self).__init__()
        # no tf.get_variable(), just simple Keras API
        self.hidden1 = kl.Dense(128, activation='relu')
        self.hidden2 = kl.Dense(128, activation='relu')
        self.value = kl.Dense(1, name='value')
        # logits are unnormalized log probabilities
        self.logits = kl.Dense(num_actions, name='policy_logits')
        self.dist = ProbabilityDistribution()

    def call(self, inputs):
        # inputs is a numpy array, convert to Tensor
        x = tf.convert_to_tensor(inputs)
        # separate hidden layers from the same input tensor
        hidden_logs = self.hidden1(x)
        hidden_vals = self.hidden2(x)
        return self.logits(hidden_logs), self.value(hidden_vals)

    def action_value(self, obs):
        # executes call() under the hood
        logits, value = self.predict(obs)
        action = self.dist.predict(logits) # get which action to take
        # action = tf.random.categorical(logits, 1)
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)
        
        
class A2CAgent:
    def __init__(self, model, buff_size = 100000):
        # hyperparameters for loss terms, gamma is the discount coefficient
        self.params = {
            'gamma': 0.99,
            'value': 0.5,
            'entropy': 0.0001
        }
        self.model = model
        self.model.compile(
            optimizer=ko.Adam(lr=0.0007),
            # define separate losses for policy logits and value estimate
            loss=[self._logits_loss, self._value_loss]
        )
        self.buffer = ReplayBuffer(buff_size, mini_batch=32)
    
    def train(self, env, batch_sz=32, updates=1000):
        # storage helpers for a single batch of data
        actions = np.empty((batch_sz,), dtype=np.int32) # (32, )
        rewards, dones, values = np.empty((3, batch_sz)) # (3, 32) - ever single has shape (32, )
        observations = np.empty((batch_sz,) + env.observation_space.shape) # (32, 4)
        # training loop: collect samples, send to optimizer, repeat updates times
        ep_rews = [0.0]
        next_obs = env.reset()
        for update in range(updates):
            for step in range(batch_sz):
                observations[step] = next_obs.copy() # fill observations from environment
                actions[step], values[step] = self.model.action_value(next_obs[None, :]) # get action and its value
                next_obs, rewards[step], dones[step], _ = env.step(actions[step]) # perform next step to get new observation, and reward
                self.buffer.store(next_obs, actions[step], rewards[step], dones[step]) # store trajectory in buffer

                ep_rews[-1] += rewards[step] # only update last column
                if dones[step]:
                    ep_rews.append(0.0) # create new column
                    next_obs = env.reset()
                    logging.info("Episode: %03d, Reward: %03d" % (len(ep_rews)-1, ep_rews[-2]))

            _, next_value = self.model.action_value(next_obs[None, :]) # get next value (int) of action
            #rewi = [x[2] for x in (self.buffer.experience[:self.buffer.current_index])] # extract rewards from buffer
            returns, advs = self._returns_advantages(rewards, dones, values, next_value)
            # returns are cumulative rewards and advantages are returns - baseline
            acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1) # (32, 2) (action, advantage)
            # performs a full training step on the collected batch
            # note: no need to mess around with gradients, Keras API handles it
            losses = self.model.train_on_batch(observations, [acts_and_advs, returns])
            logging.debug("[%d/%d] Losses: %s" % (update+1, updates, losses))
        return ep_rews

    def _returns_advantages(self, rewards, dones, values, next_value):
        # next_value is the bootstrap value estimate of a future state (the critic)
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1) # (33, ) - 32 zeroes and 1 last, value
        # returns are calculated as discounted sum of future rewards
        for t in reversed(range(rewards.shape[0])): # batch_size range - t starts at 31
            returns[t] = rewards[t] + self.params['gamma'] * returns[t+1] * (1-dones[t])
        returns = returns[:-1] # get rid of last element
        # advantages are returns - baseline (= value estimates in our case)
        advantages = returns - values # (32, ) = len(32) - len(32)
        return returns, advantages
    
    def _value_loss(self, returns, value): # V(s)
        # value loss is typically MSE between value estimates and returns
        return self.params['value']*kls.mean_squared_error(returns, value)

    def _logits_loss(self, acts_and_advs, logits): # Q(a, s)
        # a trick to input actions and advantages through same API
        actions, advantages = tf.split(acts_and_advs, 2, axis=-1)
        # sparse categorical CE loss obj that supports sample_weight arg on call()
        # from_logits argument ensures transformation into normalized probabilities
        weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)
        # policy loss is defined by policy gradients, weighted by advantages
        # note: we only calculate the loss on the actions we've actually taken
        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
        # entropy loss can be calculated via CE over itself
        entropy_loss = kls.categorical_crossentropy(logits, logits, from_logits=True)
        # here signs are flipped because optimizer minimizes
        return policy_loss - self.params['entropy']*entropy_loss


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    env = gym.make('CartPole-v0')
    model = Model(num_actions=env.action_space.n)
    agent = A2CAgent(model)

    rewards_history = agent.train(env)
    print("Finished training.")
    print("Total Episode Reward: %d out of 200" % agent.test(env, True))
    
    plt.style.use('seaborn')
    plt.plot(np.arange(0, len(rewards_history), 25), rewards_history[::25])
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()