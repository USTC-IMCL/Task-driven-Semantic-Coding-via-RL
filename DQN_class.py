"""
This part of code is the Deep Q Network (DQN) brain.
view the tensorboard picture about this DQN structure on: https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-3-DQN3/#modification
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
Tensorflow: r1.2
"""

import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.0001,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=50000,
            batch_size=64,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.1
        self.sess = tf.Session(config=config)
        self.saver = tf.train.Saver()

        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.s1 = tf.placeholder(tf.float32, [None, 64, 64, 2], name='s1')  # input State
        self.s2 = tf.placeholder(tf.float32, [None, 15, ], name='s2')
        self.s1_ = tf.placeholder(tf.float32, [None, 64, 64, 2], name='s1_')  # input Next State
        self.s2_ = tf.placeholder(tf.float32, [None, 15, ], name='s2_')
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        w_init = tf.random_normal_initializer(stddev=0.02)
        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            eval_in = InputLayer(self.s1, name='eval/in')
            eval_h0 = Conv2d(eval_in, 32, (5, 5), (2, 2), act=lambda x: tl.act.lrelu(x, 0.25),
                             padding='SAME', W_init=w_init, name='eval/h0/conv2d')  # 32x32
            eval_h1 = Conv2d(eval_h0, 64, (3, 3), (2, 2), act=lambda x: tl.act.lrelu(x, 0.25),
                             padding='SAME', W_init=w_init, name='eval/h1/conv2d')  # 16x16
            eval_h2 = Conv2d(eval_h1, 128, (3, 3), (2, 2), act=lambda x: tl.act.lrelu(x, 0.25),
                             padding='SAME', W_init=w_init, name='eval/h2/conv2d')
            eval_h3 = Conv2d(eval_h2, 128, (3, 3), (2, 2), act=lambda x: tl.act.lrelu(x, 0.25),
                             padding='SAME', W_init=w_init, name='eval/h3/conv2d')
            eval_h4 = FlattenLayer(eval_h3, name='eval/h5/flatten')
            eval_h5 = DenseLayer(eval_h4, n_units=256, act=lambda x: tl.act.lrelu(x, 0.25),
                                 W_init=w_init, name='eval/h5/dense')
            eval_h5 = DropoutLayer(eval_h5, keep=0.5, name='drop1')
            eval_h6 = DenseLayer(eval_h5, n_units=64, act=lambda x: tl.act.lrelu(x, 0.25),
                                 W_init=w_init, name='eval/h6/dense')
            eval_h6 = DropoutLayer(eval_h6, keep=0.5, name='drop2')
            eval_in_ = InputLayer(self.s2, name='eval/in_')
            #eval_h7 = FlattenLayer(eval_in_, name='eval/h7/flatten')
            eval_h7 = DenseLayer(eval_in_, n_units=32, act=lambda x: tl.act.lrelu(x, 0.25),
                                 W_init=w_init, name='eval/h7/dense')
            eval_h7 = DropoutLayer(eval_h7, keep=0.5, name='drop3')
            # eval_h8 = tl.layers.ConcatLayer(layer=[eval_h6, eval_h7], concat_dim=1, name='concat_input_layer')
            eval_h8 = tl.layers.ConcatLayer(prev_layer=[eval_h6, eval_h7], concat_dim=1, name='concat_input_layer')
            eval_h9 = DenseLayer(eval_h8, n_units=64, act=tf.identity,
                                 W_init=w_init, name='eval/h9/dense')
            eval_h9 = DropoutLayer(eval_h9, keep=0.5, name='drop4')
            self.eval_h10 = DenseLayer(eval_h9, n_units=self.n_actions, act=tf.identity,
                                       W_init=w_init, name='eval/h10/dense')

            self.q_eval = self.eval_h10.outputs

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            target_in = InputLayer(self.s1_, name='target/in')
            target_h0 = Conv2d(target_in, 32, (5, 5), (2, 2), act=lambda x: tl.act.lrelu(x, 0.25),
                               padding='SAME', W_init=w_init, name='target/h0/conv2d')  # 32x32
            target_h1 = Conv2d(target_h0, 64, (3, 3), (2, 2), act=lambda x: tl.act.lrelu(x, 0.25),
                               padding='SAME', W_init=w_init, name='target/h1/conv2d')  # 16x16
            target_h2 = Conv2d(target_h1, 128, (3, 3), (2, 2), act=lambda x: tl.act.lrelu(x, 0.25),
                               padding='SAME', W_init=w_init, name='target/h2/conv2d')
            target_h3 = Conv2d(target_h2, 128, (3, 3), (2, 2), act=lambda x: tl.act.lrelu(x, 0.25),
                               padding='SAME', W_init=w_init, name='target/h3/conv2d')
            target_h4 = FlattenLayer(target_h3, name='target/h5/flatten')
            target_h5 = DenseLayer(target_h4, n_units=256, act=lambda x: tl.act.lrelu(x, 0.25),
                                   W_init=w_init, name='target/h5/dense')
            target_h6 = DenseLayer(target_h5, n_units=64, act=lambda x: tl.act.lrelu(x, 0.25),
                                   W_init=w_init, name='target/h6/dense')
            target_in_ = InputLayer(self.s2_, name='target/in_')
            #target_h7 = FlattenLayer(target_in_, name='target/h7/flatten')
            target_h7 = DenseLayer(target_in_, n_units=32, act=lambda x: tl.act.lrelu(x, 0.25),
                                   W_init=w_init, name='target/h7/dense')
            # target_h8 = tl.layers.ConcatLayer(layer=[target_h6, target_h7], concat_dim=1, name='concat_input_layer')
            target_h8 = tl.layers.ConcatLayer(prev_layer=[target_h6, target_h7], concat_dim=1, name='concat_input_layer')
            target_h9 = DenseLayer(target_h8, n_units=64, act=tf.identity,
                                   W_init=w_init, name='target/h9/dense')
            target_h10 = DenseLayer(target_h9, n_units=self.n_actions, act=tf.identity,
                                    W_init=w_init, name='target/h10/dense')

            self.q_next = target_h10.outputs

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')  # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)  # shape=(None, )
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation1 = np.reshape(observation[:4096 * 2], (64, 64, 2))
        observation2 = np.reshape(observation[4096 * 2:], (15,))

        observation1 = observation1[np.newaxis, :]
        observation2 = observation2[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            feed_dict = {self.s1: observation1, self.s2: observation2}
            dp_dict = tl.utils.dict_to_one(self.eval_h10.all_drop)
            feed_dict.update(dp_dict)
            actions_value = self.sess.run(self.q_eval, feed_dict=feed_dict)
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        # print action
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            # print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        all_s1 = np.zeros((self.batch_size, 64, 64, 2))  # before
        all_s1_ = np.zeros((self.batch_size, 64, 64, 2))  # after
        all_s2 = np.zeros((self.batch_size, 15))
        all_s2_ = np.zeros((self.batch_size, 15))

        all_s_ = batch_memory[:, -self.n_features:]
        all_s = batch_memory[:, :self.n_features]
        for i in range(self.batch_size):
            all_s1_[i] = np.reshape(all_s_[i][:4096 * 2], (64, 64, 2))
            all_s1[i] = np.reshape(all_s[i][:4096 * 2], (64, 64, 2))
            all_s2_[i] = np.reshape(all_s_[i][4096 * 2:], (15,))
            all_s2[i] = np.reshape(all_s[i][4096 * 2:], (15,))

        feed_dict = {
            self.s1: all_s1,
            self.s2: all_s2,
            self.a: batch_memory[:, self.n_features],
            self.r: batch_memory[:, self.n_features + 1],
            self.s1_: all_s1_,
            self.s2_: all_s2_
        }

        feed_dict.update(self.eval_h10.all_drop)

        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict=feed_dict)

        self.cost_his.append(cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.savefig('result_new.png')

    def save_weights(self, filename):
        save_path = filename + '.ckpt'
        self.saver.save(self.sess, save_path)
        print('(Agent) save model to {}.'.format(save_path))

    def load_weights(self, filename):
        save_path = filename + '.ckpt'
        self.saver.restore(self.sess, save_path)
        print('(Agent) load model from {}.'.format(save_path))


if __name__ == '__main__':
    DQN = DeepQNetwork(3, 4, output_graph=True)
