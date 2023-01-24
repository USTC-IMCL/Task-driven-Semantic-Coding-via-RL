from environment_class import coding_env
from DQN_class import DeepQNetwork
import h5py
import numpy as np
import argparse
import random
import time
import matplotlib
import tensorflow as tf
import os
matplotlib.use('Agg')
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='train', action='store_false', default=False)
parser.add_argument('--model', dest='model', action='store', default="./checkpoint_all/test")
parser.add_argument('--gpu',default='0')
args = parser.parse_args()


def run_env():
    step = 0
    cost = []
    epi_reward_all = []
    for episode in range(100000):
        # initial observation
        rand = random.random()
        observation = env.reset(is_train = 1)
        epi_step = 0
        epi_reward = 0
        while True:

            # RL choose action based on observation
            action = RL.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            epi_reward += reward
            epi_step += 1
            RL.store_transition(observation, action, reward, observation_)
            if (step > 32 * 32) and (step % 4) == 0:
                cost.append(RL.learn())

            # swap observation
            observation = observation_
            step += 1
            # break while loop when end of this episode
            if done:
                epi_reward_all.append(epi_reward)
                if episode % 10 == 0:
                    print("    epi: %f" % (episode))
                    print("    num: %f" % (epi_step))
                    print("    err: %s" % (info))
                    print("    epi_reward: %s\n" % (epi_reward))
                break


        if episode % 5000 == 0:
            RL.save_weights(args.model+'{}'.format(episode//5000))
def test(i):
    epi_reward = 0
    save_name = './ImageNet_64_'+str(i)+'.txt'
    fod = open(save_name, 'w')
    for episode in range(400):
        observation = env.reset(0, episode + 2000)
        step = 1
        while True:
            action = RL.choose_action(observation)
            fod.write('%d\n' % (action+22))
            # fod.write('%f\n' % (0.57*np.power(2, (action+10)/3)))
            print("    step: {}     action: {}     ratio: {}" .format(step, action+22, observation[-5]))
            step = step + 1
            observation, reward, done, info = env.step(action)
            epi_reward += reward
            if done:
                print("    epi: %f" % (episode))
                # print("    num: %f" % (epi_step))
                print("    err: %s" % (info))
                break
        print("    epi_reward: %s\n" % (epi_reward))


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.train:
        start = 0
        end = 1999
        env = coding_env(start, end)
        RL = DeepQNetwork(
            env.n_actions, env.n_features,
            learning_rate=0.00001,
            reward_decay=0.,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=50000,
            e_greedy_increment=0.0001
            # output_graph=True
        )
    # RL.load_weights(args.model)
        run_env()

    if not args.train:
        for i in range(15, 20):
            tf.reset_default_graph()
            start = 1
            end = 1
            env = coding_env(start, end)
            RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.0001,
                      reward_decay=0.99,
                      e_greedy=1.0,
                      replace_target_iter=300,
                      memory_size=500,
                      # output_graph=True
                      )
            RL.load_weights(args.model+str(i))
            test(i)


