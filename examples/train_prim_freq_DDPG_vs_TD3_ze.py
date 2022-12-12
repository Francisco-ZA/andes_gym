import gym
import pandas as pd
import numpy as np
import andes_gym
import os
import matplotlib.pyplot as plt
import time
import torch

## DDPG
from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3 import DDPG

## TD3
from stable_baselines3 import TD3

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
plot_episode = True
save_dir = "DDPG_learning_starts_200_action_100_Primary/"

for id in range(1):

    ## DDPG
    env1 = gym.make('AndesPrimaryFreqControl-v0')
    n_actions1 = env1.action_space.shape[-1]
    action_noise1 = NormalActionNoise(mean=np.zeros(n_actions1), sigma=0.01 * np.ones(n_actions1))
    train_freq1 = (1,"episode")
    policy_kwargs1 = dict(activation_fn=torch.nn.ReLU, net_arch=[128, 64])  # kwargs == keyword arguments
    model1 = DDPG(MlpPolicy, env1, verbose=1, policy_kwargs=policy_kwargs1, action_noise=action_noise1, train_freq=train_freq1, learning_starts=200)

    time_start = time.time()
    model1.learn(total_timesteps=3000)  # we need to change the total steps with action numbers
    
    print("training {} completed using {}".format(id, time.time() - time_start))
    
    model1.save(save_dir + "andes_primfreq_ddpg_fix_{}.pkl".format(id))
    # freq1 = pd.DataFrame(env1.final_freq)
    # freq1.to_csv(save_dir + "andes_primfreq_ddpg_fix_{}.csv".format(id), index=False)
    freqRec1 = pd.DataFrame(env1.best_episode_freq)
    freqRec1.to_csv(save_dir + "andes_primfreq_ddpg_sim_{}.csv".format(id), index=False)
    coord_record1 = pd.DataFrame(env1.best_coord_record)
    coord_record1.to_csv(save_dir + "andes_primfreq_ddpg_coord_{}.csv".format(id), index=False)
    time_to_store1 = pd.DataFrame(env1.t_render)
    time_to_store1.to_csv(save_dir + "andes_primfreq_ddpg_time_{}.csv".format(id), index=False)

    obs = env1.reset()
    done = False
    while True:
        action, _states = model1.predict(obs)
        obs, rewards, done, info = env1.step(action)
        if done is True:
            break
            
    ## TD3       
    env2 = gym.make('AndesPrimaryFreqControl-v0')
    n_actions2 = env2.action_space.shape[-1]
    action_noise2 = NormalActionNoise(mean=np.zeros(n_actions2), sigma=0.01 * np.ones(n_actions2))
    train_freq2 = (1,"episode")
    policy_kwargs2 = dict(activation_fn=torch.nn.ReLU, net_arch=[128, 64])  # kwargs == keyword arguments
    model2 = DDPG(MlpPolicy, env2, verbose=1, policy_kwargs=policy_kwargs2, action_noise=action_noise2, train_freq=train_freq2, learning_starts=200)

    time_start = time.time()
    model2.learn(total_timesteps=3000)  # we need to change the total steps with action numbers
    
    print("training {} completed using {}".format(id, time.time() - time_start))
    
    model2.save(save_dir + "andes_primfreq_td3_fix_{}.pkl".format(id))
    # freq2 = pd.DataFrame(env2.final_freq)
    # freq2.to_csv(save_dir + "andes_primfreq_td3_fix_{}.csv".format(id), index=False)
    freqRec2 = pd.DataFrame(env2.best_episode_freq)
    freqRec2.to_csv(save_dir + "andes_primfreq_td3_sim_{}.csv".format(id), index=False)
    coord_record2 = pd.DataFrame(env2.best_coord_record)
    coord_record2.to_csv(save_dir + "andes_primfreq_td3_coord_{}.csv".format(id), index=False)
    time_to_store2 = pd.DataFrame(env2.t_render)
    time_to_store2.to_csv(save_dir + "andes_primfreq_td3_time_{}.csv".format(id), index=False)

    obs = env2.reset()
    done = False
    while True:
        action, _states = model2.predict(obs)
        obs, rewards, done, info = env2.step(action)
        if done is True:
            break
    
    
    plt.rcParams.update({'font.family': 'Arial'})
    plt.figure(figsize=(9, 7))
    plt.plot(env1.episode_reward, color='blue', alpha=1, linewidth=2, label = 'DDPG')
    plt.plot(env2.episode_reward, color ='red', alpha=1, linewidth=2, label = 'TD3')
    plt.legend(loc="lower right")
    # plt.show()
    plt.xlabel("Episode", fontsize=20)
    plt.ylabel("Reward", fontsize=20)
    plt.grid()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title("Episode", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_dir + "andes_primfreq_ddpg_vs_td3_fix_{}.png".format(id))
    fig1 = plt.figure(figsize=(9, 6))
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.set_xlim(left=0, right=np.max(env1.t_render))
    ax1.set_ylim(auto=True)
    ax1.xaxis.set_tick_params(labelsize=16)
    ax1.yaxis.set_tick_params(labelsize=16)
    ax1.set_xlabel("Time [s]", fontsize=16)
    ax1.set_ylabel("Bus Frequency [Hz]", fontsize=16)
    ax1.ticklabel_format(useOffset=False)
    
    for i in range(env1.N_Bus):
        ax1.plot(env1.t_render, env1.final_obs_render[:, i] * 60)
    # for i in range(env2.N_Bus):    
        ax1.plot(env2.t_render, env2.final_obs_render[:, i] * 60)
    # plt.show()
    plt.savefig(save_dir + "fig_finalobs_ddpg_vs_td3.png")

    
    fig2 = plt.figure(figsize=(9, 6))
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.set_xlim(left=0, right=np.max(env1.t_render))
    ax2.set_ylim(auto=True)
    ax2.xaxis.set_tick_params(labelsize=16)
    ax2.yaxis.set_tick_params(labelsize=16)
    ax2.set_xlabel("Time [s]", fontsize=16)
    ax2.set_ylabel("Bus Frequency [Hz]", fontsize=16)
    ax2.ticklabel_format(useOffset=False)

    TT = env1.N_Bus

    for i in range(env1.N_Bus):
        if i ==  TT-1:
            ax2.plot(env1.t_render, env1.best_episode_freq[:, i] * 60, label = 'DDPG')
            ax2.plot(env2.t_render, env2.best_episode_freq[:, i] * 60, label = 'TD3') 
        else:
            ax2.plot(env1.t_render, env1.best_episode_freq[:, i] * 60)
            ax2.plot(env2.t_render, env2.best_episode_freq[:, i] * 60) 


    # plt.show()  
    plt.savefig(save_dir + "fig_bestepi_ddpg_vs_td3.png")
    # print("training {} completed using {}".format(id, time.time() - time_start))