import gym
import pandas as pd
import numpy as np
import andes_gym
import os
import matplotlib.pyplot as plt
import time
import torch
from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3 import DDPG

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
plot_episode = True
# save_dir = "C:/Users/zguo19/andes_gym/results_zg/delay_learning_200_train3k_v5_Primary_15case/"
# save_dir = "C:/Users/zguo19/andes_gym/results_zg_v1/delay_learning_200_train3k_v3_Primary_15case/"
# save_dir = "C:/Users/zguo19/andes_gym/results_zg_v1/delay_learning_noAction_200_train3k_v1_Primary/"
# save_dir = "C:/Users/zguo19/andes_gym/results_zg_v2/delay_learning_0_train3k_v3_Primary_15case/"


for id in range(7):
    save_dir = "C:/Users/zguo19/andes_gym/results_zg_v2/delay_learning_{}_train3k_v3_Primary/".format(id*100)
    # env = gym.make('AndesPrimaryFreqControl-v0')  # use the reward function which only uses frequency 
    env = gym.make('AndesPrimaryFreqControl_rocof-v0') # use the reward function which uses frequency and ROCOF 
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.01 * np.ones(n_actions))
    train_freq = (1,"episode")
    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[128, 64])  # kwargs == keyword arguments
    #learning_starts:Number of steps for uniform-random action selection, before running real policy. Helps exploration
    model = DDPG(MlpPolicy, env, verbose=1, policy_kwargs=policy_kwargs, action_noise=action_noise, train_freq=train_freq, learning_starts=100*id)

    time_start = time.time()
    # This total_timesteps is not the total training episode but the numbers of actions applied
    # hence, we need to change the total steps with action numbers
    model.learn(total_timesteps=3000)  
    
    print("training {} completed using {}".format(id, time.time() - time_start))
    
    model.save(save_dir + "andes_primfreq_ddpg_fix_{}.pkl".format(id))
    freq = pd.DataFrame(env.final_freq)
    freq.to_csv(save_dir + "Finalfreq_ddpg_fix_{}.csv".format(id), index=False)
    freqRec = pd.DataFrame(env.best_episode_freq)
    freqRec.to_csv(save_dir + "Bestfreq_ddpg_sim_{}.csv".format(id), index=False)
    coord_record = pd.DataFrame(env.best_coord_record)
    coord_record.to_csv(save_dir + "BestCoord_ddpg_{}.csv".format(id), index=False)
    #save rocof
    rocof = pd.DataFrame(env.final_rocof)
    rocof.to_csv(save_dir + "Finalrocof_ddpg_fix_{}.csv".format(id), index=False)
    rocofBest = pd.DataFrame(env.best_episode_rocof)
    rocofBest.to_csv(save_dir + "bestrocof_ddpg_sim_{}.csv".format(id), index=False)    
        
    
    #  the training is completed, replay the trained model 

    # obs = env.reset()
    # done = False
    # while True:
    #     action, _states = model.predict(obs)
    #     obs, rewards, done, info = env.step(action)
    #     if done is True:
    #         break          

    # obs = env.reset()
    # done = False
    # while True:
    #     action, _states = model.predict(obs)
    #     action=np.array([0, 0,  0,  0, 0],dtype=float)
    #     obs, rewards, done, info = env.step(action)
    #     if done is True:
    #         break                
    
    Reward= pd.DataFrame(env.episode_reward)
    Reward.to_csv(save_dir + "reward_ddpg_fix_{}.csv".format(id), index=False)
    
    plt.rcParams.update({'font.family': 'Arial'})
    plt.figure(figsize=(9, 7))
    plt.plot(env.episode_reward, color='blue', alpha=1, linewidth=2)
    plt.xlabel("Episode", fontsize=20)
    plt.ylabel("Reward", fontsize=20)
    plt.grid()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title("Episode", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_dir + "reward_ddpg_{}.png".format(id))
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(left=0, right=np.max(env.t_render))
    ax.set_ylim(auto=True)
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    ax.set_xlabel("Time [s]", fontsize=16)
    ax.set_ylabel("Bus Frequency [Hz]", fontsize=16)
    ax.ticklabel_format(useOffset=False)
    # for i in range(env.N_Bus):
    #     ax.plot(env.t_render, env.final_obs_render[:, i] * 60)
    for i in range(env.N_Bus):
        ax.plot(env.t_render, env.best_episode_freq[:, i] * 60)
    plt.savefig(save_dir + "Bestfreq_ddpg_{}.png".format(id))
    max_best_freq=(np.max(env.best_episode_freq,axis=0))*60
    min_best_freq=(np.min(env.best_episode_freq,axis=0))*60
    max_final_freq=(np.max(env.final_obs_render,axis=0))*60
    min_final_freq=(np.min(env.final_obs_render,axis=0))*60

    #### plot ROCOF #####
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(left=0, right=np.max(env.t_render))
    ax.set_ylim(auto=True)
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    ax.set_xlabel("Time [s]", fontsize=16)
    ax.set_ylabel("Bus ROCOF [Hz/s]", fontsize=16)
    ax.ticklabel_format(useOffset=False)
    # for i in range(env.N_Bus):
    #     ax.plot(env.t_render, env.final_rocof_render[:, i]*60)
    for i in range(env.N_Bus):
        ax.plot(env.t_render, env.best_episode_rocof[:, i]*60)
    plt.savefig(save_dir + "Bestrocof_ddpg_{}.png".format(id))
    max_best_rocof=(np.max(env.best_episode_rocof,axis=0))*60
    min_best_rocof=(np.min(env.best_episode_rocof,axis=0))*60
    max_final_rocof=(np.max(env.final_rocof_render,axis=0))*60
    min_final_rocof=(np.min(env.final_rocof_render,axis=0))*60

    fre_rocof_eva=[]
    fre_rocof_eva.append(max_final_freq)
    fre_rocof_eva.append(min_final_freq)
    fre_rocof_eva.append(max_best_freq)
    fre_rocof_eva.append(min_best_freq)
    fre_rocof_eva.append(max_final_rocof)
    fre_rocof_eva.append(min_final_rocof)
    fre_rocof_eva.append(max_best_rocof)
    fre_rocof_eva.append(min_best_rocof)
    fre_rocof_eva_output = pd.DataFrame(fre_rocof_eva)
    # r1:max_final_freq,r2:min_final_freq,r3:max_best_freq,r8:....rn:min_best_rocof
    fre_rocof_eva_output.to_csv(save_dir + "Freq_rocof_eva_ddpg_fix_{}.csv".format(id), index=False)
    