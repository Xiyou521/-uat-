import re
import os
from os.path import dirname, abspath, join
import numpy as np
import pandas as pd

from main import main

config = "qmix_transf"
model_path = "results_final_2/models/qmix_transf_5v5_levy"
results_path = 'results_final_2'

n_agents = 5
n_landmarks = 5
use_pf = True

eval_linear = False
eval_levy = False
eval_turning = True


addtional_config = [
    'env_args.only_closer_agent_pos=True',
    'env_args.obs_entity_mode=True',
    'env_args.use_custom_state=True'
]

def get_stats(df):
    
    land_error_columns = [c for c in df.columns if re.match('landmark [0-9]+_error',c)]
    land_lost_columns  = [c for c in df.columns if re.match('landmark [0-9]+_lost',c)]
    land_dist_columns  = [c for c in df.columns if re.match('landmark [0-9]+_min_dist',c)]
    
    # sum of lanadmrks lost at the end of the epsiode, divided by the number of landarks -> percentage of landmarks lost
    def lost_end_episode(row):
        """Retrieve the percentage of landmarks lost only if there were any collisions"""
        last_ts = row.iloc[np.argmax(row['t'])] # last time step
        if last_ts['agent_collision']==0 and last_ts['landmark_collision']==0:
            return (last_ts[land_lost_columns]).sum()/len(land_lost_columns)
        else:
            return None
        
    def distribution_end_episode(row):
        """Retrieve the percentage of landmarks lost only if there were any collisions"""
        last_ts = row.iloc[np.argmax(row['t'])] # last time step
        if last_ts['agent_collision']==0 and last_ts['landmark_collision']==0:
            return (last_ts[land_dist_columns] <= 0.3).sum()/len(land_dist_columns)
        else:
            return None

    d = {}
    # mean of the maximum time step per episode
    d['mean_episode_length'] = df.groupby('episode')['t'].max().mean()
    
    # mean of the sum of the episode rewards
    d['mean_reward'] = df.groupby('episode')['reward'].sum().mean()
    
    # mean of all the landmark errors
    d['mean_land_err'] = df[land_error_columns].mean().mean()
    
    # mean of all the landmark distances
    d['mean_land_dist'] = df[land_dist_columns].mean().mean()
    
    # mean of the landmarks lost at the end of the episode
    lost = df.groupby('episode').apply(lost_end_episode)
    d['mean_land_lost'] = lost.mean() if len(lost) and not np.isnan(lost.mean()) else 1. # avoid empty sequences, max bound for float error
    
    # mean of the correct distribution at the end of the episode
    distr = df.groupby('episode').apply(distribution_end_episode)
    d['mean_distr'] = distr.mean() if len(distr) and not np.isnan(distr.mean()) else 0.
    
    # percentage of episode that terminate with a collision
    d['perc_collision'] = df.groupby('episode')[['agent_collision','landmark_collision']].max().max(axis=1).sum()/df['episode'].unique().size

    d['std_episode_length'] = df.groupby('episode')['t'].max().std()
    d['std_reward'] = df.groupby('episode')['reward'].sum().std()
    d['std_land_err']  = df[land_error_columns].mean().std()
    d['std_land_dist'] = df[land_dist_columns].mean().std()
    d['std_land_lost'] = lost.std() if len(lost) and not np.isnan(lost.std()) else 0.
    d['std_distr'] = distr.std() if len(distr) and not np.isnan(distr.std()) else 1.
    
    return d


for difficulty in ['easy', 'medium','hard']:

    # basic parameters
    params = [
        'src/main.py',
        f'--config={config}',
        f"--env-config={'utracking_test_pf' if use_pf else 'utracking_test'}",
        'with',
        f"name={model_path.split('/')[-1]}",
        f'checkpoint_path={model_path}',
        f'env_args.difficulty={difficulty}',
        f'env_args.num_agents={n_agents}',
        f'env_args.num_landmarks={n_landmarks}',
    ] + addtional_config

    # parameters for linear movement
    if eval_linear:
        params_linear = params + [
            f'env_args.map_name={n_agents}_{n_landmarks}_linear_{difficulty}',
            f'env_args.random_vel=True',
            f'env_args.movement=linear'
        ]
        main(params_linear)


    # parameters for levy movement
    if eval_levy:
        params_levy = params + [
            f'env_args.map_name={n_agents}_{n_landmarks}_levy_{difficulty}',
            f'env_args.random_vel=True',
            f'env_args.landmark_vel=0.02',
            f'env_args.max_vel=0.05',
            f'env_args.movement=levy'
        ]
        main(params_levy)

    # parameters for turning movement
    if eval_turning:
        params_turning = params + [
            f'env_args.map_name={n_agents}_{n_landmarks}_turning_{difficulty}',
            f'env_args.random_vel=False',
            f'env_args.movement=turning'
        ]
        main(params_turning)  

results_path = join(dirname(dirname(abspath(__file__))), results_path) 
all_results = []
for dir_ in os.listdir(join(results_path, 'benchmarks')):
    if os.path.isdir(join(results_path, 'benchmarks', dir_)):
        for filename in os.listdir(join(results_path, 'benchmarks', dir_)):
            if filename[-3:] == 'csv':
                if filename.split('.')[0] == model_path.split('/')[-1]:
                    exp_name = dir_.split('_')
                    stats = {}
                    stats['map'] = exp_name[0]
                    stats['movement'] = exp_name[1]
                    stats['difficulty'] = exp_name[2]
                    
                    df = pd.read_csv(join(results_path, 'benchmarks', dir_, filename))
                    stats = {**stats, **get_stats(df)}
                    all_results.append(stats)

df_all = pd.DataFrame(all_results)
df_all.to_csv(join(results_path, 'benchmarks', f"{model_path.split('/')[-1]}.csv"), index=False)