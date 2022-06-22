from functools import partial
import numbers
import pretrained
from utracking.environment import MultiAgentEnv
import utracking.scenarios as scenarios
from gym import ObservationWrapper, spaces
from gym.spaces import flatdim
import numpy as np
from gym.wrappers import TimeLimit as GymTimeLimit
import time

class TimeLimit(GymTimeLimit):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        # if self.env.spec is not None:
        #     self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert (self._elapsed_steps is not None), "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not done
            done = len(observation) * [True]
        return observation, reward, done, info


class FlattenObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation of individual agents."""

    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)

        ma_spaces = []

        for sa_obs in env.observation_space:
            flatdim = spaces.flatdim(sa_obs)
            ma_spaces += [
                spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(flatdim,),
                    dtype=np.float32,
                )
            ]

        self.observation_space = spaces.Tuple(tuple(ma_spaces))

    def observation(self, observation):
        return tuple(
            [
                spaces.flatten(obs_space, obs)
                for obs_space, obs in zip(self.env.observation_space, observation)
            ]
        )


class UTrackingWrapper(MultiAgentEnv):
    def __init__(self, time_limit, pretrained_wrapper, **kwargs):
        self.episode_limit = time_limit

        # generate world
        scenario = scenarios.load(kwargs["scenario_name"] + ".py").Scenario()
        world = scenario.make_world(num_agents=kwargs["num_agents"],
                                    num_landmarks=kwargs["num_landmarks"],
                                    landmark_depth=kwargs["landmark_depth"],
                                    landmark_movable=kwargs["landmark_movable"],
                                    agent_vel=kwargs["agent_vel"],
                                    landmark_vel=kwargs["landmark_vel"],
                                    max_vel=kwargs["max_vel"],
                                    random_vel=kwargs["random_vel"],
                                    movement=kwargs["movement"],
                                    pf_method=kwargs["pf_method"],
                                    rew_err_th=kwargs["rew_err_th"],
                                    rew_dis_th=kwargs["rew_dis_th"],
                                    max_range=kwargs["max_range"],
                                    sense_range=kwargs["sense_range"],
                                    only_closer_agent_pos=kwargs["only_closer_agent_pos"],
                                    min_init_dist=kwargs["min_init_dist"],
                                    max_init_dist=kwargs["max_init_dist"],
                                    difficulty=kwargs["difficulty"],
                                    obs_entity_mode=kwargs["obs_entity_mode"])

        # create env from world and scenario
        env = MultiAgentEnv(world, 
                            scenario.reset_world,
                            scenario.reward,
                            scenario.observation,
                            observation_full_callback=scenario.state if kwargs["use_custom_state"] else None,
                            info_callback=scenario.benchmark_data if kwargs["benchmark"] else None, 
                            done_callback = scenario.done)

        self._env = TimeLimit(env, max_episode_steps=time_limit)
        self._env = FlattenObservation(self._env)

        if pretrained_wrapper:
            self._env = getattr(pretrained, pretrained_wrapper)(self._env)

        # general env parameters
        self.n_agents = self._env.world.num_agents
        self.n_landmarks = self._env.world.num_landmarks
        self.benchmark = kwargs["benchmark"]

        self._obs = None
        self._state = None

        self.longest_action_space = max(self._env.action_space, key=lambda x: x.n)
        self.longest_observation_space = max(self._env.observation_space, key=lambda x: x.shape)
        self.custom_state = kwargs["use_custom_state"]

        self._seed = kwargs["seed"]
        self._env.seed(self._seed)

    def step(self, actions, xy_mode=False):
        """ Returns reward, terminated, info """
        actions = [int(a) for a in actions]
        self._obs, reward, done, self.info = self._env.step(actions)

        if self.custom_state:
            self._state = self._env.get_state() 

        return reward, done, self.get_info_stats()

    def get_agent_positions(self):
        """Returns the [x,y] positions of each agent in a list"""
        return [a.state.p_pos for a in self._env.world.agents]
    
    def get_landmark_predictions(self):
        """Returns the landmark [x,y] best predictions"""
        return [self.info[f'{landmark.name}_pred'] for landmark in self._env.world.landmarks]

    def get_landmark_positions(self):
        """Returns the [x,y] landmark positions for each landmark in a list"""
        return [self.info[f'{landmark.name}_pos'] for landmark in self._env.world.landmarks]

    def get_prediction_error(self):
        """Returns the prediction error for each landmark in a list"""
        return [self.info[f'{landmark.name}_error'] for landmark in self._env.world.landmarks]

    def get_info_stats(self):
        """Return only the variables of info that should be logged"""
        return {k:v for k, v in self.info.items() if isinstance(v, numbers.Number)}

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self._obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise self._obs[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return flatdim(self.longest_observation_space)

    def get_state(self):
        if self.custom_state:
            return self._state
        else: 
            return np.concatenate(self._obs, axis=0).astype(np.float32)
            
    def get_state_size(self):
        """ Returns the shape of the state"""
        if self.custom_state:
            return self._env.state_space
        else:
            return self.n_agents * flatdim(self.longest_observation_space)

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        valid = flatdim(self._env.action_space[agent_id]) * [1]
        invalid = [0] * (self.longest_action_space.n - len(valid))
        return valid + invalid

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return flatdim(self.longest_action_space)

    def reset(self):
        """ Returns initial observations and states"""
        self._obs = self._env.reset()
        if self.custom_state:
            self._state = self._env.get_state() 
        self._obs = [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in self._obs
        ]
        return self.get_obs(), self.get_state()

    def render(self):
        return self._env.render('rgb_array')[0]

    def close(self):
        self._env.close()

    def seed(self):
        return self._env.seed

    def save_replay(self):
        pass

    def get_stats(self):
        return {}

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info
