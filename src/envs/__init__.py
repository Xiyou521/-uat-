from functools import partial
from utracking.epymarl_wrapper import UTrackingWrapper
from utracking.environment import MultiAgentEnv

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["utracking"] = partial(env_fn, env=UTrackingWrapper)