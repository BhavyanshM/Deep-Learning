from gym.envs.registration import register

register(
    id='avoidance-v0',
    entry_point='gym_avoidance.envs:AvoidanceEnv',
)
register(
    id='avoidance-extrahard-v0',
    entry_point='gym_avoidance.envs:AvoidanceExtraHardEnv',
)
