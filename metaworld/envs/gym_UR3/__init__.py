from gym.envs.registration import register

register(
    id='UR3-v0',
    entry_point='gym_UR3.envs.mujoco:MujocoUR3Env',
)