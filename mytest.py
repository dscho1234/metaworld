
# from metaworld.core.image_env import ImageEnv
# from metaworld.envs.mujoco.cameras import sawyer_pick_and_place_camera
from metaworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place import SawyerPickAndPlaceEnv
from metaworld.envs.mujoco.ur3_xyz.ur3_pick_and_place import UR3PickAndPlaceEnv
import numpy as np
import time
import glfw
from metaworld.envs.mujoco.utils.rotation import euler2quat, quat2euler
from metaworld.envs.env_util import quat_to_zangle, zangle_to_quat, quat_create, quat_mul

"""
Sample a goal (object will be in hand as p_obj_in_hand=1) and try to set
the env state to the goal. I think there's a small chance this can fail
and the object falls out.
"""
#    env.set_to_goal(
#        {'state_desired_goal': env.generate_uncorrected_env_goals(1)['state_desired_goal'][0]}
#    )
# Close gripper for 20 timesteps

#    action = np.array([0, 0, 1])
def sample_pick_and_place():
    tasks = [{'goal': np.array([0.1, 0.4, 0.2]),  'obj_init_pos':np.array([0, 0.5, 0.02]), 'obj_init_angle': 0.3}] 
    hand_init_pos = (0, 0.6, 0.2)
    # hand_init_pos = (-0.314, 0.656, 0.066)
    env = UR3PickAndPlaceEnv(random_init=True, tasks = tasks, rotMode='rotz', hand_init_pos=hand_init_pos)
    # env = SawyerPickAndPlaceEnv(tasks = tasks, rotMode='rotz', hand_init_pos=hand_init_pos)
    # env.render()
    # print(env.get_endeff_pos())
    print(env.data.get_body_xquat('hand'))
    print(env.data.mocap_quat)
    for i in range(1):
        obs = env.reset()
    # env.render()
    print(env.data.get_body_xquat('hand'))
    print(env.data.mocap_quat)
    Ds = env.observation_space.shape[0]
    Da = env.action_space.shape[0]
    dt = env.dt
    
    print('observation space is ', Ds)
    print('action space is ', Da)
    print('qpos is ', env.data.qpos.shape)
    print('qvel is ', env.data.qvel.shape)
    print('mocap pos is ', env.data.mocap_pos)
    print('mocap quat is ', env.data.mocap_quat)
    
    print('dt is ', dt)
    for i in range(100):
        env.reset()
        
        # print(env.get_endeff_quat())
        for _ in range(50):
            # print(env.data.qpos[:7])
            # print('mocap quat is ', env.data.mocap_quat)
            # print('endeff pos : {}'.format(env.get_endeff_pos()))
            env.render()
            action = env.action_space.sample()
            # action[-1] = -1
            # action = np.zeros(Da)
            #*np.random.normal(size = 1)
            # action[0]=-1
            action[-1] = 1
            obs, _, _ , _ = env.step(action)
            # print(obs)
            # time.sleep(0.05)
        # glfw.destroy_window(env.viewer.window)

sample_pick_and_place()
