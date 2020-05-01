
import numpy as np
import os
from gym import utils
from gym.envs.mujoco import mujoco_env
from scipy.spatial.transform import Rotation as R



class MujocoUR3Env(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, xml_name="ur3.xml"):        
        xml_path = os.path.join(os.path.dirname(__file__), "./assets", xml_name)
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, xml_path, 5)  # frameskip=1
        print('for check@@@@@@@@@@@@@@@@@@@')
    
    
    def step(self, action):  # information은 뭘로 주지..? # gripping에 대한건??
        self.do_simulation(action, self.frame_skip)
        rew = 1
        obs = self._get_obs()
        done = False
        return obs, rew, done, {}
        
    def _get_obs(self):
        obs = np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel() 

        return obs

    #TODO: env.reset에서 어처피 reset_model을 부름. 문제 생기면 이걸 고쳐야
    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()    

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 10

    @property
    def mass(self):
        return self.model.body_mass[1]

    @property
    def gravity(self):
        return self.model.opt.gravity
    
  
  