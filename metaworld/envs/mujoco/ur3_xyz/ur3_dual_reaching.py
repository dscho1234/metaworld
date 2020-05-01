from collections import OrderedDict
import numpy as np
from gym.spaces import  Dict , Box


from metaworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from metaworld.core.multitask_env import MultitaskEnv
from metaworld.envs.mujoco.ur3_xyz.ur3_dual_base import UR3XYZEnv


from metaworld.envs.mujoco.utils.rotation import euler2quat

from metaworld.envs.env_util import quat_to_zangle, zangle_to_quat, quat_create, quat_mul

class UR3Reaching(UR3XYZEnv):
    def __init__(
            self,
            random_init=False,
            goal_low=None,
            goal_high=None,
            hand_init_pos = (0.2, 0.4, 0.2),
            reachThresh = 0.01,
            rotMode='horizontal_fixed',#'vertical_fixed',
            **kwargs
    ):
        raise NotImplementedError('코드 짜다 말았음!')
        self.quick_init(locals())
               
        #for right arm
        hand_low=(-0.0, 0.25, 0.05)
        hand_high=(0.3, 0.55, 0.3)
        target_low=(-0.0, 0.25, 0.04)
        target_high=(0.3, 0.55, 0.06)
        alpha_low = (0.0)
        alpha_high = (1.0)
        UR3XYZEnv.__init__(
            self,
            frame_skip=5,
            action_scale=1./200,
            hand_low=hand_low,
            hand_high=hand_high,
            model_name=self.model_name,
            **kwargs
        )
        self.targets = [np.array(first point), np.array(second point), ...]
        self.num_targets = len(self.targets)
        self.current_target = self.targets[0]
        self.random_init = random_init
        self.max_path_length = 200#150
        self.reachThresh = reachThresh
        self.rotMode = rotMode
        self.hand_init_pos = np.array(hand_init_pos)
        if rotMode == 'horizontal_fixed' or rotMode=='vertical_fixed':
            self.action_space = Box(
                np.array([-1, -1, -1, -1]),
                np.array([1, 1, 1, 1]),
            )
        elif rotMode == 'rotz':
            raise NotImplementedError
            self.action_rot_scale = 1./50
            self.action_space = Box(
                np.array([-1, -1, -1, -np.pi, -1]),
                np.array([1, 1, 1, np.pi, 1]),
            )
        elif rotMode == 'quat':
            raise NotImplementedError
            self.action_rot_scale = 1./10 #dscho mod
            self.action_space = Box(
                np.array([-1, -1, -1, 0, -1, -1, -1, -1]),
                np.array([1, 1, 1, 2*np.pi, 1, 1, 1, 1]),
            )
        else:
            raise NotImplementedError
            self.action_rot_scale = 1./10 #dscho mod
            self.action_space = Box(
                np.array([-1, -1, -1, -np.pi/2, -np.pi/2, 0, -1]),
                np.array([1, 1, 1, np.pi/2, np.pi/2, np.pi*2, 1]),
            )
        self.hand_and_space = Box(hand_low, hand_high)
        self.target_space = Box(target_low, target_high)
        self.observation_space = Box(
                np.hstack((hand_low, target_low, alpha_low)),
                np.hstack((hand_high, target_high, alpha_high)),
        )
       

    @property
    def model_name(self):     
        return get_asset_full_path('ur3_xyz/ur3_dual_pick_and_place.xml')


    def get_goal(self):
        return {
            'state_desired_goal': self._state_goal,
    }
    def viewer_setup(self):
        # top view
        # self.viewer.cam.trackbodyid = 0
        # self.viewer.cam.lookat[0] = 0
        # self.viewer.cam.lookat[1] = 1.0
        # self.viewer.cam.lookat[2] = 0.5
        # self.viewer.cam.distance = 0.6
        # self.viewer.cam.elevation = -45
        # self.viewer.cam.azimuth = 270
        # self.viewer.cam.trackbodyid = -1
        # side view
        # self.viewer.cam.trackbodyid = 0
        # self.viewer.cam.lookat[0] = 0.2
        # self.viewer.cam.lookat[1] = 0.75
        # self.viewer.cam.lookat[2] = 0.4
        # self.viewer.cam.distance = 0.4
        # self.viewer.cam.elevation = -55
        # self.viewer.cam.azimuth = 180
        # self.viewer.cam.trackbodyid = -1
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 0.7

    def step(self, action):
        # self.set_xyz_action_rot(action[:7])
        if self.rotMode == 'euler': #ee pos xyz control + xyz rotation by euler
            action_ = np.zeros(7)
            action_[:3] = action[:3]
            action_[3:] = euler2quat(action[3:6])
            self.set_xyz_action_rot(action_)
        elif self.rotMode == 'horizontal_fixed' or self.rotMode == 'vertical_fixed':
            self.set_xyz_action(action[:3]) #ee pos xyz control
        elif self.rotMode == 'rotz':
            self.set_xyz_action_rotz(action[:4]) #ee pos xyz control + z rotation
        else:
            self.set_xyz_action_rot(action[:7]) #ee pos xyz control + xyz rotation by quat? 불확실
        # self.do_simulation([action[-1], -action[-1]]) #gripper 여닫는거인듯
        self.do_simulation([action[-1], action[-1]]) #gripper 여닫는거인듯
        
        # The marker seems to get reset every time you do a simulation
        self._set_goal_marker(self._state_goal)
        ob = self._get_obs()
        obs_dict = self._get_obs_dict()
        reward = self.compute_reward(action, obs_dict, mode = self.rewMode)
        self.curr_path_length +=1
        #info = self._get_info()
        if self.curr_path_length == self.max_path_length:
            done = True
        else:
            done = False
        return ob, reward, done, { 'reachRew':reachRew, 'reachDist': reachDist, 'pickRew':pickRew, 'placeRew': placeRew, 'epRew' : reward, 'placingDist': placingDist}
   
    def _get_obs(self):
        hand = self.get_endeff_pos()
        objPos =  self.data.get_geom_xpos('objGeom')
        flat_obs = np.concatenate((hand, objPos))
        return np.concatenate([
                flat_obs,
                self._state_goal
            ])

    def _get_obs_dict(self):
        hand = self.get_endeff_pos()
        objPos =  self.data.get_geom_xpos('objGeom')
        flat_obs = np.concatenate((hand, objPos))
        return dict(
            state_observation=flat_obs,
            state_desired_goal=self._state_goal,
            state_achieved_goal=objPos,
        )

    def _get_info(self):
        pass
    
    def _set_goal_marker(self, goal):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
        self.data.site_xpos[self.model.site_name2id('goal')] = (
            goal[:3]
        )
    
    #dscho modified
    def set_joint_state(self, joint_angles):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[0:6] = joint_angles.copy()
        qvel[14:20] = 0 
        self.set_state(qpos, qvel)


    def _set_obj_xyz(self, pos):
        #dscho modified 
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[14:17] = pos.copy() #0 ~ 5 ur3, 6~9 grip_r, 10~13 grip_l, 14~16 object pos, 17~20 object quat
        qvel[14:20] = 0 #0~5 ur3, 6~9 grip_r, 10~13 grip_l, 14~19 object vel, angvel
        self.set_state(qpos, qvel)


    def sample_goals(self, batch_size):
        #Required by HER-TD3
        goals = []
        for i in range(batch_size):
            task = self.tasks[np.random.randint(0, self.num_tasks)]
            goals.append(task['goal'])
        return {
            'state_desired_goal': goals,
        }



    def reset_model(self):
        self._reset_hand()
        task = self.sample_task()
        self._state_goal = np.array(task['goal'])
        self.obj_init_pos = self.adjust_initObjPos(task['obj_init_pos'])
        self.obj_init_angle = task['obj_init_angle']
        self.objHeight = self.data.get_geom_xpos('objGeom')[2]
        self.heightTarget = self.objHeight + self.liftThresh
        if self.random_init:
            goal_pos = np.random.uniform(
                self.hand_and_obj_space.low,
                self.hand_and_obj_space.high,
                size=(self.hand_and_obj_space.low.size),
            )
            while np.linalg.norm(goal_pos[:3] - goal_pos[-3:]) < 0.1:
                goal_pos = np.random.uniform(
                    self.hand_and_obj_space.low,
                    self.hand_and_obj_space.high,
                    size=(self.hand_and_obj_space.low.size),
                )
            self._state_goal = goal_pos[:3]
            self.obj_init_pos = np.concatenate((goal_pos[-3:-1], np.array([self.obj_init_pos[-1]])))
        self._set_goal_marker(self._state_goal)
        self._set_obj_xyz(self.obj_init_pos)
        #self._set_obj_xyz_quat(self.obj_init_pos, self.obj_init_angle)
        self.curr_path_length = 0
        self.maxPlacingDist = np.linalg.norm(np.array([self.obj_init_pos[0], self.obj_init_pos[1], self.heightTarget]) - np.array(self._state_goal)) + self.heightTarget
        #Can try changing this
        return self._get_obs()

    def _reset_hand(self):
        #10번씩 하는건 gripper 닫는 시간때문
        for _ in range(10): 
            self.data.set_mocap_pos('mocap', self.hand_init_pos)
            # self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0])) #w v 순인듯
            # self.data.set_mocap_quat('mocap', np.array([1, 0, 0, 0])) #w v 순인듯
            if self.rotMode=='vertical_fixed':
                quat = quat_mul(quat_create(np.array([1., 0, 0]), np.pi) ,quat_create(np.array([0, 0, 1.]), np.pi/2)) #ref 기준 x축 180, z축 90순
            elif self.rotMode=='horizontal_fixed':
                quat = quat_mul(quat_create(np.array([0, 0, 1.]), np.pi) ,quat_create(np.array([0, 1., 0]), np.pi/2)) #ref 기준 z축 180, y축 90순
            else: #그 외 경우도 vertically initialize
                quat = quat_mul(quat_create(np.array([1., 0, 0]), np.pi) ,quat_create(np.array([0, 0, 1.]), np.pi/2)) #ref 기준 x축 180, z축 90순
            
            self.data.set_mocap_quat('mocap', quat)
            self.do_simulation([0,0], self.frame_skip)
            
        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.pickCompleted = False

    def get_site_pos(self, siteName):
        _id = self.model.site_names.index(siteName)
        return self.data.site_xpos[_id].copy()

    def compute_reward(self, actions, obs, mode = 'general'):
        if isinstance(obs, dict):
            obs = obs['state_observation']

        objPos = obs[3:6]

        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        fingerCOM  =  (rightFinger + leftFinger)/2

        heightTarget = self.heightTarget
        placingGoal = self._state_goal

        reward =
        return reward

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        return statistics

    def log_diagnostics(self, paths = None, logger = None):
        pass
