from collections import OrderedDict
import numpy as np
from gym.spaces import  Dict , Box


from metaworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from metaworld.core.multitask_env import MultitaskEnv
from metaworld.envs.mujoco.ur3_xyz.ur3_dual_base import UR3DualXYZEnv


from metaworld.envs.mujoco.utils.rotation import euler2quat

from metaworld.envs.env_util import quat_to_zangle, zangle_to_quat, quat_create, quat_mul

class UR3DualPickAndPlaceEnv(UR3DualXYZEnv):
    def __init__(
            self,
            obj_low=None,
            obj_high=None,
            random_init=False,
            tasks = [{'goal': np.array([0.1, 0.8, 0.2]),  'obj_init_pos':np.array([0, 0.6, 0.02]), 'obj_init_angle': 0.3}], 
            goal_low=None,
            goal_high=None,
            hand_init_pos = (0.1, 0.5, 0.2),
            second_hand_init_pos = (-0.1, 0.5, 0.2),
            liftThresh = 0.04,
            rewMode = 'orig',
            rotMode='horizontal_fixed',#'vertical_fixed',
            **kwargs
    ):
        

        #for right arm
        hand_low=(0.0, 0.25, 0.05)
        hand_high=(0.3, 0.55, 0.3)
        second_hand_low=(-0.3, 0.55, 0.3)
        second_hand_high=(0.0, 0.25, 0.05)
        obj_low=(-0.0, 0.25, 0.04)
        obj_high=(0.3, 0.55, 0.06)
        UR3DualXYZEnv.__init__(
            self,
            frame_skip=5,
            action_scale=1./100,
            hand_low=hand_low,
            hand_high=hand_high,
            model_name=self.model_name,
            **kwargs
        )
        if obj_low is None:
            obj_low = self.hand_low

        if goal_low is None:
            goal_low = self.hand_low

        if obj_high is None:
            obj_high = self.hand_high
        
        if goal_high is None:
            goal_high = self.hand_high

        self.random_init = random_init
        self.liftThresh = liftThresh
        self.max_path_length = 200#150
        self.tasks = tasks
        self.num_tasks = len(tasks)
        self.rewMode = rewMode
        self.rotMode = rotMode
        self.hand_init_pos = np.array(hand_init_pos)
        self.second_hand_init_pos = np.array(second_hand_init_pos)
        if rotMode == 'horizontal_fixed' or rotMode=='vertical_fixed':
            self.action_space = Box(
                np.array([-1, -1, -1, -1, -1, -1, -1, -1]),
                np.array([1, 1, 1, 1, 1, 1, 1, 1]),
            )
        elif rotMode == 'rotz':
            
            self.action_rot_scale = 1./50
            self.action_space = Box(
                np.array([-1, -1, -1, -np.pi, -1,    -1, -1, -1, -np.pi, -1]),
                np.array([1, 1, 1, np.pi, 1,    1, 1, 1, np.pi, 1]),
            )
        elif rotMode == 'quat':
            
            self.action_rot_scale = 1./10 #dscho mod
            self.action_space = Box(
                np.array([-1, -1, -1, 0, -1, -1, -1, -1,     -1, -1, -1, 0, -1, -1, -1, -1]),
                np.array([1, 1, 1, 2*np.pi, 1, 1, 1, 1,    1, 1, 1, 2*np.pi, 1, 1, 1, 1]),
            )
        else:
            
            self.action_rot_scale = 1./10 #dscho mod
            self.action_space = Box(
                np.array([-1, -1, -1, -np.pi/2, -np.pi/2, 0, -1,    -1, -1, -1, -np.pi/2, -np.pi/2, 0, -1]),
                np.array([1, 1, 1, np.pi/2, np.pi/2, np.pi*2, 1,    1, 1, 1, np.pi/2, np.pi/2, np.pi*2, 1]),
            )
        self.hand_and_obj_space = Box(
            np.hstack((self.hand_low, self.second_hand_low, obj_low)),
            np.hstack((self.hand_high, self.second_hand_high, obj_high)),
        )
        self.goal_space = Box(goal_low, goal_high)
        self.observation_space = Box(
                np.hstack((self.hand_low, self.second_hand_low, obj_low, obj_low)),
                np.hstack((self.hand_high, self.second_hand_high, obj_high, obj_high)),
        )
        # self.observation_space = Dict([
        #     ('state_observation', self.hand_and_obj_space),
        #     ('state_desired_goal', self.goal_space),
        #     ('state_achieved_goal', self.goal_space),
        # ])


    def get_goal(self):
        return {
            'state_desired_goal': self._state_goal,
    }

    @property
    def model_name(self):     
        return get_asset_full_path('ur3_xyz/ur3_dual_pick_and_place.xml')

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
        v.cam.distance = self.model.stat.extent * 1.7

    def step(self, action):
        # self.set_xyz_action_rot(action[:7])
        if self.rotMode == 'euler': #ee pos xyz control + xyz rotation by euler
            # action_ = np.zeros(7)
            # action_[:3] = action[:3]
            # action_[3:] = euler2quat(action[3:6])
            # self.set_xyz_action_rot(action_)
            #action 6+1 6+1
            action_ = np.zeros(14)
            action_[:3] = action[:3]
            action_[3:7] = euler2quat(action[3:6])
            action_[7:10] = action[7:10]
            action_[10:14] = euler2quat(action[10:13])
            self.set_xyz_action_rot(action_)
            self.do_simulation([action[7], action[7], action[-1], action[-1]])
            
        elif self.rotMode == 'horizontal_fixed' or self.rotMode == 'vertical_fixed':
            # self.set_xyz_action(action[:3]) #ee pos xyz control
            #action 3+1 3+1
            action_ = np.zeros(6)
            action_[:3] = action[:3]
            action_[3:6] = action[4:7]
            self.set_xyz_action(action_) #ee pos xyz control
            self.do_simulation([action[3], action[3], action[-1], action[-1]])
        elif self.rotMode == 'rotz':
            # self.set_xyz_action_rotz(action[:4]) #ee pos xyz control + z rotation
            #action 4+1 4+1
            action_ = np.zeros(8)
            action_[:4] = action[:4]
            action_[4:8] = action[5:9]
            self.set_xyz_action_rotz(action_) #ee pos xyz control + z rotation
            self.do_simulation([action[4], action[4], action[-1], action[-1]])
        else:
            # self.set_xyz_action_rot(action[:7]) #ee pos xyz control + xyz rotation by quat? 불확실
            #action 7+1 7+1
            action_ = np.zeros(14)
            action_[:7] = action[:7]
            action_[7:14] = action[8:15]
            self.set_xyz_action_rot(action_)
            self.do_simulation([action[8], action[8], action[-1], action[-1]])

        
        
        # The marker seems to get reset every time you do a simulation
        self._set_goal_marker(self._state_goal)
        ob = self._get_obs()
        obs_dict = self._get_obs_dict()
        
        reward , reachRew, reachDist, pickRew, placeRew , placingDist = self.compute_reward(action, obs_dict, mode = self.rewMode)
        self.curr_path_length +=1
        #info = self._get_info()
        if self.curr_path_length == self.max_path_length:
            done = True
        else:
            done = False
        return ob, reward, done, { 'reachRew':reachRew, 'reachDist': reachDist, 'pickRew':pickRew, 'placeRew': placeRew, 'epRew' : reward, 'placingDist': placingDist}
   
    def _get_obs(self):
        hand = self.get_endeff_pos()
        second_hand = self.get_second_endeff_pos()
        objPos =  self.data.get_geom_xpos('objGeom')
        flat_obs = np.concatenate((hand, second_hand, objPos))
        return np.concatenate([
                flat_obs,
                self._state_goal
            ])

    def _get_obs_dict(self):
        hand = self.get_endeff_pos()
        second_hand = self.get_second_endeff_pos()
        objPos =  self.data.get_geom_xpos('objGeom')
        flat_obs = np.concatenate((hand, second_hand, objPos))
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

    def _set_objCOM_marker(self):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
        objPos =  self.data.get_geom_xpos('objGeom')
        self.data.site_xpos[self.model.site_name2id('objSite')] = (
            objPos
        )
    
    #dscho modified
    def set_joint_state(self, joint_angles):
        #qpos : 0~5 ur3, 6~9 grip_r, 10~13 grip_l, 14~19 second_ur3, 20~23 second_grip_r, 24~27 second_grip_l,  28~30 object pos, 31~34 object quat
        #qvel : 0~5 ur3, 6~9 grip_r, 10~13 grip_l, 14~19 second_ur3, 20~23 second_grip_r, 24~27 second_grip_l, 28~33 object vel, angvel
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        angles = joint_angles.copy()
        qpos[0:6] = angles[0:6] #first ur3
        qpos[14:20] = angles[6:12] #second ur3
        qvel[28:34] = 0 #object vel

        self.set_state(qpos, qvel)
    #dscho modified
    def get_joint_state(self):
        #qpos : 0~5 ur3, 6~9 grip_r, 10~13 grip_l, 14~19 second_ur3, 20~23 second_grip_r, 24~27 second_grip_l,  28~30 object pos, 31~34 object quat
        #qvel : 0~5 ur3, 6~9 grip_r, 10~13 grip_l, 14~19 second_ur3, 20~23 second_grip_r, 24~27 second_grip_l, 28~33 object vel, angvel
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        return np.concatenate([qpos[0:6], qpos[14:20]]) 

    def _set_obj_xyz(self, pos):
        #dscho modified 
        #qpos : 0~5 ur3, 6~9 grip_r, 10~13 grip_l, 14~19 second_ur3, 20~23 second_grip_r, 24~27 second_grip_l,  28~30 object pos, 31~34 object quat
        #qvel : 0~5 ur3, 6~9 grip_r, 10~13 grip_l, 14~19 second_ur3, 20~23 second_grip_r, 24~27 second_grip_l, 28~33 object vel, angvel
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[28:31] = pos.copy()
        qvel[28:34] = 0
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


    def sample_task(self):
        task_idx = np.random.randint(0, self.num_tasks)
        return self.tasks[task_idx]

    def adjust_initObjPos(self, orig_init_pos):
        #This is to account for meshes for the geom and object are not aligned
        #If this is not done, the object could be initialized in an extreme position
        diff = self.get_body_com('obj')[:2] - self.data.get_geom_xpos('objGeom')[:2]
        adjustedPos = orig_init_pos[:2] + diff

        #The convention we follow is that body_com[2] is always 0, and geom_pos[2] is the object height
        return [adjustedPos[0], adjustedPos[1],self.data.get_geom_xpos('objGeom')[-1]]


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

    def _dscho_reset_hand(self):
        #10번씩 하는건 gripper 닫는 시간때문
        joint_angles = self.init_joint_angles
        self.set_joint_state(joint_angles=joint_angles)
        for _ in range(10): 
            self.data.set_mocap_pos('mocap', self.hand_init_pos)
            self.data.set_mocap_pos('second_mocap', self.second_hand_init_pos)
            # self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0])) #w v 순인듯
            # self.data.set_mocap_quat('mocap', np.array([1, 0, 0, 0])) #w v 순인듯
            if self.rotMode=='vertical_fixed':
                quat = quat_mul(quat_create(np.array([1., 0, 0]), np.pi) ,quat_create(np.array([0, 0, 1.]), np.pi/2)) #ref 기준 x축 180, z축 90순
                second_quat = quat_mul(quat_create(np.array([0, 1., 0]), np.pi) ,quat_create(np.array([0, 0, 1.]), np.pi/2)) #ref 기준 y축 180, z축 90순
            elif self.rotMode=='horizontal_fixed':
                quat = quat_mul(quat_create(np.array([0, 0, 1.]), np.pi) ,quat_create(np.array([0, 1., 0]), np.pi/2)) #ref 기준 z축 180, y축 90순
                second_quat = quat_create(np.array([0, 1., 0]), np.pi/2) #ref 기준 y축 90
            else: #그 외 경우도 vertically initialize
                quat = quat_mul(quat_create(np.array([1., 0, 0]), np.pi) ,quat_create(np.array([0, 0, 1.]), np.pi/2)) #ref 기준 x축 180, z축 90순
                second_quat = quat_mul(quat_create(np.array([0, 1., 0]), np.pi) ,quat_create(np.array([0, 0, 1.]), np.pi/2)) #ref 기준 y축 180, z축 90순
            self.data.set_mocap_quat('mocap', quat)
            self.data.set_mocap_quat('second_mocap', second_quat)
            # self.do_simulation([-1,1], self.frame_skip)
            # self.do_simulation([0,0], self.frame_skip)
            self.do_simulation([0,0,0,0], self.frame_skip)
            
        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        second_rightFinger, second_leftFinger = self.get_site_pos('second_rightEndEffector'), self.get_site_pos('second_leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.second_init_fingerCOM  =  (second_rightFinger + second_leftFinger)/2
        self.pickCompleted = False

    def _reset_hand(self):
        #10번씩 하는건 gripper 닫는 시간때문
        for _ in range(10): 
            self.data.set_mocap_pos('mocap', self.hand_init_pos)
            self.data.set_mocap_pos('second_mocap', self.second_hand_init_pos)
            # self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0])) #w v 순인듯
            # self.data.set_mocap_quat('mocap', np.array([1, 0, 0, 0])) #w v 순인듯
            if self.rotMode=='vertical_fixed':
                quat = quat_mul(quat_create(np.array([1., 0, 0]), np.pi) ,quat_create(np.array([0, 0, 1.]), np.pi/2)) #ref 기준 x축 180, z축 90순
                second_quat = quat_mul(quat_create(np.array([0, 1., 0]), np.pi) ,quat_create(np.array([0, 0, 1.]), np.pi/2)) #ref 기준 y축 180, z축 90순
            elif self.rotMode=='horizontal_fixed':
                quat = quat_mul(quat_create(np.array([0, 0, 1.]), np.pi) ,quat_create(np.array([0, 1., 0]), np.pi/2)) #ref 기준 z축 180, y축 90순
                second_quat = quat_create(np.array([0, 1., 0]), np.pi/2) #ref 기준 y축 90
            else: #그 외 경우도 vertically initialize
                quat = quat_mul(quat_create(np.array([1., 0, 0]), np.pi) ,quat_create(np.array([0, 0, 1.]), np.pi/2)) #ref 기준 x축 180, z축 90순
                second_quat = quat_mul(quat_create(np.array([0, 1., 0]), np.pi) ,quat_create(np.array([0, 0, 1.]), np.pi/2)) #ref 기준 y축 180, z축 90순
            self.data.set_mocap_quat('mocap', quat)
            self.data.set_mocap_quat('second_mocap', second_quat)
            # self.do_simulation([-1,1], self.frame_skip)
            # self.do_simulation([0,0], self.frame_skip)
            self.do_simulation([0,0,0,0], self.frame_skip)
            
        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        second_rightFinger, second_leftFinger = self.get_site_pos('second_rightEndEffector'), self.get_site_pos('second_leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.second_init_fingerCOM  =  (second_rightFinger + second_leftFinger)/2
        self.pickCompleted = False

    def get_site_pos(self, siteName):
        _id = self.model.site_names.index(siteName)
        return self.data.site_xpos[_id].copy()

    def compute_rewards(self, actions, obsBatch):
        #Required by HER-TD3
        assert isinstance(obsBatch, dict) == True
        obsList = obsBatch['state_observation']
        rewards = [self.compute_reward(action, obs)[0] for  action, obs in zip(actions, obsList)]
        return np.array(rewards)

    def compute_reward(self, actions, obs, mode = 'general'):
        if isinstance(obs, dict):
            obs = obs['state_observation'] # [hand, second_hand, obj]
        
        
        objPos = obs[6:9]
        
        # raise NotImplementedError('아래 아직 안봄. 이 reward쓸지 안쓸지 판단하기')
        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        fingerCOM  =  (rightFinger + leftFinger)/2

        heightTarget = self.heightTarget
        placingGoal = self._state_goal

        reachDist = np.linalg.norm(objPos - fingerCOM)

        placingDist = np.linalg.norm(objPos - placingGoal)
      

        def reachReward():
            reachRew = -reachDist# + min(actions[-1], -1)/50
            reachDistxy = np.linalg.norm(objPos[:-1] - fingerCOM[:-1])
            zRew = np.linalg.norm(fingerCOM[-1] - self.init_fingerCOM[-1])
            if reachDistxy < 0.05: #0.02
                reachRew = -reachDist
            else:
                reachRew =  -reachDistxy - 2*zRew
            #incentive to close fingers when reachDist is small
            if reachDist < 0.05:
                reachRew = -reachDist + max(actions[-1],0)/50
            return reachRew , reachDist

        def pickCompletionCriteria():
            tolerance = 0.01
            if objPos[2] >= (heightTarget- tolerance):
                return True
            else:
                return False

        if pickCompletionCriteria():
            self.pickCompleted = True


        def objDropped():
            return (objPos[2] < (self.objHeight + 0.005)) and (placingDist >0.02) and (reachDist > 0.02) 
            # Object on the ground, far away from the goal, and from the gripper
            #Can tweak the margin limits
       
        def objGrasped(thresh = 0):
            sensorData = self.data.sensordata
            return (sensorData[0]>thresh) and (sensorData[1]> thresh)

        def orig_pickReward():       
            # hScale = 50
            hScale = 100
            if self.pickCompleted and not(objDropped()):
                return hScale*heightTarget
            # elif (reachDist < 0.1) and (objPos[2]> (self.objHeight + 0.005)) :
            elif (reachDist < 0.1) and (objPos[2]> (self.objHeight + 0.005)) :
                return hScale* min(heightTarget, objPos[2])
            else:
                return 0

        def general_pickReward():
            hScale = 50
            if self.pickCompleted and objGrasped():
                return hScale*heightTarget
            elif objGrasped() and (objPos[2]> (self.objHeight + 0.005)):
                return hScale* min(heightTarget, objPos[2])
            else:
                return 0

        def placeReward():
            # c1 = 1000 ; c2 = 0.03 ; c3 = 0.003
            c1 = 1000 ; c2 = 0.01 ; c3 = 0.001
            if mode == 'general':
                cond = self.pickCompleted and objGrasped()
            else:
                cond = self.pickCompleted and (reachDist < 0.1) and not(objDropped())
            if cond:
                placeRew = 1000*(self.maxPlacingDist - placingDist) + c1*(np.exp(-(placingDist**2)/c2) + np.exp(-(placingDist**2)/c3))
                placeRew = max(placeRew,0)
                return [placeRew , placingDist]
            else:
                return [0 , placingDist]

        reachRew, reachDist = reachReward()
        if mode == 'general':
            pickRew = general_pickReward()
        else:
            pickRew = orig_pickReward()
        placeRew , placingDist = placeReward()
        assert ((placeRew >=0) and (pickRew>=0))
        reward = reachRew + pickRew + placeRew
        return [reward, reachRew, reachDist, pickRew, placeRew, placingDist] 

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        return statistics

    def log_diagnostics(self, paths = None, logger = None):
        pass
