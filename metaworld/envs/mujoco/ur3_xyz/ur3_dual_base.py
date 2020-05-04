import abc
import copy

from gym.spaces import Discrete
import mujoco_py
import numpy as np


from metaworld.envs.mujoco.mujoco_env import MujocoEnv
from metaworld.envs.env_util import quat_to_zangle, zangle_to_quat, quat_create, quat_mul, ur3_quat_to_zangle, ur3_zangle_to_quat


OBS_TYPE = ['plain', 'with_goal_id', 'with_goal_and_id', 'with_goal', 'with_goal_init_obs']


class UR3MocapBase(MujocoEnv, metaclass=abc.ABCMeta):
    """
    Provides some commonly-shared functions for UR3 Mujoco envs that use
    mocap for XYZ control.
    """
    mocap_low = np.array([-0.2, 0.5, 0.06])
    mocap_high = np.array([0.2, 0.7, 0.6])

    def __init__(self, model_name, frame_skip=20):
        MujocoEnv.__init__(self, model_name, frame_skip=frame_skip)
        self.reset_mocap_welds()

    def get_endeff_pos(self):
        return self.data.get_body_xpos('hand').copy()
    #dscho mod
    def get_second_endeff_pos(self):
        return self.data.get_body_xpos('second_hand').copy()
    #dscho mod
    def get_endeff_quat(self):
        return self.data.get_body_xquat('hand').copy()
    #dscho mod
    def get_second_endeff_quat(self):
        return self.data.get_body_xquat('second_hand').copy()
        
    def get_gripper_pos(self):
        raise NotImplementedError('no 쓸모인듯')
        return np.array([self.data.qpos[7]])
        #qpos : 0~5 ur3, 6~9 grip_r, 10~13 grip_l, 14~19 second_ur3, 20~23 second_grip_r, 24~27 second_grip_l,  28~30 object pos, 31~34 object quat
        #qvel : 0~5 ur3, 6~9 grip_r, 10~13 grip_l, 14~19 second_ur3, 20~23 second_grip_r, 24~27 second_grip_l, 28~33 object vel, angvel
        

    def get_env_state(self):
        raise NotImplementedError('dscho did not modify because it doesnt seems to be used')
        joint_state = self.sim.get_state()
        mocap_state = self.data.mocap_pos, self.data.mocap_quat
        state = joint_state, mocap_state
        return copy.deepcopy(state)

    def set_env_state(self, state):
        raise NotImplementedError('dscho did not modify because it doesnt seems to be used')
        joint_state, mocap_state = state
        self.sim.set_state(joint_state)
        mocap_pos, mocap_quat = mocap_state
        self.data.set_mocap_pos('mocap', mocap_pos)
        self.data.set_mocap_quat('mocap', mocap_quat)
        self.sim.forward()

    def __getstate__(self):
        state = super().__getstate__()
        return {**state, 'env_state': self.get_env_state()}

    def __setstate__(self, state):
        super().__setstate__(state)
        self.set_env_state(state['env_state'])

    def reset_mocap_welds(self):
        """Resets the mocap welds that we use for actuation."""
        sim = self.sim
        if sim.model.nmocap > 0 and sim.model.eq_data is not None:
            for i in range(sim.model.eq_data.shape[0]):
                if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                    sim.model.eq_data[i, :] = np.array(
                        [0., 0., 0., 1., 0., 0., 0.])
        sim.forward()


class UR3DualXYZEnv(UR3MocapBase, metaclass=abc.ABCMeta):
    def __init__(
            self,
            *args,
            hand_low=(-0.2, 0.55, 0.05),
            hand_high=(0.2, 0.75, 0.3),
            second_hand_low=(-0.2, 0.55, 0.05),
            second_hand_high=(0.2, 0.75, 0.3),
            mocap_low=None,
            mocap_high=None,
            second_mocap_low=None,
            second_mocap_high=None,
            action_scale=2./100,
            action_rot_scale=1.,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.action_scale = action_scale
        self.action_rot_scale = action_rot_scale
        self.hand_low = np.array(hand_low)
        self.hand_high = np.array(hand_high)
        self.second_hand_low = np.array(second_hand_low)
        self.second_hand_high = np.array(second_hand_high)
        if mocap_low is None:
            mocap_low = hand_low
        if mocap_high is None:
            mocap_high = hand_high
        if second_mocap_low is None:
            second_mocap_low = second_hand_low
        if second_mocap_high is None:
            second_mocap_high = second_hand_high
        self.mocap_low = np.hstack(mocap_low)
        self.mocap_high = np.hstack(mocap_high)
        self.second_mocap_low = np.hstack(second_mocap_low)
        self.second_mocap_high = np.hstack(second_mocap_high)
        # We use continuous goal space by default and
        # can discretize the goal space by calling
        # the `discretize_goal_space` method.
        self.discrete_goal_space = None
        self.discrete_goals = []
        self.active_discrete_goal = None
    
    def set_xyz_action(self, action):
        #action : xyz * 2 arm
        action = np.clip(action, -1, 1)
        pos_delta = action[:3] * self.action_scale
        second_pos_delta = action[3:6] * self.action_scale
        new_mocap_pos = self.data.mocap_pos[0] + pos_delta[None]
        second_new_mocap_pos = self.data.mocap_pos[1] + second_pos_delta[None]
        new_mocap_pos[0, :] = np.clip(
            new_mocap_pos[0, :],
            self.mocap_low,
            self.mocap_high,
        )
        second_new_mocap_pos[0, :] = np.clip(
            second_new_mocap_pos[0, :],
            self.second_mocap_low,
            self.second_mocap_high,
        )
        
        self.data.set_mocap_pos('mocap', new_mocap_pos)
        self.data.set_mocap_pos('second_mocap', second_new_mocap_pos)

        if self.rotMode =='vertical_fixed':
            quat = quat_mul(quat_create(np.array([1., 0, 0]), np.pi) ,quat_create(np.array([0, 0, 1.]), np.pi/2)) #ref 기준 x축 180, z축 90순
            second_quat = quat_mul(quat_create(np.array([0, 1., 0]), np.pi) ,quat_create(np.array([0, 0, 1.]), np.pi/2)) #ref 기준 y축 180, z축 90순
        elif self.rotMode =='horizontal_fixed':
            quat = quat_mul(quat_create(np.array([0, 0, 1.]), np.pi), quat_create(np.array([0, 1., 0]), np.pi/2)) #ref 기준 z축 180, y축 90순
            second_quat = quat_create(np.array([0, 1., 0]), np.pi/2) #ref 기준 y축 90
        #TODO: if it is not proper, you should consider different quat for second hand(100퍼 고쳐야함!)
        self.data.set_mocap_quat('mocap', quat) #w v 순인듯
        self.data.set_mocap_quat('second_mocap', second_quat) #w v 순인듯
        # self.data.set_mocap_quat('mocap', np.array([1, 0, 0, 0])) #w v 순인듯

    def set_xyz_action_rot(self, action):
        #action : xyz, quat * 2 arm
        action[:3] = np.clip(action[:3], -1, 1)
        action[7:10] = np.clip(action[7:10], -1, 1)
        pos_delta = action[:3] * self.action_scale
        second_pos_delta = action[7:10] *self.action_scale
        new_mocap_pos = self.data.mocap_pos[0] + pos_delta[None]
        second_new_mocap_pos = self.data.mocap_pos[1] + second_pos_delta[None]
        new_mocap_pos[0, :] = np.clip(
            new_mocap_pos[0, :],
            self.mocap_low,
            self.mocap_high,
        )
        second_new_mocap_pos[0, :] = np.clip(
            second_new_mocap_pos[0, :],
            self.second_mocap_low,
            self.second_mocap_high,
        )
        rot_axis = action[4:7] / np.linalg.norm(action[4:7])
        second_rot_axtis = action[11:14] / np.linalg.norm(action[11:14])
        action[3] = action[3] * self.action_rot_scale
        action[10] = action[10] * self.action_rot_scale
        self.data.set_mocap_pos('mocap', new_mocap_pos)
        self.data.set_mocap_pos('second_mocap', second_new_mocap_pos)
        # replace this with learned rotation

        
        quat = quat_mul(quat_create(np.array([0, 1., 0]), np.pi),
                        quat_create(np.array(rot_axis).astype(np.float64), action[3]))

        #TODO: if it is not proper, you should consider different quat for second hand(100퍼 고쳐야함!)
        self.data.set_mocap_quat('mocap', quat)
        self.data.set_mocap_quat('second_mocap', quat)
        # self.data.set_mocap_quat('mocap', np.array([np.cos(action[3]/2), np.sin(action[3]/2)*rot_axis[0], np.sin(action[3]/2)*rot_axis[1], np.sin(action[3]/2)*rot_axis[2]]))
        # self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))

    def set_xyz_action_rotz(self, action):
        #action : xyz, rotz * 2 arm
        action[:3] = np.clip(action[:3], -1, 1)
        action[4:7] = np.clip(action[4:7], -1, 1)
        pos_delta = action[:3] * self.action_scale
        second_pos_delta = action[4:7] * self.action_scale
        new_mocap_pos = self.data.mocap_pos[0] + pos_delta[None]
        second_new_mocap_pos = self.data.mocap_pos[1] + second_pos_delta[None]
        new_mocap_pos[0, :] = np.clip(
            new_mocap_pos[0, :],
            self.mocap_low,
            self.mocap_high,
        )
        second_new_mocap_pos[0, :] = np.clip(
            second_new_mocap_pos[0, :],
            self.second_mocap_low,
            self.second_mocap_high,
        )
        self.data.set_mocap_pos('mocap', new_mocap_pos)
        self.data.set_mocap_pos('second_mocap', second_new_mocap_pos)
        zangle_delta = action[3] * self.action_rot_scale
        second_zangle_delta = action[7] * self.action_rot_scale
        new_mocap_zangle = ur3_quat_to_zangle(self.data.mocap_quat[0]) + zangle_delta
        second_new_mocap_zangle = ur3_quat_to_zangle(self.data.mocap_quat[1]) + second_zangle_delta
        # new_mocap_zangle = action[3]
        new_mocap_zangle = np.clip(
            new_mocap_zangle,
            -3.0,
            3.0,
        )
        second_new_mocap_zangle = np.clip(
            second_new_mocap_zangle,
            -3.0,
            3.0,
        )
        if new_mocap_zangle < 0:
            new_mocap_zangle += 2 * np.pi
        if second_new_mocap_zangle < 0:
            second_new_mocap_zangle += 2 * np.pi
        self.data.set_mocap_quat('mocap', ur3_zangle_to_quat(new_mocap_zangle))
        self.data.set_mocap_quat('second_mocap', ur3_zangle_to_quat(second_new_mocap_zangle))

    def set_xy_action(self, xy_action, fixed_z):
        raise NotImplementedError('dscho did not modify because it doesnt seems to be used')
        delta_z = fixed_z - self.data.mocap_pos[0, 2]
        xyz_action = np.hstack((xy_action, delta_z))
        self.set_xyz_action(xyz_action)

    def discretize_goal_space(self, goals=None):
        if goals is None:
            self.discrete_goals = [self.default_goal]
        else:
            assert len(goals) >= 1
            self.discrete_goals = goals
        # update the goal_space to a Discrete space
        self.discrete_goal_space = Discrete(len(self.discrete_goals))

    # Belows are methods for using the new wrappers.
    # `sample_goals` is implmented across the sawyer_xyz
    # as sampling from the task lists. This will be done
    # with the new `discrete_goals`. After all the algorithms
    # conform to this API (i.e. using the new wrapper), we can
    # just remove the underscore in all method signature.
    def sample_goals_(self, batch_size):
        if self.discrete_goal_space is not None:
            return [self.discrete_goal_space.sample() for _ in range(batch_size)]
        else:
            return [self.goal_space.sample() for _ in range(batch_size)]

    def set_goal_(self, goal):
        if self.discrete_goal_space is not None:
            self.active_discrete_goal = goal
            self.goal = self.discrete_goals[goal]
            self._state_goal_idx = np.zeros(len(self.discrete_goals))
            self._state_goal_idx[goal] = 1.
        else:
            self.goal = goal
    
    def set_init_config(self, config):
        assert isinstance(config, dict)
        for key, val in config.items():
            self.init_config[key] = val

    '''
    Functions that are copied and pasted everywhere and seems
    to be not used.
    '''
    def sample_goals(self, batch_size):
        '''Note: should be replaced by sample_goals_ if not used''' 
        # Required by HER-TD3
        goals = self.sample_goals_(batch_size)
        if self.discrete_goal_space is not None:
            goals = [self.discrete_goal_space[g].copy() for g in goals]
        return {
            'state_desired_goal': goals,
        }

    def sample_task(self):
        '''Note: this can be replaced by sample_goal_(batch_size=1)'''
        goal = self.sample_goals_(1)
        if self.discrete_goal_space is not None:
            return self.discrete_goals[goal]
        else:
            return goal

    def _set_obj_xyz_quat(self, pos, angle):
        raise NotImplementedError('dscho did not modify because it doesnt seems to be used')
        quat = quat_create(np.array([0, 0, .1]), angle)
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy()
        qpos[12:16] = quat.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def _set_obj_xyz(self, pos):
        raise NotImplementedError('dscho did not modify because it is gonna be orverrided in subclass')
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)
