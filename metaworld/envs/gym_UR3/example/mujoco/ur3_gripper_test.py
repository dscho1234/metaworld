import gym
import numpy as np
from gym_UR3.envs.mujoco import MujocoUR3Env

import time





def main():
    env = gym.make('UR3-v0')
    Da = env.action_space.shape[0]
    obs=env.reset()
    start = time.time()
    
    for i in range(100):
        env.reset()
        print('{}th episode'.format(i+1))
        for j in range(100):
            env.render()
            # env.step(env.action_space.sample())
            a = np.zeros(8)
            a[:6] = 0.01*np.random.uniform(size = 6)
            a[-1] = 1
            a[-2] = 1
            env.step(a)
            
    end = time.time()

    print('Done! {}'.format(end-start))
#action[0] : qpos[0] radian

#action[4] : qpos[4] radian  

#action[5] : qpos[5] radian  

#action[6] : qpos[7] radian인가?? 여튼 밑에 finger

#action[7] : qpos[11] radian인가?? 여튼 위에 finger

#action[8] : qpos[15] radian인가?? 여튼 가운데 finger

#action[9] : qpos[6] qpos[10] radian인가?? 여튼 밑, 위 finger 위아래로 벌어짐



if __name__=="__main__":
    main()