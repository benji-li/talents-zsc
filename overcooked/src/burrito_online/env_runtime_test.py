from src.mdp.burrito_env import BurritoEnv
from src.mdp.steakhouse_mdp import SteakhouseGridworld
from src.planners.burrito_planner import HighAction
import numpy as np


if __name__ == '__main__':
    np.random.seed(12)
    mdp = SteakhouseGridworld.from_layout_name('burrito_3p')
    env = BurritoEnv.from_mdp(mdp, horizon=5000)
    env.setup_planner('burrito_3p', ['A', 'A', 'A'])
    env.reset()
    episode = 0
    info = None
    not_found = 0
    total_step = 0
    invalid_goal_steps = 0
    correct_dish_explored = 0
    while episode < 10:
        total_step += 1
        if info is not None:
            action = []
            action_status = info['action_status']
            solution_found = info['solution_found']
            if not solution_found:
                not_found += 1
            for agent in range(3):
                action_done = action_status[agent]['status']
                if not action_done:
                    agent_action = action_status[agent]['prev_action']
                else:
                    agent_action = np.random.choice(24)
                action.append(agent_action)
        else:
            action = np.random.choice(24, size=(3,)).tolist()
        obs, rew, done, info = env.step(action)
        for action in env.planner.actions:
            if action is None: continue
            goal = action.goal_pos
            if goal in HighAction.INVALID_GOALS:
                invalid_goal_steps += 1
                break
        if done:
            correct_dish_explored += info['episode']['ep_correct_delivery']
            env.reset()
            episode += 1
            info = None

    print("No planner solution times: ", not_found)
    print("Total env step running: ", total_step)
    print("Total invalid goals: ", invalid_goal_steps)
    print("Correct order delivery explored: ", correct_dish_explored)





# 3 times in 5 episodes where planner solution not found
# 183 seconds to run 25000
# 400 seconds to run 50000
# 100 - 150 step per second
### after masking out corridor goalss
# ~150 seconds to run 50000 steps, with all navigation actions, with ~<= 5 planner timeouts
### change to PIBT no replan
# ~ 90s to run 50000 steps
    

# CBS with invalid goal masking, cnt=2000: 100s, 30 delivery explored
# PIBT, 80s, 25 delivery explored