import sys
import os
import traceback
from burrito.mdp.burrito_mdp import BurritoGridworld, BurritoState 
import numpy as np
from overcooked_ai_py.mdp.overcooked_mdp import (
    Action,
    Direction,
)
from enum import Enum
from collections import deque
from copy import deepcopy

import sys,os


def evaluate_condition(cond, flags):
    # Base case: if cond is a string, treat it as a flag name.
    # Support a "not_" prefix for negation.
    if isinstance(cond, str):
        if cond.startswith("not_"):
            flag_name = cond[4:]
            return not flags.get(flag_name, False)
        return flags.get(cond, False)
    
    # If cond is a dictionary, it should have a single key: the operator.
    if isinstance(cond, dict):
        if len(cond) != 1:
            raise ValueError("Condition dictionaries must have a single logical operator.")
        operator, subconds = next(iter(cond.items()))
        
        if operator == 'all':
            # 'all' returns True only if all subconditions are True.
            return all(evaluate_condition(sub, flags) for sub in subconds)
        elif operator == 'or':
            # 'any' returns True if at least one subcondition is True.
            return any(evaluate_condition(sub, flags) for sub in subconds)
        elif operator == 'xor':
            # 'xor' returns True if exactly one subcondition is True.
            results = [evaluate_condition(sub, flags) for sub in subconds]
            return results.count(True) == 1
        else:
            raise ValueError(f"Unknown operator: {operator}")
    
    # If the condition is neither a string nor a dict, we don't know how to handle it.
    raise ValueError("Condition must be either a string or a dictionary.")


class HighAction:
    INVALID_GOALS = [] #[(4,8), (5,8), (6,8), (7,8)]
    BACKUP_GOALS = [] #[[(3,8),(0,1)], [(4,7), (1,0)], [(2,8),(0,1)], [(8,8),(0,1)], [(7,7),(-1,0)], [(9,8), (0,1)]]
    def __init__(self, *args, **kwargs):
        self.goal_pos = 'current'
        self.goal_or = 'current'
        self.to_interact = 0

    @property
    def nav_done(self):
        raise NotImplementedError
    
    @property
    def interact_done(self):
        raise NotImplementedError
    
    @property
    def goal(self):
        return self.goal_pos, self.goal_or

    @property
    def done(self):
        """
        High-level action status
        """
        raise NotImplementedError
    
    def valid_on_step_start(self, agent, terrain_mtx, burrito_state:BurritoState):
        return True

    def update_on_step_end(self, action:Action):
        """
        Called on execution of low level action for this high level plan for one step
        Update the action status
        """
        raise NotImplementedError
            


class NavAction(HighAction):
    """
    High-level actions that require navigation, and potentially one interaction with fixed properties
    This can include the low-level navigation actions (up, down, left, right)
    """
    def __init__(self, agent, grid_distances, terrain_mtx, terrain_pos_dict, burrito_state:BurritoState, **kwargs):
        self.stay = False
        self.terrain_name = kwargs.get('terrain_type', None)
        self.name_in_hand = kwargs.get('name_in_hand', None)
        self.name_in_terrain = kwargs.get('name_in_terrain', None)
        self.id = kwargs.get('id', None)
        self.to_interact = kwargs.get('int_at_end', 0)
        self.required_object_state = kwargs.get('required_object_state', None)
        self.logical_combine = kwargs.get('logical_combine', None)
        self.nav_status = False
        self.inter_status = False
        self.set_goal(agent, grid_distances, terrain_mtx, terrain_pos_dict, burrito_state)
        self.set_interact(terrain_pos_dict, burrito_state)
    

    def set_interact(self, terrain_pos_dict, burrito_state:BurritoState):
        if self.target_pos == self.agent_pos:
            # the desired action is invalid, set interact to 0
            self.to_interact = 0
            return
        if not type(self.to_interact) == dict:
            return
        if (self.target_pos not in terrain_pos_dict['B']) and (self.target_pos not in terrain_pos_dict['W']):
            return
    
        if burrito_state.has_object(self.target_pos):
            target_obj = burrito_state.get_object(self.target_pos)        
            try:
                if hasattr(target_obj, "_cook_time"):
                    max_interact = target_obj._cook_time - target_obj._cooking_tick
                else:
                    max_interact = 1
                min_interact = 1
            except Exception as e:
                print(f"Exception occurred: {e}", target_obj.name)
                traceback.print_exc()  # This will print the full traceback
                print("================================***************************")

        else:
            max_interact = self.to_interact['max_interact']
            min_interact = 1
        self.to_interact = np.random.randint(min_interact, max_interact+1)

    def set_goal(self, agent, grid_distances, terrain_mtx, terrain_pos_dict, burrito_state:BurritoState):
        """
        Set goal will only pick goal positions that follow the `name_in_terrain` constraint.
            Assume the `name_in_hand` constraint is verified by `valid_on_step_start`
        """
        # NOTE: enforce passing? 
        if agent == 0:
            invalid_targets = [(i,j) for i in range(6,12) for j in range(0,10)]
        if agent == 1:
            invalid_targets = [(i,j) for i in range(0,6) for j in range(0,10)]
            invalid_targets.remove((5,1))
            invalid_targets.remove((5,2))
            invalid_targets.remove((5,3))

        invalid_targets = []
        self.agent_pos = burrito_state.players[agent].position
        self.agent_or = burrito_state.players[agent].orientation
        # print(agent, invalid_targets)
        # print("agent: ", agent)
        # print("terrain: ", self.terrain_name)
        # print("object desired in terrain: ", self.name_in_terrain)
        # print("object supposed in agent hand: ", self.name_in_hand)

        self.target_pos = self.agent_pos
        self.goal_pos = self.agent_pos
        self.goal_or = self.agent_or

        if not self.terrain_name and (self.name_in_hand is None and self.name_in_terrain is None):
            # STAY action
            self.stay = True
            return

        try:
            # get all possible target terrain / object pos
            terrain_target_pos_list = []
            object_target_pos_list = []
            target_pos_list = []
            if self.terrain_name:
                for terrain_name in self.terrain_name:
                    for pos in terrain_pos_dict[terrain_name]:
                        # print(terrain_name, pos)
                        if pos in invalid_targets:
                            continue
                        terrain_target_pos_list.append(pos)
                    # terrain_target_pos_list += terrain_pos_dict[terrain_name]
            if self.id is not None:
                terrain_target_pos_list = self.id.copy()
            
            # print('terrain_pos_list: ', terrain_target_pos_list)

            if self.name_in_terrain and ' ' not in self.name_in_terrain:
                # terrain shouldn't be empty
                for position, object in burrito_state.objects.items():
                    if position in invalid_targets:
                        continue
                    if object.name in self.name_in_terrain or 'any' in self.name_in_terrain:
                        # print(f"object {object.name} in terrain objects {self.name_in_terrain}, type is {type(object)}")
                        if self.required_object_state:
                            state_flag = False
                            for required_state in self.required_object_state:
                                if '_not_' in required_state:
                                    required_state_ = required_state.replace("_not_", "")
                                    # print(f'obj in terrain at state {required_state_} is {getattr(object, required_state_, True)}, False desired')
                                    if not getattr(object, required_state_, True):
                                        state_flag = True
                                        # print(state_flag)
                                        break
                                else:
                                    # print(f'obj in terrain at state {required_state} is {getattr(object, required_state, False)}, True desired')
                                    if getattr(object, required_state, False):
                                        state_flag = True
                                        # print(state_flag)
                                if state_flag:
                                    object_target_pos_list.append(position)
                                    break
                        else:
                            # print(f"append target object {object.name} position {position}")
                            object_target_pos_list.append(position)
                # print('object_target_pos_list: ', object_target_pos_list)
                if self.logical_combine:
                    if 'all' in self.logical_combine and 'terrain' in self.logical_combine['all'] and 'object' in self.logical_combine['all']:
                        for target in terrain_target_pos_list:
                            if target in object_target_pos_list:
                                target_pos_list.append(target)
                    else:
                        target_pos_list = object_target_pos_list + terrain_target_pos_list
                else:
                    target_pos_list = object_target_pos_list + terrain_target_pos_list
            elif self.name_in_terrain and ' ' in self.name_in_terrain:
                # empty terrain is allowed
                target_pos_list = terrain_target_pos_list.copy()
            else:
                # there shouldn't be any object in the terrain
                trimmed_terrain_target_pos_list = []
                for idx, pos in enumerate(terrain_target_pos_list):
                    if not burrito_state.has_object(pos):
                        trimmed_terrain_target_pos_list.append(pos)
                target_pos_list = trimmed_terrain_target_pos_list
            
            # print('terrain_pos_list: ', terrain_target_pos_list)
            # print('object_target_pos_list: ', object_target_pos_list)
            # print('target_pos_list: ', target_pos_list)
            # compute candidate agent goal position and orientation to interact with that terrain / object
            candidate_list = []
            for target_pos in target_pos_list:
                for direction in Direction.ALL_DIRECTIONS:
                    new_pos = Action.move_in_direction(target_pos, direction)
                    # print(new_pos)
                    if (not new_pos in grid_distances) or (self.agent_pos not in grid_distances[new_pos]) or new_pos in HighAction.INVALID_GOALS or new_pos in invalid_targets:
                        continue
                    else:
                        candidate_list.append((grid_distances[new_pos][self.agent_pos], target_pos, new_pos, direction))

            # pick the closest valid candidate
            sorted_candidate_list = sorted(candidate_list)
            for _, target_pos, new_pos, direction in sorted_candidate_list:
                self.target_pos = target_pos
                self.goal_pos = new_pos
                self.goal_or = Direction.OPPOSITE_DIRECTIONS[direction]
                # print(f'verify target {target_pos}, goal pos {new_pos}')
                if self.valid_on_step_start(agent, terrain_mtx, burrito_state):
                    return
            self.goal_pos = self.agent_pos
            self.goal_or = self.agent_or
            self.target_pos = self.agent_pos
        
        except Exception as e:
            if len(target_pos_list) == 0 or len(candidate_list) == 0:
                # print(f"target terrain {self.terrain_name} or object {self.name_in_terrain} does not exist in the environment")
                pass
            else:
                print(f"Exception occurred: {e}")
                traceback.print_exc()  # This will print the full traceback
                print("================================***************************")
            self.goal_pos = self.agent_pos
            self.goal_or = self.agent_or
            self.target_pos = self.agent_pos
        return

    @property
    def nav_done(self):
        return self.nav_status
    
    @property
    def interact_done(self):
        return (not self.to_interact)

    @property
    def done(self):
        # print("interact status: ", self.interact_done)
        # print("nav: ", self.nav_status)
        return self.nav_done and self.interact_done

    def valid_on_step_start(self, agent, terrain_mtx, burrito_state: BurritoState):
        """
        Judge if the action is still valid
        """
        # print(f"valid agent {agent} action")
        # print("terrain: ", self.terrain_name)
        # print("object desired in terrain: ", self.name_in_terrain)
        # print("object supposed in agent hand: ", self.name_in_hand)

        player = burrito_state.players[agent]
        self.agent_pos = player.position
        self.agent_or = player.orientation

        # The goal position has been ocupied by others
        # if self.goal_pos in burrito_state.player_positions:
        #     if list(burrito_state.player_positions).index(self.goal_pos) != agent:
        #         return False
        # print("goal position valid for no other agents there")
        
        terrain_flag = True
        object_flag = True
        player_flag = True
        state_flag = True
        empty_hand = True

        # The agent should hold correct object in hand
        if self.name_in_hand:
            if not player.has_object() and ' ' not in self.name_in_hand:
                player_flag = False
            elif not player.has_object() and ' ' in self.name_in_hand:
                player_flag = True
            elif player.has_object():
                empty_hand = False
                obj_name = player.get_object().name
                # print(f'object {obj_name} in hand')
                # print(obj_name not in self.name_in_hand)
                # print('any' not in self.name_in_hand)
                if ('any' not in self.name_in_hand) and (obj_name not in self.name_in_hand):
                    player_flag = False
        else:
            # The agent shouldn't hold anything, but holding
            if player.has_object():
                empty_hand = False
                player_flag = False

        # terrain is specified, but agent is not going there
        if self.terrain_name:
            y, x = self.target_pos
            if terrain_mtx[x,y] not in self.terrain_name:
                # print(f"{self.target_pos} not in {self.terrain_name}")
                terrain_flag = False

        # There should be correct object with correct state (if required) in terrain
        if self.name_in_terrain:
            if burrito_state.has_object(self.target_pos):
                obj = burrito_state.get_object(self.target_pos)
                # print(f'obj in terrain is {obj.name}')
                if (obj.name not in self.name_in_terrain) and ('any' not in self.name_in_terrain):
                    object_flag =  False
                if self.required_object_state:
                    state_flag = False
                    for required_state in self.required_object_state:
                        if '_not_' in required_state:
                            required_state_ = required_state.replace("_not_", "")
                            # print(f'obj in terrain at state {required_state_} is {getattr(obj, required_state_, False)}')
                            if not getattr(obj, required_state_, True):
                                state_flag = True
                                # print(state_flag)
                                break
                        else:
                            if getattr(obj, required_state, False):
                                state_flag = True
                    # if not state_flag:
                    #     object_flag =  False
            elif ' ' not in self.name_in_terrain:
                object_flag = False
        else:
            # There shouldn't be object in the terrain
            if burrito_state.has_object(self.target_pos):
                terrain_flag = False


        if self.logical_combine:
            condition_flags = {'terrain': terrain_flag, 'object': object_flag, 'player': player_flag, 'state': state_flag, 'empty_hand': empty_hand}
            # print("current conditions: ", condition_flags)
            # print("required condition: ", self.logical_combine)
            return evaluate_condition(self.logical_combine, condition_flags)
        # print(f"action is valid for {agent}")
        return True

    def update_on_step_end(self, action):
        if action in Action.MOTION_ACTIONS:
            new_agent_pos = Action.move_in_direction(self.agent_pos, action)
            if action == Action.STAY:
                new_agent_or = self.agent_or
            else:
                new_agent_or = action
            if new_agent_pos == self.goal_pos:
                if not self.stay:
                    if new_agent_or == self.goal_or:
                        self.nav_status = True
                else:
                    self.nav_status = True
        else:
            # --> action == Action.INTERACT
            if self.nav_done: 
                self.to_interact -= 1
                # print(f"after 1 interaction, {self.to_interact} left")

    def fallback_to(self, goal_pos, goal_or):
        self.goal_pos = goal_pos
        self.goal_or = goal_or
        self.terrain_name = None
        self.to_interact = 0
        self.name_in_terrain = None
        self.name_in_hand = None



class InterAction(HighAction):
    """
    High-level actions that don't require navigation. STAY is considered as InterAction.
    """
    def __init__(self, *args, **kwargs):
        self.to_interact = kwargs.get('int_at_end', 1)
        self.object_name = kwargs.get('name', None)
        self.terrain_name = kwargs.get('terrain_type', None)
        self.goal_pos = 'current'
        self.goal_or = 'current'
        self.set_interact(args[0], args[3])
        # print(f"{self.to_interact} times of interaction set for action {self.object_name}")
    
    @property
    def nav_done(self):
        return True
    
    @property
    def interact_done(self):
        return (not self.to_interact)

    @property
    def done(self):
        return (not self.to_interact)
    

    def set_interact(self, agent, burrito_state:BurritoState):
        """
        For multi-step interaction actions, if object is already there, don't need to put down
        """
        self.agent_pos = burrito_state.players[agent].position
        self.goal_pos = self.agent_pos
        self.agent_or = burrito_state.players[agent].orientation
        self.goal_or = self.agent_or
        obj_pos = Action.move_in_direction(self.agent_pos, self.agent_or)
        if burrito_state.has_object(obj_pos):
            obj = burrito_state.get_object(obj_pos)
            if obj.name == self.object_name:
                self.to_interact -= 1


    def valid_on_step_start(self, agent, terrain_mtx, burrito_state: BurritoState):
        self.agent_pos = burrito_state.players[agent].position
        self.agent_or = burrito_state.players[agent].orientation
        # not able to interact with incorrect terrains
        terrain_pos = Action.move_in_direction(self.agent_pos, self.agent_or)
        terrain_name = terrain_mtx[terrain_pos[1], terrain_pos[0]]
        terrain_list = self.terrain_name
        if terrain_name not in terrain_list:
            return False
        
        agent_state = burrito_state.players[agent]
        if agent_state.held_object:
            player_obj = agent_state.get_object().name
            # If the agent is not holding correct object, interaction action not available
            if self.object_name and player_obj != self.object_name[0]:
                return False
            else:
                return True
        else:
            # If agent not holding object, then the correct object should on the correct terrain
            # Only chopped meat, chopped mushroom, and clean plate can be processed when they are already on the terrain
            if burrito_state.has_object(terrain_pos):
                obj = burrito_state.get_object(terrain_pos)
                obj_name = obj.name
                obj_info_dict = obj.to_dict()
                
                if obj.is_ready:
                    self.to_interact = 0
                    return False

                if self.object_name and len(self.object_name)>1 and obj_name == self.object_name[1]:

                    return True
        return False
        

    def update_on_step_end(self, action):
        if action == Action.INTERACT:
            self.to_interact -= 1
        # print(f"update required interaction time: {self.to_interact}")



class HighLevelActions(Enum):
    """
    45 Nav or Inter actions
    """
    # Navigate to fixed properties (10) from idx 1
    GO_TO_PAN_AND_FRY_INGREDIENT = (NavAction, {
        'terrain_type': ['G'],
        'name_in_hand': ['chopped_meat', 'chopped_mushroom'],
        'int_at_end': 1,
        'logical_combine': {'all': ['terrain', 'player', 'object']}
    }, 0)
    GO_TO_POT_AND_BOIL_RICE = (NavAction, {
        'terrain_type': ['P'],
        'name_in_hand': ['rice'],
        'int_at_end': 1,
        'logical_combine': {'all': ['terrain', 'player', 'object']}
    }, 1)
    # NOTE: cannot deal with the situation to chop ingredients others put down
    GO_TO_CHOP_BOARD_AND_CHOP_INGREDIENT = (NavAction, {
        'terrain_type': ['B'],
        'name_in_hand': ['meat', 'mushroom'],
        'name_in_terrain': ['chopped_meat', 'chopped_mushroom'],
        'required_object_state': ['_not_is_ready'],
        # 'logical_combine': {'all':[{'all': ['terrain']}, {'xor': ['player', 'object']}]},
        'logical_combine': {'all':[{'all':['terrain']}, {'or':[{'all':['player', 'not_object']}, {'all': ['empty_hand', 'object', 'state']}]}]},
        'int_at_end': {'max_interact': 3}
    }, 2)
    GO_TO_PUT_OUT_FIRE = (NavAction, {
        'terrain_type': ['P', 'G'],
        'name_in_hand': ['fire_ext'],
        'name_in_terrain': ['chopped_steak', 'fried_mushroom', 'boiled_rice'],
        'required_object_state': ['is_burnt'],
        'logical_combine': {'all': ['terrain', 'player', 'object']},
        'int_at_end': 1,
    }, 3)
    STAY = (NavAction, {
        'int_at_end': 0
    }, 4) # make STAY to action index 4, corresponding to the low-level stay action
    # The logic is not sound when a clean plate is ready and in the sink
    GO_TO_SINK_TO_WASH_PLATE = (NavAction, {
        'terrain_type': ['W'],
        'int_at_end': {'max_interact': 4},
        'name_in_hand': ['dirty_plate'],
        'name_in_terrain': ['dirty_plate'],
        # 'logical_combine': {'all':[{'all': ['terrain']}, {'xor': ['player', 'object']}]},
        'logical_combine': {'all':[{'all':['terrain']}, {'or':[{'all':['player', 'not_object']}, {'all': ['empty_hand', 'object']}]}]},
    }, 5)
    GO_TO_TRASH_AND_THROW = (NavAction, {
        'terrain_type': ['U'],
        'int_at_end': 1,
        'logical_combine': {'all': ['terrain', 'player']},
        'name_in_hand': ["charcoal"]  # ["charcoal"]
    }, 6)
    GO_TO_SERVE_DISH = (NavAction, {
        'terrain_type': ['S'],
        'name_in_hand': ['steak_burrito', 'mushroom_burrito'],
        'logical_combine': {'all': ['terrain', 'player']},
        'int_at_end': 1,
    }, 7)
    GRAB_MEAT = (NavAction, {
        'terrain_type': ['M'], 
        'name_in_terrain': ['meat'], 
        'name_in_hand': [' '],
        'logical_combine': {'xor': [{'all':['terrain', 'player']}, {'all':['object']}]},
        'int_at_end': 1
    }, 8)
    GRAB_MUSHROOM = (NavAction, {
        'terrain_type': ['Z'], 
        'name_in_terrain': ['mushroom'], 
        'name_in_hand': [' '],
        'logical_combine': {'xor': [{'all':['terrain', 'player']}, {'all':['object']}]},
        'int_at_end': 1
    }, 9)
    GRAB_TORTILLA = (NavAction, {
        'terrain_type': ['T'], 
        'name_in_terrain': ['tortilla'],
        'name_in_hand': [
            ' ', # nothing in hand
            'clean_plate',
            'fried_mushroom-plate',
            'chopped_steak-plate',
            'boiled_rice-plate',
            'chopped_steak-boiled_rice-plate',
            'fried_mushroom-boiled_rice-plate'
        ], 
        'logical_combine': {'all': [{'all': ['player']}, {'or': ['terrain', 'object']}]},
        'int_at_end': 1
    }, 10)
    GRAB_RICE = (NavAction, {
        'terrain_type': ['R'], 
        'name_in_terrain': ['rice'], 
        'name_in_hand': [' '],
        'logical_combine': {'xor': [{'all':['terrain', 'player']}, {'all':['object']}]},
        'int_at_end': 1
    }, 11)
    GRAB_DIRTY_PLATE = (NavAction, {
        'terrain_type': ['D'],
        'name_in_terrain': ['dirty_plate'],
        'name_in_hand': [' '],
        'logical_combine': {'xor': [{'all':['terrain', 'player']}, {'all':['object']}]},
        'int_at_end': 1,
    }, 12)
    GRAB_CLEAN_PLATE = (NavAction, {
        'terrain_type': ['W'],
        'name_in_terrain': ['clean_plate'],
        'int_at_end': 1,
        'name_in_hand': [
            ' ', # nothing in hand
        ],
        # 'required_object_state': ['is_ready'],
        'logical_combine': {'xor': [{'all':['terrain', 'player', 'object']}, {'all':['object', 'not_terrain']}]},
    }, 13)
    GRAB_FIRE_EXT = (NavAction, {
        'name_in_terrain': ['fire_ext'],
        'int_at_end': 1,
        'name_in_hand': [
            ' ', # nothing in hand
            'any' # or anything in hand, can switch with ext
        ],
        'logical_combine': {'all': ['object', 'player']},
    }, 14)
    GRAB_CHOPPED_MUSHROOM = (NavAction, {
        'terrain_type': ['B'],
        'name_in_terrain': ['chopped_mushroom'],
        'name_in_hand': [' '],
        'int_at_end': 1,
        'required_object_state': ['is_ready'],
        'logical_combine': {'xor': [{'all':['terrain', 'player', 'object', 'state']}, {'all':['object', 'not_terrain']}]},
    }, 15)
    GRAB_CHOPPED_MEAT = (NavAction, {
        'terrain_type': ['B'],
        'name_in_terrain': ['chopped_meat'],
        'name_in_hand': [' '],
        'int_at_end': 1,
        'required_object_state': ['is_ready'],
        'logical_combine': {'xor': [{'all':['terrain', 'player', 'object', 'state']}, {'all':['object', 'not_terrain']}]},
    }, 16)
    GRAB_ONE_INGREDIENT_PLATE = (NavAction, {
        'name_in_terrain': [
            'fried_mushroom-plate',
            'chopped_steak-plate',
            'boiled_rice-plate',
            'tortilla-plate'
        ], 
        'int_at_end': 1,
        'name_in_hand': [' ', 'any'],
        'logical_combine': {'all': ['object', 'player']},
    }, 17)
    GRAB_TWO_INTREDIENT_PLATE = (NavAction, {
        'name_in_terrain': [
            'chopped_steak-tortilla-plate',
            'boiled_rice-tortilla-plate',
            'fried_mushroom-tortilla-plate',
            'chopped_steak-boiled_rice-plate',
            'fried_mushroom-boiled_rice-plate'
        ], 
        'int_at_end': 1,
        'name_in_hand': [' ', 'any'],
        'logical_combine': {'all': ['object', 'player']},
    }, 18)
    GRAB_MURHSOOM_BURRITO_PLATE = (NavAction, {
        'name_in_terrain': [
            'mushroom_burrito'
        ],
        'int_at_end': 1,
        'name_in_hand': [' ', 'any'],
        'logical_combine': {'all': ['object', 'player']},
    }, 19)
    GRAB_STEAK_BURRITO_PLATE = (NavAction, {
        'name_in_terrain': [
            'steak_burrito'
        ],
        'int_at_end': 1,
        'name_in_hand': [' ', 'any'],
        'logical_combine': {'all': ['object', 'player']},
    }, 20)
    GET_MURHSOOM_FROM_GRILLER = (NavAction, {
        'name_in_terrain': [
            'fried_mushroom'
        ],
        'name_in_hand': [
            'clean_plate',
            'boiled_rice-plate',
            'tortilla-plate',
            'boiled_rice-tortilla-plate'
        ],
        'required_object_state': ['is_waiting_for_pickup', 'is_warning'],
        'int_at_end': 1,
        'logical_combine': {'all': ['object', 'player', 'state']},
    }, 21)
    GET_STEAK_FROM_GRILLER = (NavAction, {
        'name_in_terrain': [
            'chopped_steak'
        ],
        'name_in_hand': [
            'clean_plate',
            'boiled_rice-plate',
            'tortilla-plate',
            'boiled_rice-tortilla-plate'
        ],
        'required_object_state': ['is_waiting_for_pickup', 'is_warning'],
        'int_at_end': 1,
        'logical_combine': {'all': ['object', 'player', 'state']},
    }, 22)
    GET_BOILED_RICE = (NavAction, {
        'name_in_terrain': ['boiled_rice'],
        'name_in_hand': [
            'clean_plate',
            'tortilla-plate',
            'fried_mushroom-plate',
            'chopped_steak-plate',
            'chopped_steak-tortilla-plate',
            'fried_mushroom-tortilla-plate'
        ],
        'required_object_state': ['is_waiting_for_pickup', 'is_warning'],
        'int_at_end': 1,
        'logical_combine': {'all': ['object', 'player', 'state']},
    }, 23)
    GRAB_CHARCOAL = (NavAction, {
        'terrain_type': ['G', 'P'],
        'name_in_terrain': ['charcoal'],
        'int_at_end': 1,
        'name_in_hand': [
            ' ', 'rice', 'chopped_meat', 'chopped_mushroom' # or anything in hand, can switch with charcoal
        ],
        'logical_combine': {'xor': [{'all':['terrain', 'player', 'object']}, {'all':['object', 'not_terrain']}]},
    }, 24)
    PASS_OBJECT = (NavAction, {
        'terrain_type': ['X'], 
        'id': [(5,1),(5,2),(5,3),(4,3),(4,6)], #[(4,3),(4,6)], #<-counters for FC #[(5,1),(5,2),(5,3)],  # Uhhh, so ugly
        'int_at_end': 1,
        'name_in_hand': ['any'],
        'name_in_terrain': ['any', ' '],
        'logical_combine': {'all': ['object', 'player', 'terrain']},
    }, 25) # might swap object in hand
    PUT_DOWN_OBJECT = (NavAction, {
        'terrain_type': ['X'], 
        'int_at_end': 1, 
        'name_in_hand': ['any'],
        'logical_combine': {'all': ['object', 'player', 'terrain']},
    }, 26) # think of something more elegant than 'place' parameter

    def __init__(self, action_class, action_kwargs, action_index):
        self._action_class = action_class
        self._action_kwargs = action_kwargs
        self._action_index = action_index

    @property
    def action_class(self):
        return self._action_class

    @property
    def action_kwargs(self):
        return self._action_kwargs
    
    @property
    def action_index(self):
        return self._action_index


    @staticmethod
    def get_navigation_indices():
        return [i for i in range(1, 34)]

    @staticmethod
    def get_interact_indices():
        return [0] + [i for i in range(34, 43)]
    
    @staticmethod
    def agent_action_mask(agent, grid_distances, terrain_pos_dict, terrain_mtx, burrito_state, **action_kwargs):
        # call all the valid at step start methods
        valid_mask = []
        for action_init in list(HighLevelActions):
            action_cls = action_init.action_class
            action_kwargs = action_init.action_kwargs
            action = action_cls(agent, grid_distances, terrain_mtx, terrain_pos_dict, burrito_state, **action_kwargs)
            is_valid = action.valid_on_step_start(agent, terrain_mtx, burrito_state)
            valid_mask.append(int(is_valid))
        return np.array(valid_mask, dtype=np.float32)



class BurritoPlanner:
    """
    Handle high-level action input of agents
    Generate joint one-step low level actions from high-level actions
    Track agent high-level action executing status
    Interface between agents' high-level action outputs from policies and environment step function
    """
    _gridworld_cache = {}
    def __init__(self, layout, dummy_env_state, player_types, restrict_capability = True):
        """
        Initialize MDP and other properties
        """
        self.mdp = self._get_cached_gridworld(layout)
        self.terrain_mtx = np.array(self.mdp.mdp_params["terrain"])
        self.grid_distances = self._compute_grid_distances(self.terrain_mtx)
        self.dim = self.terrain_mtx.shape
        env_mask = (self.terrain_mtx != ' ') & ~np.char.isdigit(self.terrain_mtx) # everything that isn't a player spawn location or floor should be an obstacle
        self.obstacles = map(tuple,np.argwhere(env_mask))
        self.env_state = dummy_env_state
        self.joint_starts = dummy_env_state.players_pos_and_or # this we might wanna move
        self.players = player_types
        self.player_indices = [i for i in range(len(dummy_env_state.players))]

        self.agents = [idx for label, idx in zip(self.players,self.player_indices) if label=="A"]
        self.humans = [idx for label, idx in zip(self.players,self.player_indices) if label=="H"]
        self.high_level_action_priority = self.agents.copy()
        np.random.shuffle(self.high_level_action_priority)
        self.restrict_capability = restrict_capability
        self.planner = MAPFWrapper(self.dim, self.obstacles, self.agents, restrict_capability=restrict_capability) # this self.agents should NOT include human indices
        self.actions = [None for _ in range(len(self.agents))]
        self.high_level_action_indices = [None for _ in range(len(self.player_indices))]

    @staticmethod
    def _get_cached_gridworld(layout):
        """
        Returns a cached burritoGridworld instance for the given layout.
        If it doesn't exist, creates and caches it. Prevents burritoGridworld from being instantiated every instance.
        """
        if layout not in BurritoPlanner._gridworld_cache:
            BurritoPlanner._gridworld_cache[layout] = BurritoGridworld.from_layout_name(layout)
        return BurritoPlanner._gridworld_cache[layout]
    
    @staticmethod
    def _compute_grid_distances(terrain_mtx):
        """
        Given a 2D numpy array `terrain_mtx` where:
        - ' ' grids are vacant
        - any other token is not vacant
        returns a dictionary where:
        distances[(r1, c1)][(r2, c2)] = minimum number of steps
        to get from (r1, c1) to (r2, c2) through only vacant grids.
        """

        # Dimensions of the grid
        if 'distance' in BurritoPlanner._gridworld_cache:
            return BurritoPlanner._gridworld_cache['distance']
        
        rows, cols = terrain_mtx.shape
        # 1) Collect the list of all vacant grids
        vacant_grids = []
        for r in range(rows):
            for c in range(cols):
                if terrain_mtx[r, c] == ' ':
                    vacant_grids.append((r, c))

        # Data structure to store distances
        # A nested dict: distances[(r1, c1)][(r2, c2)] = dist
        distances = {}

        # Helper function to get valid neighbors
        def get_neighbors(r, c):
            """Return valid 4-neighbors (up,down,left,right) that are vacant."""
            for nr, nc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
                if 0 <= nr < rows and 0 <= nc < cols:
                    if terrain_mtx[nr, nc] == ' ':
                        yield (nr, nc)

        # 2) For each vacant grid, do a BFS to compute distance to all other vacant grids
        for start_grid in vacant_grids:
            r0, c0 = start_grid
            
            # Prepare a queue for BFS and a dict for visited grids + distance
            queue = deque()
            queue.append((r0, c0, 0))  # (row, col, distance)
            visited = {(r0, c0): 0}

            # Standard BFS
            while queue:
                r, c, dist = queue.popleft()

                # Explore neighbors
                for nr, nc in get_neighbors(r, c):
                    if (nr, nc) not in visited:
                        visited[(nr, nc)] = dist + 1
                        queue.append((nr, nc, dist + 1))

            # 3) Store the BFS distances in the nested dictionary
            distances[start_grid[::-1]] = {}
            for grid, d in visited.items():
                distances[start_grid[::-1]][grid[::-1]] = d

        return distances


    def reset(self):
        self.actions = [None for _ in range(len(self.agents))]
        self.high_level_action_indices = [None for _ in range(len(self.player_indices))]
        self.planner.reset()

    def goal_masking(self, agent_idx, agent, burrito_state:BurritoState):
        if self.actions[agent_idx].goal_pos in HighAction.INVALID_GOALS:
            backup_candidate = []
            backup_distance = []
            for goal_pos_or in HighAction.BACKUP_GOALS:
                if goal_pos_or[0] in burrito_state.player_positions or goal_pos_or[0] in self.target_goal_pos_list:
                    continue
                else:
                    backup_candidate.append(goal_pos_or)
                    backup_distance.append(self.grid_distances[goal_pos_or[0]][burrito_state.players[agent].position])
            backup_goal_idx = backup_distance.index(min(backup_distance))
            backup_goal = backup_candidate[backup_goal_idx]
            ### in-place modification of action's goal
            # might cause difficulty for RL Policy to relate action idx and mdp transition
            self.actions[agent_idx].fallback_to(backup_goal[0], backup_goal[1])

    def create_action_object(self, agent_idx, agent, burrito_state:BurritoState, high_level_action):
        """
        Create a high-level action object when a high-level action command is received
        """
        if self.actions[agent_idx] is None:
            action_init = list(HighLevelActions)[high_level_action]
            action_cls = action_init.action_class
            action_kwargs = action_init.action_kwargs
            action = action_cls(agent, self.grid_distances, self.terrain_mtx, self.mdp.terrain_pos_dict, burrito_state, **action_kwargs)
            self.actions[agent_idx] = action
            self.high_level_action_indices[agent] = high_level_action

        # validate action
        # print("validate action")
        valid = self.actions[agent_idx].valid_on_step_start(agent, self.terrain_mtx, burrito_state)
        if not valid: #or self.actions[agent_idx].goal_pos in self.target_goal_pos_list:
            action_init = HighLevelActions.STAY
            action = action_init.action_class(agent, self.grid_distances, self.terrain_mtx, self.mdp.terrain_pos_dict, burrito_state, **action_init.action_kwargs)
            self.actions[agent_idx] = action
            self.high_level_action_indices[agent] = HighLevelActions.STAY.action_index
        # self.goal_masking(agent_idx, agent, burrito_state)

        # self.target_goal_pos_list.add(self.actions[agent_idx].goal_pos)

    def goal_conflict_resolve(self, burrito_state:BurritoState):
        ## Highest priority for agents that wants to STAY
        goal_poses = [None for agent in self.agents]
        goal_ores = [None for agent in self.agents]
        for agent_idx, agent in enumerate(self.agents):
            if self.high_level_action_indices[agent] == HighLevelActions.STAY.action_index:
                goal_poses[agent_idx], goal_ores[agent_idx] = self.actions[agent_idx].goal
        
        ## randomized priority for rest of the agents to determine goal
        chances = np.random.rand()
        if chances < 0.1:
            np.random.shuffle(self.high_level_action_priority)
        for agent in self.high_level_action_priority:
            agent_idx = self.agents.index(agent)
            current_goal_pos, _ = self.actions[agent_idx].goal
            if current_goal_pos in goal_poses:
                action_init = HighLevelActions.STAY
                action = action_init.action_class(agent, self.grid_distances, self.terrain_mtx, self.mdp.terrain_pos_dict, burrito_state, **action_init.action_kwargs)
                self.actions[agent_idx] = action
                self.high_level_action_indices[agent] = HighLevelActions.STAY.action_index
            goal_poses[agent_idx], goal_ores[agent_idx] = self.actions[agent_idx].goal
        return goal_poses, goal_ores


    def generate_goal_for_Action(self, burrito_state:BurritoState, high_level_action_index):
        """
        Find a proper goal for the corresponding NavAction.
        The logic now is finding the closest goal.
        :return: Position and orientation which will be used to create action object
        """
        goal = []
        self.target_goal_pos_list = set()
        for agent_idx, agent in enumerate(self.agents):
            self.create_action_object(agent_idx, agent, burrito_state, high_level_action_index[agent])
        
        goal_poses, goal_ores = self.goal_conflict_resolve(burrito_state)
        for agent_idx, agent in enumerate(self.agents):
            goal.append([goal_poses[agent_idx][::-1], goal_ores[agent_idx]])
        return goal
    

    def joint_path_planner(self, burrito_state, goal_pos_or):
        """
        Generate low-level actions from current state and high-level actions.
        Update action status.
        :return: low-level actions and high-level action status
        """
        # InterAction (including stay) should be executed anyway, not influenced by the path planner
        agent_poses = []
        for agent in self.agents:
            pos, ori = burrito_state.players_pos_and_or[agent]
            agent_poses.append([pos[::-1], ori])
        # TODO: should be human next positions
        human_positions = []
        for human in self.humans:
            pos, ori = burrito_state.players_pos_and_or[human]
            human_positions.append(pos[::-1])
        joint_action_array, solution_found = self.planner.generate_trajectories(agent_poses, goal_pos_or, self.high_level_action_indices)
        
        return joint_action_array, solution_found

    def define_plan(self, burrito_state: BurritoState, high_level_action_index):
        goal = self.generate_goal_for_Action(burrito_state,high_level_action_index)
        # print(high_level_action_index,"HIGH LEVEL INDEX ACTION")     
        joint_action_array, solution_found = self.joint_path_planner(burrito_state, goal)
        action_done = []
        if not solution_found:
            for agent_idx, agent in enumerate(self.agents):
                self.actions[agent_idx] = None
                action_done.append({"status":True,"prev_action":self.high_level_action_indices[agent]})
        else:
            for agent_idx, agent in enumerate(self.agents):
                if self.actions[agent_idx].nav_done and self.actions[agent_idx].to_interact and joint_action_array[agent_idx] == Action.STAY:
                    joint_action_array[agent_idx] = Action.INTERACT
                if joint_action_array[agent_idx] != Action.INTERACT:
                    joint_action_array[agent_idx] = tuple(joint_action_array[agent_idx])
                self.actions[agent_idx].update_on_step_end(joint_action_array[agent_idx])
                agent_action_done = self.actions[agent_idx].done
                action_done.append({"status": agent_action_done, "prev_action": self.high_level_action_indices[agent]})
                if agent_action_done:
                    self.actions[agent_idx] = None

        for human_idx, human in enumerate(self.humans[::-1]):
            self.high_level_action_indices[human] = high_level_action_index[human]
            joint_action_array.insert(human,Action.INDEX_TO_ACTION[high_level_action_index[human]])
            action_done.insert(human, {"status": True, "prev_action": self.high_level_action_indices[human]})

        # print(action_done)
        # print('\n\n\n')
        return joint_action_array, action_done, solution_found


from burrito.planners.mapf.cbs import CBSSolver
from burrito.planners.pypibt.pibt import PIBT
class MAPFWrapper:
    """
    Wraps an Overcooked env in a format for multi-agent pathfinding algorithms
    Gets agent positions, obstacles, start/goal states
    Generates multi-step plan for each controlled agent in the env
    """
    
    def __init__(self, dim, obstacles, agents, restrict_capability):
        self.dim = dim # tuple of layout dim
        self.obstacles = obstacles # list of tuples of untraversable grid tiles
        self.agents = agents
        self.plan_alg = "PIBT"
        self.current_goals = [(None,None) for _ in self.agents]
        self.trajectories = [[(None)] for _ in self.agents]
        self.current_pos_history = []
        self.current_goal_history = []
        self.trajectory_history = []
        self._init_planning_algorithm(self.plan_alg)
        self.restrict_capability = restrict_capability
    
    def _init_planning_algorithm(self,planner):
        if planner == "CBS":
            self.planner = CBSSolver
            self.initial_map = self._wrap_env_CBS(self.obstacles)
        elif planner == "PIBT":
            self.planner = PIBT
            self.initial_map = self._wrap_env(self.obstacles)
            self.priorities = [i for i, ag in enumerate(self.agents)]
            np.random.shuffle(self.priorities)
    
    def setup_planning(self, curr_pos_or, goal_pos_or):
        """
        Initializes the agent state for planning.
        :param curr_pos_or: List of tuples, each (position, orientation) for each agent.
        :param goal_pos_or: List of tuples, each (goal position, desired orientation) for each agent.
        """

        self.agent_state = {
            "curr_pos_or": curr_pos_or,
            "goal_pos_or": goal_pos_or,
        }
        agent_starts = []
        agent_goals = []
        assert len(curr_pos_or) == len(goal_pos_or)
        for ag_idx in range(len(curr_pos_or)):
            agent_starts.append(tuple(curr_pos_or[ag_idx][0]))
            agent_goals.append(tuple(goal_pos_or[ag_idx][0]))
        return agent_starts,agent_goals

    def reset(self):
        self.current_goals = [(None,None) for _ in self.agents]
        self.trajectories = [[(None)] for _ in self.agents]

    def _wrap_env_CBS(self, obs):
        map = np.zeros(self.dim,dtype=bool)
        for row,col in obs:
            map[row,col] = True 
        return map

    def _wrap_env(self, obs):
        map = np.ones(self.dim,dtype=bool)
        for row,col in obs:
            map[row,col] = False
        return map

    def _map_to_action(self,solution,high_level_actions):
        traj_actions = []
        for idx,agent_traj in enumerate(solution):
            if len(agent_traj) > 1:
                transition = [agent_traj[1][0]-agent_traj[0][0],agent_traj[1][1]-agent_traj[0][1]] # [x1-x0,y1-y0]
                if transition == [1,0]:
                    action = Direction.SOUTH
                elif transition == [-1,0]:
                    action = Direction.NORTH
                elif transition == [0,1]:
                    action = Direction.EAST
                elif transition == [0,-1]:
                    action = Direction.WEST
                else:
                    try:
                        assert transition == [0,0]
                    except AssertionError:
                        print("transition not [0,0]: ", transition)
                        print(self.current_pos, self.current_goals)
                        print(self.trajectories)
                        print("history of this replanned trajectory:")
                        for i in range(len(self.trajectory_history)):
                            print("current pos and goal:")
                            print(self.current_pos_history[i], self.current_goal_history[i])
                            print("current trajectory:")
                            print(self.trajectory_history[i])
                            print('\n')
                        raise
                    action = Action.STAY
            else: # [0,0]
                if self.agent_state["curr_pos_or"][idx][1] == self.agent_state["goal_pos_or"][idx][1]:
                    action = Action.STAY
                else:
                    if not high_level_actions[self.agents[idx]] == HighLevelActions.STAY.action_index:
                        action = self.agent_state["goal_pos_or"][idx][1] #adjust to the desired orientation
            # print("COMPUTED ACTIONS", action)        
            if action != Action.STAY and action != Action.INTERACT:
                if self.restrict_capability:
                    change = np.random.rand()
                    if change <= 0.65:
                        action = Action.STAY
            traj_actions.append(action)
        return traj_actions

    def _check_replan_needed(self,goal_pos_or):
        replan = goal_pos_or != self.current_goals
        if replan:
            self.current_goals = goal_pos_or
        return replan

    def generate_trajectories(self, curr_pos_or, goal_pos_or, high_level_action_indices):
        """
        Calls centralized path planner to generate trajectories for each agent
        :return: array of actions for each agent in the env
        """
        agent_starts, agent_goals = self.setup_planning(curr_pos_or,goal_pos_or)
        plan = self._check_replan_needed(goal_pos_or)
        self.current_pos = agent_starts
        # checks if agent got blocked the last timestep
        for i in range(len(self.trajectories)):
            if self.trajectories[i][0] != agent_starts[i]:
                if self.restrict_capability:
                    plan = True
                else:
                    self.trajectories[i].insert(0,agent_starts[i])
        if plan:
            self.trajectory_history.clear()
            self.current_goal_history.clear()
            self.current_pos_history.clear()
            wrapped_env = self.initial_map # this is "greedy" agent planning, will ignore humans
            #wrapped_env = self.initial_map + self._wrap_env(addtnl_obs)
            if self.plan_alg == "CBS":
                cbs = self.planner(wrapped_env,agent_starts,agent_goals)
                solution = cbs.find_solution()
            elif self.plan_alg == "PIBT":
                pibt = self.planner(wrapped_env, agent_starts, agent_goals)
                if np.random.rand() < 0.1:
                    np.random.shuffle(self.priorities)
                solution = pibt.run(max_timestep=1000, priorities=self.priorities)
            if not solution:
                # print("PATH COULD NOT BE SOLVED")
                # print("CURRENT POSITION",curr_pos_or)
                # print("TARGET POSITION",goal_pos_or)
                return [Action.STAY for _ in self.agents], False # just default to STAY, False indicates no successful plan found
            if self.plan_alg == "PIBT":
                solution = [list(pair) for pair in zip(*solution)]

            self.trajectories = deepcopy(solution)

            def remove_tail_duplicate(lst):
                last = lst[-1]
                idx = len(lst) - 1
                while idx > 0 and lst[idx - 1] == last:
                    idx -= 1
                del lst[idx+1:]
            for i in range(len(self.trajectories)):
                remove_tail_duplicate(self.trajectories[i])
            
            self.trajectory_history.append(deepcopy(self.trajectories))
        self.current_pos_history.append(agent_starts)
        self.current_goal_history.append(agent_goals)
        # breakpoint()
        # print(self.agent_state["curr_pos_or"], self.agent_state["goal_pos_or"])
        # print(agent_starts, agent_goals)
        # print("REPLAN", plan)
        # print("TRAJECTORY")
        # for trajectory in self.trajectories:
        #     print(trajectory)
        joint_action_array = self._map_to_action(self.trajectories, high_level_action_indices)
        for i in range(len(self.trajectories)):
            if len(self.trajectories[i]) > 1:
                self.trajectories[i].pop(0)
        self.trajectory_history.append(deepcopy(self.trajectories))
        # print("ACTIONS",joint_action_array)
        # print('\n\n\n')
        return joint_action_array, True # [agent1.Action, agent2.Action,...]
