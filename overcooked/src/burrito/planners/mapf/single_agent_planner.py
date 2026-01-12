import heapq
import itertools

def move(loc, dir):
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
    return loc[0] + directions[dir][0], loc[1] + directions[dir][1]


def move_joint_state(locs, dirs):

    new_locs = []
    for loc, dir in zip(locs, dirs):
        new_loc=(loc[0] + dir[0], loc[1] + dir[1])
        new_locs.append(new_loc)
    
    return new_locs

def generate_motions_recursive(num_agents,cur_agent):
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
    
    joint_state_motions = list(itertools.product(directions, repeat=num_agents))

    return joint_state_motions


def is_valid_motion(old_loc, new_loc):
    ##############################
    # Task 1.3/1.4: Check if a move from old_loc to new_loc is valid
    # Check if two agents are in the same location (vertex collision)
    for i in range(len(new_loc)):
        for j in range(i+1,len(new_loc)):
            if new_loc[i] == new_loc[j]:
                return False

    # Check edge collision
    for i in range(len(new_loc)):
        for j in range(i+1,len(new_loc)):
            if new_loc[i] == old_loc[j] and new_loc[j] == old_loc[i]:
                return False

    return True

def get_sum_of_cost(paths):
    rst = 0
    if paths is None:
        return -1
    for path in paths:
        rst += len(path) - 1
    return rst


def compute_heuristics(my_map, goal):
    # Use Dijkstra to build a shortest-path tree rooted at the goal location
    open_list = []
    closed_list = dict()
    root = {'loc': goal, 'cost': 0}
    heapq.heappush(open_list, (root['cost'], goal, root))
    closed_list[goal] = root
    while len(open_list) > 0:
        (cost, loc, curr) = heapq.heappop(open_list)
        for dir in range(4):
            child_loc = move(loc, dir)
            child_cost = cost + 1
            if child_loc[0] < 0 or child_loc[0] >= len(my_map) \
               or child_loc[1] < 0 or child_loc[1] >= len(my_map[0]):
               continue
            if my_map[child_loc[0]][child_loc[1]]:
                continue
            child = {'loc': child_loc, 'cost': child_cost}
            if child_loc in closed_list:
                existing_node = closed_list[child_loc]
                if existing_node['cost'] > child_cost:
                    closed_list[child_loc] = child
                    # open_list.delete((existing_node['cost'], existing_node['loc'], existing_node))
                    heapq.heappush(open_list, (child_cost, child_loc, child))
            else:
                closed_list[child_loc] = child
                heapq.heappush(open_list, (child_cost, child_loc, child))

    # build the heuristics table
    h_values = dict()
    for loc, node in closed_list.items():
        h_values[loc] = node['cost']
    return h_values


def build_constraint_table(constraints, agent, goal_loc):
    ##############################
    # Task 1.2/1.3/1.4: Return a table that constains the list of constraints of
    #               the given agent for each time step. The table can be used
    #               for a more efficient constraint violation check in the 
    #               is_constrained function.
    earliest_goal_timestep=0
    
    constraint_table={"vertex": {}, "edge": {}, "goal": {}}
    for constraint in constraints:
        if constraint["agent"] is None:
            constraint_table["goal"][constraint["loc"][0]]=constraint["timestep"]
        
        if agent!=constraint["agent"]:
            continue
        loc=constraint["loc"]
        timestep=constraint["timestep"]
        if len(loc)==1:
            if timestep not in constraint_table["vertex"]:
                constraint_table["vertex"][timestep]=dict()
            constraint_table["vertex"][timestep][loc[0]]=None
            if loc[0]==goal_loc:
                earliest_goal_timestep=max(timestep+1,earliest_goal_timestep)
        elif len(loc)==2:
            # Note this timestep is the timestep for loc[1].
            if timestep not in constraint_table["edge"]:
                constraint_table["edge"][timestep]=dict()
            constraint_table["edge"][timestep][loc[0]]=loc[1]
        else:
            raise NotImplementedError
        
    return constraint_table,earliest_goal_timestep


def get_location(path, time):
    if time < 0:
        return path[0]
    elif time < len(path):
        return path[time]
    else:
        return path[-1]  # wait at the goal location


def get_path(goal_node):
    path = []
    curr = goal_node
    while curr is not None:
        path.append(curr['loc'])
        curr = curr['parent']
    path.reverse()
    return path


def is_constrained(curr_loc, next_loc, next_time, constraint_table):
    ##############################
    # Task 1.2/1.3/1.4: Check if a move from curr_loc to next_loc at time step next_time violates
    #               any given constraint. For efficiency the constraints are indexed in a constraint_table
    #               by time step, see build_constraint_table.

    # vertex constraint
    if next_time in constraint_table["vertex"] and next_loc in constraint_table["vertex"][next_time]:
        return True

    # edge constraint
    if next_time in constraint_table["edge"] and curr_loc in constraint_table["edge"][next_time] and next_loc==constraint_table["edge"][next_time][curr_loc]:
        return True
    
    if next_loc in constraint_table["goal"] and next_time>constraint_table["goal"][next_loc]:
        return True
    
    return False


def push_node(open_list, node):
    heapq.heappush(open_list, (node['g_val'] + node['h_val'], node['h_val'], node['loc'], node))


def pop_node(open_list):
    _, _, _, curr = heapq.heappop(open_list)
    return curr


def compare_nodes(n1, n2):
    """Return true is n1 is better than n2."""
    return n1['g_val'] + n1['h_val'] < n2['g_val'] + n2['h_val']

def in_map(map, loc):
    if loc[0] >= len(map) or loc[1] >= len(map[0]) or min(loc) < 0:
        return False
    else:
        return True

def all_in_map(map, locs):
    for loc in locs:
        if not in_map(map, loc):
            return False
    return True

def a_star(my_map, start_loc, goal_loc, h_values, agent, constraints, max_prev_timestep=float('inf')):
    """ my_map      - binary obstacle map
        start_loc   - start position
        goal_loc    - goal position
        agent       - the agent that is being re-planned
        constraints - constraints defining where robot should or cannot go at each timestep
    """

    ##############################
    # Task 1.2/1.3/1.4: Extend the A* search to search in the space-time domain
    #           rather than space domain, only.
    constraint_table,earliest_goal_timestep = build_constraint_table(constraints,agent,goal_loc)
    
    # print(constraint_table)

    open_list = []
    closed_list = dict()
    h_value = h_values[start_loc]
    root = {'loc': start_loc, "timestep":0, 'g_val': 0, 'h_val': h_value, 'parent': None}
    push_node(open_list, root)
    closed_list[(root['loc'])] = root
    while len(open_list) > 0:
        curr = pop_node(open_list)
        # if agent==1:
        #     print(curr['loc'],earliest_goal_timestep)
        #############################
        # Task 2.2: Adjust the goal test condition to handle goal constraints
        if curr['loc'] == goal_loc and curr['timestep'] >= earliest_goal_timestep:
            return get_path(curr)
        for dir in range(5):
            child_loc = move(curr['loc'], dir)
            if not in_map(my_map, child_loc) or my_map[child_loc[0]][child_loc[1]]:
                continue
            if is_constrained(curr['loc'], child_loc, curr['timestep']+1, constraint_table):
                continue
            child = {'loc': child_loc,
                     'timestep': curr['timestep'] + 1 if curr['timestep']<=max_prev_timestep else curr['timestep'], # after all prior agents have reached their goal, we fall back to the spatial A* search
                    'g_val': curr['g_val'] + 1,
                    'h_val': h_values[child_loc],
                    'parent': curr}
            if (child['loc'],child['timestep']) in closed_list:
                existing_node = closed_list[(child['loc'],child['timestep'])]
                if compare_nodes(child, existing_node):
                    closed_list[(child['loc'],child['timestep'])] = child
                    push_node(open_list, child)
            else:
                closed_list[(child['loc'],child['timestep'])] = child
                push_node(open_list, child)

    return None  # Failed to find solutions


def joint_state_a_star(my_map, starts, goals, h_values, num_agents):
    """ my_map      - binary obstacle map
        start_loc   - start positions
        goal_loc    - goal positions
        num_agent   - total number of agents in fleet
    """

    open_list = []
    closed_list = dict()
    earliest_goal_timestep = 0
    h_value = 0
     ##############################
    # Task 1.1: Iterate through starts and use list of h_values to calculate total h_value for root node
    #
    for i in range(num_agents):
        h_value += h_values[i][starts[i]]
    
    root = {'loc': starts, 'g_val': 0, 'h_val': h_value, 'parent': None }
    push_node(open_list, root)
    closed_list[tuple(root['loc'])] = root

     ##############################
    # Task 1.1:  Generate set of all possible motions in joint state space
    #
    directions = generate_motions_recursive(num_agents,0)
    while len(open_list) > 0:
        curr = pop_node(open_list)
        # print(curr['loc'])
        
        if curr['loc'] == goals:
            return get_path(curr)

        for dirs in directions:
            
            ##############################
            # Task 1.1:  Update position of each agent
            #
            child_loc = move_joint_state(curr['loc'], dirs)
            # print("child_loc",child_loc)
            
            if not all_in_map(my_map, child_loc):
                continue
             ##############################
            # Task 1.1:  Check if any agent is in an obstacle
            #
            valid_move = True
            for loc in child_loc:
                if my_map[loc[0]][loc[1]]:
                    valid_move = False
                    break
            
            if not valid_move:
                continue

             ##############################
            # Task 1.1:   check for collisions
            #
            if not is_valid_motion(curr['loc'],child_loc):
                continue
            
             ##############################
            # Task 1.1:  Calculate heuristic value
            #
            h_value = 0
            for i in range(num_agents):
                h_value += h_values[i][child_loc[i]]

            # Create child node
            child = {'loc': child_loc,
                    'g_val': curr['g_val'] + num_agents,
                    'h_val': h_value,
                    'parent': curr}
            if tuple(child['loc']) in closed_list:
                existing_node = closed_list[tuple(child['loc'])]
                if compare_nodes(child, existing_node):
                    closed_list[tuple(child['loc'])] = child
                    push_node(open_list, child)
            else:
                closed_list[tuple(child['loc'])] = child
                push_node(open_list, child)

    return None  # Failed to find solutions
