import time as timer
import heapq
import random
from burrito.planners.mapf.single_agent_planner import compute_heuristics, a_star, get_location, get_sum_of_cost
from copy import deepcopy

def detect_first_collision_for_path_pair(path1, path2):
    ##############################
    # Task 2.1: Return the first collision that occurs between two robot paths (or None if there is no collision)
    #           There are two types of collisions: vertex collision and edge collision.
    #           A vertex collision occurs if both robots occupy the same location at the same timestep
    #           An edge collision occurs if the robots swap their location at the same timestep.
    #           You should use "get_location(path, t)" to get the location of a robot at time t.

    for t in range(max(len(path1),len(path2))):
        loc1=get_location(path1,t)
        loc2=get_location(path2,t)
        if loc1==loc2:
            # vertex collision
            collision={
                "timestep": t,
                "loc": [loc1], 
            }
            return collision

        if t!=0:
            loc1_prev=get_location(path1,t-1)
            loc2_prev=get_location(path2,t-1)
            if loc1_prev==loc2 and loc2_prev==loc1:
                # edge collision
                collision={
                    "timestep": t,
                    "loc": [loc1_prev,loc1]
                }
                return collision

    return None


def detect_collisions_among_all_paths(paths):
    ##############################
    # Task 2.1: Return a list of first collisions between all robot pairs.
    #           A collision can be represented as dictionary that contains the id of the two robots, the vertex or edge
    #           causing the collision, and the timestep at which the collision occurred.
    #           You should use your detect_collision function to find a collision between two robots.

    collisions=[]
    
    for i in range(len(paths)):
        for j in range(i+1,len(paths)):
            path1=paths[i]
            path2=paths[j]
            collision=detect_first_collision_for_path_pair(path1,path2)
            if collision is not None:
                collision["a1"]=i
                collision["a2"]=j
                collisions.append(collision)
    
    return collisions


def standard_splitting(collision):
    ##############################
    # Task 2.2: Return a list of (two) constraints to resolve the given collision
    #           Vertex collision: the first constraint prevents the first agent to be at the specified location at the
    #                            specified timestep, and the second constraint prevents the second agent to be at the
    #                            specified location at the specified timestep.
    #           Edge collision: the first constraint prevents the first agent to traverse the specified edge at the
    #                          specified timestep, and the second constraint prevents the second agent to traverse the
    #                          specified edge at the specified timestep

    loc=collision["loc"]
    if len(loc)==1:
        # vertex collision
        constraint1={
            "agent": collision["a1"],
            "loc": [loc[0]],
            "timestep": collision["timestep"]
        }
        constraint2={
            "agent": collision["a2"],
            "loc": [loc[0]],
            "timestep": collision["timestep"]
        }
        return [constraint1,constraint2]
    elif len(loc)==2:
        # edge collision
        constraint1={
            "agent": collision["a1"],
            "loc": [loc[0],loc[1]],
            "timestep": collision["timestep"]
        }
        constraint2={
            "agent": collision["a2"],
            "loc": [loc[1],loc[0]],
            "timestep": collision["timestep"]
        }
        return [constraint1,constraint2]
    else:
        raise NotImplementedError

class CBSSolver(object):
    """The high-level search of CBS."""

    def __init__(self, my_map, starts, goals):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0

        self.open_list = []

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def push_node(self, node):
        heapq.heappush(self.open_list, (node['cost'], len(node['collisions']), self.num_of_generated, node))
        # print("Generate node {}".format(self.num_of_generated))
        self.num_of_generated += 1

    def pop_node(self):
        _, _, id, node = heapq.heappop(self.open_list)
        # print("Expand node {}".format(id))
        self.num_of_expanded += 1
        return node

    def find_solution(self):
        """ Finds paths for all agents from their start locations to their goal locations

        """

        self.start_time = timer.time()
        self.ctr=0

        # Generate the root node
        # constraints   - list of constraints
        # paths         - list of paths, one for each agent
        #               [[(x11, y11), (x12, y12), ...], [(x21, y21), (x22, y22), ...], ...]
        # collisions     - list of collisions in paths
        root = {'cost': 0,
                'constraints': [],
                'paths': [],
                'collisions': []}
        for i in range(self.num_of_agents):  # Find initial path for each agent
            self.ctr+=1
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                          i, root['constraints'])
            if path is None:
                return None
            root['paths'].append(path)

        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions_among_all_paths(root['paths'])
        self.push_node(root)

        # Task 2.1: Testing
        #print(root['collisions'])

        # Task 2.2: Testing
        #for collision in root['collisions']:
            #print(standard_splitting(collision))

        ##############################
        # Task 2.3: High-Level Search
        #           Repeat the following as long as the open list is not empty:
        #             1. Get the next node from the open list (you can use self.pop_node()
        #             2. If this node has no collision, return solution
        #             3. Otherwise, choose the first collision and convert to a list of constraints (using your
        #                standard_splitting function). Add a new child node to your open list for each constraint
        #           Ensure to create a copy of any objects that your child nodes might inherit

        # These are just to print debug output - can be modified once you implement the high-level search
        
        while len(self.open_list)>0:
            node=self.pop_node()
            if self.ctr>2000: # 1000: # 5000 ~= 1.3s, 10000 ~= 6s, 50000 ~= 22s, 100000 ~= 43s
                self.print_results(node)
                print(self.ctr)
                break
            if timer.time()-self.start_time>300:
                print("Time out")
                print(len(node["collisions"]))
                self.print_results(node)
                break
            if len(node["collisions"])==0:
                self.print_results(node)
                #print("CBS result found: ", node["paths"])
                return node["paths"]
            # the first collision to branch
            idx=random.randint(0,len(node["collisions"])-1)
            collision=node["collisions"][idx]
            constraints=standard_splitting(collision)
            for constraint in constraints:
                new_node= {
                    'cost': None,
                    'constraints': [],
                    'paths': [],
                    'collisions': []
                }
                new_node["constraints"]=deepcopy(node["constraints"])
                new_node["constraints"].append(constraint)
                new_node["paths"]=deepcopy(node["paths"])
                agent=constraint["agent"]
                self.ctr+=1
                path=a_star(
                    self.my_map,
                    self.starts[agent],
                    self.goals[agent],
                    self.heuristics[agent],
                    agent,
                    new_node['constraints']
                )
                if path is not None:
                    new_node["paths"][agent]=path
                    new_node["collisions"]=detect_collisions_among_all_paths(new_node["paths"])
                    new_node["cost"]=get_sum_of_cost(new_node["paths"])
                    self.push_node(new_node)
        
        print("CBS search results: None")
        return None


    def print_results(self, node):
        return
#        print("\n Found a solution! \n")
#        CPU_time = timer.time() - self.start_time
#        print("CPU time (s):    {:.2f}".format(CPU_time))
#        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
#        print("Expanded nodes:  {}".format(self.num_of_expanded))
#        print("Generated nodes: {}".format(self.num_of_generated))
