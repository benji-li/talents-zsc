import json
import os
import numpy as np

def map_event_to_planner_index(event_name):
    event_to_index = {
        # GO_TO_PAN_AND_FRY_INGREDIENT (0)
        "chopped_steak_cooking": 0,
        "fried_mushroom_cooking": 0,
        "steak_cooking": 0,
        
        # GO_TO_POT_AND_BOIL_RICE (1)
        "potting_rice": 1,
        
        # GO_TO_CHOP_BOARD_AND_CHOP_INGREDIENT (2)
        "meat_chopping": 2,
        "onion_chopping": 2,
        
        # GO_TO_PUT_OUT_FIRE (3)
        "extinguish_fire": 3,
        
        # STAY (4) - No specific events
        
        # GO_TO_SINK_TO_WASH_PLATE (5)
        "plate_rinsing": 5,
        
        # GO_TO_TRASH_AND_THROW (6)
        "object_in_trash": 6,
        
        # GO_TO_SERVE_DISH (7)
        "dish_delivery": 7,
        
        # GRAB_MEAT (8)
        "meat_pickup": 8,
        
        # GRAB_MUSHROOM (9)
        "mushroom_pickup": 9,
        
        # GRAB_TORTILLA (10)
        "tortilla_pickup": 10,
        
        # GRAB_RICE (11)
        "rice_pickup": 11,
        
        # GRAB_DIRTY_PLATE (12)
        "dirty_plate_pickup": 12,
        
        # GRAB_CLEAN_PLATE (13)
        "clean_plate_pickup": 13,
        
        # GRAB_FIRE_EXT (14)
        "fire_ext_pickup": 14,
        
        # GRAB_CHOPPED_MUSHROOM (15)
        "chopped_mushroom_pickup": 15,
        
        # GRAB_CHOPPED_MEAT (16)
        "chopped_meat_pickup": 16,
        
        # GRAB_ONE_INGREDIENT_PLATE (17)
        "boiled_rice-plate_pickup": 17,
        "tortilla-plate_pickup": 17,
        "chopped_steak-plate_pickup": 17,
        "fried_mushroom-plate_pickup": 17,
        
        # GRAB_TWO_INTREDIENT_PLATE (18)
        "chopped_steak-boiled_rice-plate_pickup": 18,
        "fried_mushroom-boiled_rice-plate_pickup": 18,
        "boiled_rice-tortilla-plate_pickup": 18,
        "fried_mushroom-tortilla-plate_pickup": 18,
        "chopped_steak-tortilla-plate_pickup": 18,
        
        # GRAB_MURHSOOM_BURRITO_PLATE (19)
        "mushroom_burrito_pickup": 19,
        
        # GRAB_STEAK_BURRITO_PLATE (20)
        "steak_burrito_pickup": 20,
        
        # GET_MURHSOOM_FROM_GRILLER (21)
        "fried_mushroom-plate_grill_pickup": 21,
        "fried_mushroom-tortilla-plate_grill_pickup": 21,
        "fried_mushroom-boiled_rice-plate_grill_pickup": 21,
        "mushroom_burrito_grill_pickup": 21,
        
        # GET_STEAK_FROM_GRILLER (22)
        "chopped_steak-plate_grill_pickup": 22,
        "chopped_steak-tortilla-plate_grill_pickup": 22,
        "chopped_steak-boiled_rice-plate_grill_pickup": 22,
        "steak_burrito_grill_pickup": 22,
        
        # GET_BOILED_RICE (23)
        "boiled_rice-plate_pot_pickup": 23,
        "chopped_steak-boiled_rice-plate_pot_pickup": 23,
        "fried_mushroom-boiled_rice-plate_pot_pickup": 23,
        "boiled_rice-tortilla-plate_pot_pickup": 23,
        "mushroom_burrito_pot_pickup": 23,
        "steak_burrito_pot_pickup": 23,
        
        # GRAB_CHARCOAL (24)
        "charcoal_pickup": 24,
        
        # PASS_OBJECT/PUT_DOWN_OBJECT (25/26)
        "mushroom_drop": 25,
        "chopped_mushroom_drop": 25,
        "rice_drop": 25,
        "boiled_rice-plate_drop": 25,
        "tortilla_drop": 25,
        "tortilla-plate_drop": 25,
        "fire_ext_drop": 25,
        "charcoal_drop": 25,
        "chopped_steak-plate_drop": 25,
        "meat_drop": 25,
        "chopped_meat_drop": 25,
        "chopped_steak-boiled_rice-plate_drop": 25,
        "fried_mushroom-boiled_rice-plate_drop": 25,
        "boiled_rice-tortilla-plate_drop": 25,
        "fried_mushroom-plate_drop": 25,
        "fried_mushroom-tortilla-plate_drop": 25,
        "chopped_steak-tortilla-plate_drop": 25,
        "chopped_steak_drop": 25,
        "dish_drop": 25,
        "steak_drop": 25,
        "mushroom_burrito_drop": 25,
        "steak_burrito_drop": 25,
        "dirty_plate_drop": 25,
    }
    
    return event_to_index.get(event_name, -1)

def parse_layout(grid_str):
    """Parse the layout grid to find all station positions"""
    lines = [line.strip() for line in grid_str.strip().split('\n') if line.strip()]
    
    grills = []
    pots = []
    trash = []
    serving = []
    chopboards = []
    sinks = []
    
    for y, line in enumerate(lines):
        for x, char in enumerate(line):
            if char == 'G':
                grills.append((x, y))
            elif char == 'P':
                pots.append((x, y))
            elif char == 'U': 
                trash.append((x, y))
            elif char == 'S': 
                serving.append((x, y))
            elif char == 'B':
                chopboards.append((x, y))
            elif char == 'W':
                sinks.append((x, y))
    
    return grills, pots, trash, serving, chopboards, sinks

def is_adjacent(pos1, pos2):
    """Check if two positions are adjacent (excluding diagonals)"""
    dx = abs(pos1[0] - pos2[0])
    dy = abs(pos1[1] - pos2[1])
    return (dx == 0 and dy <=1) or (dx <= 1 and dy == 0)

def detect_events(prev_state, curr_state, prev_actions, grills, pots, trash, serving, chopboards, sinks):
    """Detect events between two states for each player"""
    events = [None, None]  # Events for player 0 and 1
    
    for player_idx in range(2):
        # Only check for events if player performed an interact action
        if prev_actions[player_idx] != "interact":
            continue
        prev_player = prev_state['players'][player_idx]
        curr_player = curr_state['players'][player_idx]
        
        # Get held object name (handle both dict and string formats)
        prev_held = prev_player['held_object']
        curr_held = curr_player['held_object']
        
        prev_obj_name = prev_held['name'] if isinstance(prev_held, dict) else prev_held
        curr_obj_name = curr_held['name'] if isinstance(curr_held, dict) else curr_held
        #print(f"prev_obj_name: {prev_obj_name}, curr_obj_name: {curr_obj_name}")
        
        if prev_obj_name == curr_obj_name:
            # No change in held object, skip event detection
            continue
        
        # Check for pickup events
        if curr_obj_name is not None:
            player_pos = tuple(curr_player['position'])
            # Check if pickup is from grill or pot
            is_grill_pickup = any(is_adjacent(player_pos, grill_pos) for grill_pos in grills)
            is_pot_pickup = any(is_adjacent(player_pos, pot_pos) for pot_pos in pots)

            if prev_obj_name is None:
                event_name = f"{curr_obj_name}_pickup"
            else:
                # Format event name based on pickup location
                if is_grill_pickup and ('chopped_steak' in curr_obj_name or 'fried_mushroom' in curr_obj_name or 
                                        'steak_burrito' in curr_obj_name or 'mushroom_burrito' in curr_obj_name):
                    event_name = f"{curr_obj_name}_grill_pickup"
                elif is_pot_pickup and ('boiled_rice' in curr_obj_name or 'burrito' in curr_obj_name):
                    event_name = f"{curr_obj_name}_pot_pickup"
                else:
                    event_name = f"{curr_obj_name}_pickup"
            
            events[player_idx] = event_name
            
        # Check for drop events
        elif prev_obj_name is not None and curr_obj_name is None:
            player_pos = tuple(curr_player['position'])
            
            # Check drop location
            is_grill_drop = any(is_adjacent(player_pos, grill_pos) for grill_pos in grills)
            is_pot_drop = any(is_adjacent(player_pos, pot_pos) for pot_pos in pots)
            is_trash_drop = any(is_adjacent(player_pos, trash_pos) for trash_pos in trash)
            is_serving_drop = any(is_adjacent(player_pos, serve_pos) for serve_pos in serving)
            is_chop_drop = any(is_adjacent(player_pos, chop_pos) for chop_pos in chopboards)
            is_sink_drop = any(is_adjacent(player_pos, sink_pos) for sink_pos in sinks)
            
            # Determine specific drop event
            if is_grill_drop and 'plate' not in prev_obj_name and ('meat' in prev_obj_name or 'chopped_steak' in prev_obj_name):
                event_name = "chopped_steak_cooking" 
            elif is_grill_drop and 'mushroom' in prev_obj_name:
                event_name = "fried_mushroom_cooking"
            elif is_pot_drop and 'rice' in prev_obj_name:
                event_name = "potting_rice"
            elif is_pot_drop and 'plate' in prev_obj_name and 'rice' in prev_obj_name:
                # Dropping a plate with rice at pot
                event_name = "potting_rice"
            elif is_chop_drop and prev_obj_name == 'meat':
                event_name = "meat_chopping"
            elif is_chop_drop and prev_obj_name == 'onion':
                event_name = "onion_chopping"
            elif is_sink_drop and 'dirty_plate' in prev_obj_name:
                event_name = "plate_rinsing"
            elif is_trash_drop:
                event_name = "object_in_trash"
            elif is_serving_drop and ('dish' in prev_obj_name or 'burrito' in prev_obj_name):
                event_name = "dish_delivery"
            else:
                event_name = f"{prev_obj_name}_drop"
            
            events[player_idx] = event_name
        elif prev_obj_name is None and curr_obj_name is None:
            # no object held, could be chopping ingredients, washing dishes
            player_pos = tuple(curr_player['position'])
            is_chop = any(is_adjacent(player_pos, chop_pos) for chop_pos in chopboards)
            is_sink = any(is_adjacent(player_pos, sink_pos) for sink_pos in sinks)
            if is_chop:
                event_name = "meat_chopping" # doesn't matter if meat or onion, same mapping
            elif is_sink:
                event_name = "plate_rinsing" 

        #print("events", events)
    
    return events

def process_episode(episode_data, layout_name):
    """Process episode and return only timestep and joint action pairs"""
    # Maps dictionary
    maps = {
        "open":"""XXBBXDSSXX
                    X        X
                    X  1     W
                    X  RZMT  X
                    U     2  X
                    X        X
                    XXPPXXGGXX""", 
        "burrito": """XXPPXXXXXXXX
                Z    X     X
                T1   X2    W
                X    X     X
                X    XX    G
                D    XB    G
                S    XB    M
                S    XX    R
                X          U
                XXXXXXXXXXXX""",
        "forced_coordination":"""XXXXXXXX
                    X  GX  X
                    W1 GB 2X
                    X   X  S
                    Z  XX  S
                    M  XX  D
                    T   X  X
                    R  PB  U
                    U  PX  X
                    XXXXXXXX""",
        "ring": """XXXXDSSXXX
                    XZMR1 2XWX
                    T        U
                    X XXXXXX X
                    X XXXXXX X
                    X XXXXXX X
                    X        X
                    XBXBXGGPPX"""
    }
    
    # Get the grid for this layout
    grid = maps.get(layout_name)
    if not grid:
        raise ValueError(f"Unknown layout: {layout_name}")
    
    episodes = episode_data['episode']
    grills, pots, trash, serving, chopboards, sinks = parse_layout(grid)
    
    # First pass: detect all events
    all_events = []
    for i in range(1, len(episodes)):
        prev_state = episodes[i-1]['state']
        curr_state = episodes[i]['state']
        
        # Get previous actions - check if each player interacted
        prev_join_action = episodes[i-1]['join_action']
        curr_join_action = episodes[i]['join_action']
        
        # Determine if each player performed an interact action
        prev_action_types = []
        for player_action in curr_join_action:
            if player_action == "interact":
                prev_action_types.append("interact")
            else:
                prev_action_types.append(None)
        
        # Detect events for both players
        events = detect_events(prev_state, curr_state, prev_action_types, grills, pots, trash, serving, chopboards, sinks)
        all_events.append(events)
    
    # Second pass: map events to actions and fill in movement actions
    all_actions = []
    for i, events in enumerate(all_events):
        actions = [4, 4]  # Default to STAY (4)
        for player_idx, event in enumerate(events):
            if event:
                action_idx = map_event_to_planner_index(event)
                if action_idx != -1:
                    actions[player_idx] = action_idx
        all_actions.append(actions)
    
    # Third pass: backfill STAY actions with the next non-STAY action
    for player_idx in range(2):
        i = 0
        while i < len(all_actions):
            if all_actions[i][player_idx] == 4:  # STAY action
                # Find the next non-STAY action for this player
                j = i + 1
                while j < len(all_actions) and all_actions[j][player_idx] == 4:
                    j += 1
                
                # If found a non-STAY action, backfill all STAYs with it
                if j < len(all_actions):
                    next_action = all_actions[j][player_idx]
                    for k in range(i, j):
                        all_actions[k][player_idx] = next_action
                    i = j
                else:
                    # No more non-STAY actions, move to next
                    i += 1
            else:
                i += 1
    
    # Create simplified output with only timestep and joint actions
    simplified_data = []
    for i in range(1, len(episodes)):
        simplified_data.append({
            "timestep": i,
            "join_action": [all_actions[i-1][0], all_actions[i-1][1]]
        })
    
    return simplified_data

# def process_states(env_grid):
#     obs={}
#     for timestep, info in enumerate(env_grid):
#         state = np.array(info["state"])
#         egos = {}
#         for ag in range(2):
#             pos = state[:,:,:,8]
#             ego_layer = np.where(pos==ag, ag, 0)
#             ego_layer.reshape((state.shape[0], state.shape[1], 1))
#             state = np.concatenate((ego_layer, state[0]), axis=-1)
#             egos[ag] = state
#         obs[timestep] = egos
#     return obs
def process_states(env_grid):
    obs={}
    for timestep, info in env_grid.items():
        state = np.array(info["state"])
        egos = {}
        for ag in range(1,3):
            pos = state[:,:,:,8]
            #print("POS",pos)
            ego_layer = np.where(pos==ag, ag, 0)
            ego_layer = ego_layer.reshape((state.shape[1], state.shape[2], 1))
            state_with_ego = np.concatenate((ego_layer, state[0]), axis=-1)
            #print("EGOOO",state_with_ego[:,:,0])
            egos[ag] = state_with_ego.tolist()  # Convert to list for JSON serialization
        obs[timestep] = egos
    #print(state.shape)
    return obs

# Process all files in human_data directory
input_files = [f for f in os.listdir('./human_data') if f.startswith('room')]
game_data = [os.path.join('./human_data', room, f) for room in input_files 
             for f in os.listdir(os.path.join('./human_data', room)) 
             if not f.startswith('post') and not f.startswith('layout')]

# Process each file
for file_path in game_data:
    try:
        with open(os.path.join(file_path,'states.json'), 'r') as f:
            data = json.load(f)
        with open(os.path.join(file_path,'env_grid.json'), 'r') as f:
            env_grid = json.load(f)
        
        print(f"Processing: {file_path}")
        # Get layout name from the data
        layout_name = data.get('layout_name', 'open')  # default to 'open' if not specified
        
        # Process the episode
        processed_data = process_episode(data, layout_name)
        states = process_states(env_grid)
        # Combine actions and states into final output
        final_output = []
        for item in processed_data:
            timestep = item["timestep"]
            # Get state for this timestep (accounting for 0-based indexing in states)
            state_data = states.get(str(timestep - 1), None)
            #print("state_data", np.array(state_data[2])[:,:,0])
            final_output.append({
                "timestep": timestep,
                "join_action": item["join_action"],
                "states": state_data
            })
        
        parent_dir_name = os.path.basename(os.path.dirname(file_path))
        # Save to output file (same name with _processed suffix)
        output_path = './human_data_processed/' + os.path.basename(file_path) + '/'+parent_dir_name+'.json'
        print(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(final_output, f, indent=4)
        
        print(f"Processed: {file_path}")
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")