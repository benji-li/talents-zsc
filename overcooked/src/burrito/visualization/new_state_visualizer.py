import os
import json
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageSequenceClip
import numpy as np
import math

from burrito.mdp.burrito_mdp import BurritoState, BurritoGridworld

from burrito.encoding.env_enums import (
    counter_mapping,
    ingredient_mapping,
    kitchen_tool_mapping,
    held_item_mapping,
    recipe_mapping,
    orientation_mapping,
    action_mapping,
)

class StateVisualizer:

    def __init__(self, layout_name):
        self.COLORS = ["red", "blue", "yellow", "green", "purple"]

        self.layout_name = layout_name
        self.inv_counter = self._invert_dict(counter_mapping)
        self.inv_ingredient = self._invert_dict(ingredient_mapping)
        self.inv_tool = self._invert_dict(kitchen_tool_mapping)
        self.inv_held_item = self._invert_dict(held_item_mapping)
        self.inv_counter_item = self._invert_dict(held_item_mapping)
        self.inv_recipe = self._invert_dict(recipe_mapping)
        self.inv_orientation = self._invert_dict(orientation_mapping)
        self.inv_action = self._invert_dict(action_mapping)
        
        self.sprites = self.get_sprite_data()
        
    def _invert_dict(self,d):
        return {v: k for k, v in d.items()}

    def reconstruct_environment(self,current_env_grid):
        H, W, L = current_env_grid.shape  # e.g. H=10, W=12, L=10
        # We'll reconstruct:
        # terrain_mtx-like cells
        # dynamic_objects: a dict of {id: obj} as originally stored
        # players: a list of player states

        # For simplicity, assign each dynamic object a unique ID if needed
        dynamic_objects = {}
        players = []
        terrain_reconstructed = [[" " for _ in range(W)] for _ in range(H)]

        # We may need to track object counters
        obj_id_counter = 0

        for row in range(H):
            for col in range(W):
                c_val = current_env_grid[row, col, 0]
                i_val = current_env_grid[row, col, 1]
                d_val = current_env_grid[row, col, 2]
                t_val = current_env_grid[row, col, 3]
                timer_val = current_env_grid[row, col, 4]
                waiting_val = current_env_grid[row, col, 5]
                warning_val = current_env_grid[row, col, 6]
                burnt_val = current_env_grid[row, col, 7]
                player_pos_val = current_env_grid[row, col, 8]
                held_obj_val = current_env_grid[row, col, 9]
                orientation_val = current_env_grid[row, col, 10]

                # Reconstruct the cell from layers 0,1,2
                # Priority as done in handle_cell: try to find a char that maps to these values
                # The original cell was likely something that matched one of these mappings.
                cell_char = " "
                if t_val != 0:  # kitchen tool present
                    # Find the tool char
                    tool_char = self.inv_tool.get(t_val, None)
                    if tool_char is not None:
                        cell_char = tool_char
                elif i_val != 0:
                    # This might mean the cell was originally an ingredient symbol
                    ingr_char = self.inv_ingredient.get(i_val, None)
                    if ingr_char is not None:
                        cell_char = ingr_char
                elif c_val != 0:
                    # This might mean the cell was a counter-type symbol
                    ctr_char = self.inv_counter.get(c_val, None)
                    if ctr_char is not None:
                        cell_char = ctr_char

                terrain_reconstructed[row][col] = cell_char

                # Check for dynamic objects
                # handle_dynamic_objects sets counter_item_mapping in layer1 if dynamic obj present
                # If i_val doesn't map to an ingredient (already checked), it might be a counter item
                # ingr_char = inv_ingredient.get(i_val, None)
                counter_item_name = self.inv_counter_item.get(d_val, None)

                # print("dynamic obj value", d_val)
                # print("counter item name", counter_item_name)

                if counter_item_name:
                    # There is a dynamic object here
                    obj_id_counter += 1
                    obj = {}
                    obj["name"] = counter_item_name
                    obj["position"] = (col, row)  # TODO: Check if this is correct
                    obj["is_burnt"] = burnt_val == 1 or burnt_val == 2
                    obj["is_extinguished"] = burnt_val == 2

                    # Determine which timer was set (cooking, warning, dishes)
                    # Original logic:
                    # if _cooking_tick: layer3 = cooking_tick
                    # elif _warning_tick: layer4 = warning_tick
                    # elif _dishes_tick: layer3 = dishes_tick
                    # If warning_val > 0, then it's warning_tick
                    if warning_val > 0:
                        obj["warning_tick"] = warning_val
                        obj["is_warning"] = True
                        obj["is_cooking"] = False
                    if waiting_val > 0:
                        obj["waiting_tick"] = waiting_val
                        obj["is_waiting"] = True
                        obj["is_cooking"] = False
                    if timer_val:
                        # Could be cooking_tick or dishes_tick.
                        # Without more info, assume cooking_tick if the object is cookable,
                        # else dishes_tick. This is domain-specific logic.
                        # For demonstration, let's just call it "cooking_tick".
                        obj["cooking_tick"] = timer_val
                        obj["is_cooking"] = True

                    dynamic_objects[obj_id_counter] = obj

                # Check for players
                if player_pos_val > 0:
                    player_id = player_pos_val
                    p = {}
                    p["position"] = (col, row)  # TOODO: Check if this is correct
                    p["orientation"] = self.inv_orientation.get(orientation_val, "NORTH")
                    p["held_object"] = self.inv_held_item.get(held_obj_val, None)
                    p["player_id"] = player_id
                    players.append(p)

        return terrain_reconstructed, W, H, dynamic_objects, players


    def reconstruct_meta(self,meta_array):
        # meta_array is like [score, timestep, max_time, num_plates, r1_type, r1_time, r1_max_time, r2_type, r2_time, r2_max_time r3_type, r3_time, r3_max_time, r4_type, r4_time, r4_max_time, p1_action, p2_action]
        score = meta_array[0]
        timestep = meta_array[1]
        max_time = meta_array[2]
        num_plates = meta_array[3]

        order_list = []
        max_recipes = 4
        meta_items = 4  # score, timestep, max_time, num_plates -- but use 3 bc indexing
        index = meta_items
        for i in range(max_recipes):
            r_type = meta_array[index]
            r_time = meta_array[index + 1]
            r_total_time = meta_array[index + 2]
            if r_type != 0:
                recipe_name = self.inv_recipe.get(r_type, None)
                # The posted code did: order[2]-order[1], so r_time is remaining time
                # Assume we store orders as (recipe_name, remaining_time)
                order_list.append((recipe_name, r_time, r_total_time))
            index += 3

        # After these 2 + max_recipes*3 = 2+12=14 indices, we have player actions
        # p1_action_val = meta_array[14]
        # p2_action_val = meta_array[15]

        player_values = meta_array[meta_items + max_recipes * 3 :]

        joint_action = tuple(
            self.inv_action.get(action_val, "NONE") for action_val in player_values
        )

        return {
            "score": score,
            "timestep": timestep,
            "max_time": max_time,
            "order_list": order_list,
            "joint_action": joint_action,
            "num_plates": num_plates,
        }
    
    def process_burrito_episode(self, burrito_states: list[BurritoState], burrito_infos: dict, max_time: int, joint_action):
        img_array = []
        rew = 0
        shaped_rew = [0 for _ in burrito_infos[0]["shaped_r_by_agent"]]
        sparse_rew = 0
        for idx,state in enumerate(burrito_states):
            for agt_idx, rew in enumerate(burrito_infos[idx]["shaped_r_by_agent"]):
                shaped_rew[agt_idx]+=rew
                sparse_rew += burrito_infos[idx]["sparse_r_by_agent"][agt_idx]
            img = self.process_burrito_state(state,{"shaped_rew":shaped_rew,"sparse_rew":sparse_rew},joint_action[idx],max_time)
            img_array.append(img)
        return img_array

    def process_burrito_state(self, burrito_state: BurritoState, reward, joint_action, max_time):
        mdp = BurritoGridworld.from_layout_name(self.layout_name)
        env_grid = np.array(mdp.lossless_state_encoding(burrito_state)[0]) # returns as many layers as there are players
        terrain, W,H, dyn_objs, players = self.reconstruct_environment(env_grid)

        max_recipes = 4
        current_orders = []
        for i, order in enumerate(burrito_state.order_list[:max_recipes]):
            recipe_name = self.inv_recipe.get(recipe_mapping.get(order[0],0))
            remaining_time = order[2]-order[1]
            total_time = order[2]
            current_orders.append((recipe_name,remaining_time,total_time)) # total time
        img = self.make_image(
            terrain,
            W,
            H,
            dyn_objs,
            players,
            self.chef_sprites_img,
            self.chef_sprite_json,
            self.obj_sprites_img,
            self.obj_sprite_json,
            {
            "score": reward,
            "timestep": burrito_state.timestep*10,
            "max_time": max_time,
            "order_list": current_orders,
            "joint_action": joint_action,
            "num_plates": burrito_state.num_plates,
            }
        )
        return img
# Usage example:
# Assuming current_env_grid is the 10x12x9 numpy array and meta_array is the metadata row


    def get_sprite_data(self):
        # TODO:save these as dictionaries instead of lists
        self.chef_sprites_img = self.load_image(self.image_path("all_chef_sprites.png"))
        chef_sprite_json = self.load_json_data(self.image_path("all_chef_sprites.json"))
        self.obj_sprites_img = self.load_image(self.image_path("object_sprites.png"))

        self.chef_sprite_json = {
            frame["filename"]: frame["frame"] for frame in chef_sprite_json["frames"]
        }
        obj_sprite_json = self.load_json_data(self.image_path("object_sprites.json"))

        self.obj_sprite_json = {
            frame["filename"]: frame["frame"] for frame in obj_sprite_json["frames"]
        }
        return self.chef_sprites_img, self.chef_sprite_json, self.obj_sprites_img, self.obj_sprite_json


    def image_path(self,filename):
        target = os.path.normpath(
            os.path.join(
                os.path.dirname(__file__), "..", "..", "burrito", "html", "assets", filename
            )
        )
        return target


    def load_json_data(self,file_path):
        with open(file_path, "r") as file:
            data = json.load(file)
        return data


    def load_image(self,file_path):
        return Image.open(file_path)


    def get_sprite(self,image, sprite_data):
        # "frame": {"x":64,"y":0,"w":64,"h":64},
        x, y, width, height = sprite_data.values()
        return image.crop((x, y, x + width, y + height))


    def find_sprite(self,sprite_data, sprite_key):
        sprite_key = f"{sprite_key}.png"
        return sprite_data[sprite_key]


    def make_image(
        self, terrain, W, H, dyn_objs, players, chef_img, chef_json, obj_img, obj_json, meta_info
    ):
        # Create an image of the environment
        # Assume we have terrain, W, H, dyn_objs, players
        # Assume we have chef_img, chef_json, obj_img, obj_json

        # Create a blank image
        img = Image.new("RGB", (W * 64, H * 64), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default(16)

        order_width = 264
        total_width = order_width + (W * 64)
        total_height = H * 64
        max_time = meta_info["max_time"] * 10  # fps

        combined_image = Image.new(
            "RGB", (total_width, total_height), (0, 0, 0)
        )  # Black background
        # Draw the terrain

        draw = ImageDraw.Draw(combined_image)
        y_offset = 50  # Start below the order title

        draw.text(
            (10, 0),
            f"Orders:",
            font=font,
            fill=(255, 255, 255),
        )
        draw.text(
            (10,total_height-80),
            f"Joint Action: {meta_info['joint_action']}",
            font=font,
            fill=(255,255,255),
        )
        draw.text(
            (10, total_height-60),
            f"Shaped Reward: {round(meta_info['score']['shaped_rew'][0], 2)}, {round(meta_info['score']['shaped_rew'][1], 2)}",
            font=font,
            fill=(255,255,255)
        )
        draw.text(
            (10, total_height - 40),
            f"Sparse Reward: {meta_info['score']['sparse_rew']}",
            font=font,
            fill=(255, 255, 255),
        )
        draw.text(
            (10, total_height - 20),
            f"Time left: {(max_time - meta_info['timestep']) // 10}",
            font=font,
            fill=(255, 255, 255),
        )

        for dish, abs_time, total_time in meta_info["order_list"]:
            # Load the order image

            # Add the text (e.g., "steak burrito dish (30)")
            dish_name = dish.replace("dish", "").replace("_", " ")
            remaining_time = (total_time - abs_time)
            if remaining_time > total_time * 0.66:
                rect_color = "green"
            elif remaining_time > total_time * 0.33:
                rect_color = "yellow"
            else:
                rect_color = "red"

            rect_width = order_width - 10  # add the offset bc of the order list
            draw.rectangle(
                [10, y_offset, rect_width, y_offset + 80],
                fill=rect_color,
            )
            draw.text(
                (84, y_offset + 30),
                f"{dish_name} ({remaining_time})",
                fill="black",
                font=font,
            )

            sprite_loc = self.find_sprite(obj_json, dish)
            sprite = self.get_sprite(obj_img, sprite_loc)
            combined_image.paste(sprite, (10, y_offset + 10))  # Add some padding

            y_offset += 100  # Move down for the next order

        # This is the current map
        cnt = 0
        data = {}
        text_data = {}

        # Precompute static file paths
        static_file_paths = {
            "X": "counter",
            " ": "floor",
            "S": "deliver",
            "D": "dishes",
            "K": "clean_plate",
            "O": "onions",
            "W": "sink",
            "M": "meat",
            "P": "pot",
            "B": "cutting_board",
            "G": "grill",
            "T": "tortillas",
            "Z": "mushroom",
            "U": "trash",
            "R": "rice",
        }

        for row in terrain:
            for cell in row:
                if cell in static_file_paths:
                    fileName = static_file_paths[cell]
                    data[cnt] = [fileName, 0]
                    if cell == "D":
                        if cnt not in text_data:
                            text_data[cnt] = {
                                "plates": meta_info["num_plates"],
                                "position": (cnt // W, cnt % W),
                            }
                else:
                    print(f"Unknown cell: {cell}")
                cnt += 1

        # Now place the players
        for i, agent in enumerate(players):
            x, y = agent["position"]
            orientation = agent["orientation"]
            original_held_obj = agent["held_object"]
            player_id = agent["player_id"]
            color = self.COLORS[player_id - 1]
            # if player_id == 2:
            #     print(
            #         f"Player {player_id} at {x},{y} facing {orientation} holding '{original_held_obj}'"
            #     )
            if not original_held_obj:
                fileName = f"{orientation}-{color}"
            else:
                held_obj = (
                    original_held_obj.replace("{✓", "clean_plate")
                    .replace("{!✓", "steak-dish")
                    .replace("steak_onion", "steak-onion-dish")
                    .replace("{%✓", "chopped_steak-plate")
                    .replace("{R✓", "boiled_rice-plate")
                    .replace("{Rx", "charcoal")
                    .replace("{@✓", "chicken-dish")
                    .replace(
                        "boiled_chicken_onion", "chicken-onion-dish"
                    )  ## TODO CHANGE FILE NAMES
                    .replace("{^✓", "fried_mushroom-plate")
                    .replace("{^%", "chopped_steak-plate")
                    # .replace("steak_burrito_dish", "steak_burrito")
                    # .replace("mushroom_burrito_dish", "mushroom_burrito")
                )
                fileName = f"{orientation}-{held_obj}-{color}"

            data[y * W + x] = [fileName, 1]

        # Place the dynamic objects
        for obj in dyn_objs.values():
            original_name = obj["name"]
            name = obj["name"]
            x, y = obj["position"]
            if name == "boiled_rice":
                if "is_burnt" in obj and obj["is_burnt"]:
                    name = "burning_pot"
                elif "is_warning" in obj and obj["is_warning"]:
                    name = "warning_pot"
            elif name == "fried_mushroom" or name == "chopped_steak":
                if "is_burnt" in obj and obj["is_burnt"]:
                    name = "burning_grill"
                elif "is_warning" in obj and obj["is_warning"]:
                    name = "warning_grill"
            elif name == "mushroom_burrito":
                name = "mushroom_burrito_dish"
            elif name == "steak_burrito":
                name = "steak_burrito_dish"
            data[y * W + x] = [name, 0]

        # Draw some text on this...
        for obj in dyn_objs.values():
            x_pos, y_pos = obj["position"]
            grid = terrain
            text = {}

            # print(obj)

            if (
                (obj["name"] == "steak" and grid[y_pos][x_pos] == "G")
                or (obj["name"] == "chopped_steak" and grid[y_pos][x_pos] == "G")
                or (obj["name"] == "fried_mushroom" and grid[y_pos][x_pos] == "G")
                or (obj["name"] == "boiled_chicken" and grid[y_pos][x_pos] == "P")
                or (obj["name"] == "boiled_rice" and grid[y_pos][x_pos] == "P")
            ):
                # print(obj)
                if "warning_tick" in obj and obj["warning_tick"] != -1:
                    text.update(
                        {
                            "burning": str(obj["warning_tick"] / 10),
                            "position": (y_pos, x_pos),
                        }
                    )
                if "cooking_tick" in obj and obj["cooking_tick"] != -1:
                    text.update(
                        {
                            "cooking": str(obj["cooking_tick"] / 10),
                            "position": (y_pos, x_pos),
                        }
                    )
                if "waiting_tick" in obj and obj["waiting_tick"] != -1:
                    text.update(
                        {
                            "waiting": str(obj["waiting_tick"] / 10),
                            "position": (y_pos, x_pos),
                        }
                    )

                if "is_burnt" in obj and obj["is_burnt"]:
                    text.update({"is_burnt": True})

            if (
                (obj["name"] == "clean_plate" and grid[y_pos][x_pos] == "W")
                or (obj["name"] == "chopped_meat" and grid[y_pos][x_pos] == "B")
                or (obj["name"] == "chopped_mushroom" and grid[y_pos][x_pos] == "B")
                or (obj["name"] == "dirty_dishes" and grid[y_pos][x_pos] == "D")
            ):
                if "cooking_tick" in obj and obj["cooking_tick"] != -1:
                    text.update(
                        {"cooking": str(obj["cooking_tick"]), "position": (y_pos, x_pos)}
                    )
            if text:
                text_data[y_pos * W + x_pos] = text

        for key, value in data.items():
            x = key % W
            y = key // W
            sprite = None
            if value[1] == 0:
                sprite_loc = self.find_sprite(obj_json, value[0])
                sprite = self.get_sprite(obj_img, sprite_loc)
            else:
                sprite_loc = self.find_sprite(chef_json, value[0])
                sprite = self.get_sprite(chef_img, sprite_loc)

            img.paste(sprite, (x * 64, y * 64))
            combined_image.paste(img, (order_width, 0))

        font = ImageFont.load_default(24)

        for key, value in text_data.items():
            cooking = value.get("cooking", None)
            waiting = value.get("waiting", None)
            burning = value.get("burning", None)
            is_burnt = value.get("is_burnt", False)
            plates = value.get("plates", None)
            text_color = "black"
            rect_color = "blue"
            y, x = value["position"]

            if burning or is_burnt:
                rect_color = "red"
            elif waiting:
                rect_color = "yellow"
            elif cooking or waiting:
                rect_color = "green"

            # Draw the rectangle
            rect_x = order_width + x * 64  # add the offset bc of the order list
            rect_y = y * 64 + (64 // 4) * 3
            rect_width = 64
            rect_height = 64 // 4
            draw.rectangle(
                [rect_x, rect_y, rect_x + rect_width, rect_y + rect_height],
                fill=rect_color,
            )

            # print(f"Drawing text {cooking or waiting or burning or plates} at {x},{y}")

            # Draw the text
            text_value = str(int(float(burning or waiting or cooking or plates)))
            text_x = rect_x + rect_width // 4
            text_y = rect_y - 64 // 8
            draw.text((text_x, text_y), text_value, fill=text_color, font=font)

        return combined_image

    def make_video(self,img_array, custom_filename = None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"file_{timestamp}.mp4" if custom_filename == None else custom_filename
        filepath = os.path.join(os.getcwd(),"agent_data", filename)
        directory = os.path.dirname(filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        numpy_frames = [np.array(frame) for frame in img_array]

        clip = ImageSequenceClip(numpy_frames, fps=10)

        # Write the video file
        clip.write_videofile(filepath, codec="libx264")

