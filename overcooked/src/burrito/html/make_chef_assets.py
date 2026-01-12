import json
from PIL import Image

# Load JSON data
with open('chefs.json', 'r') as file:
    data = json.load(file)

# Load the background image
background_image = Image.open('./assets/floor_3.png')

# Load the base image that contains all components
base_image = Image.open('chefs.png')

# List of hat colors and cardinal directions
colors = ['blue', 'green', 'orange', 'purple', 'red']
directions = ['NORTH', 'EAST', 'SOUTH', 'WEST']

# Iterate through each frame in the JSON data
for frame_name, frame_data in data['frames'].items():
    frame_info = frame_data['frame']
    
    # Extract the frame from the base image
    cropped_image = base_image.crop((
        frame_info['x'],
        frame_info['y'],
        frame_info['x'] + frame_info['w'],
        frame_info['y'] + frame_info['h']
    ))

    # Check if the frame is a chef (non-hat images)
    if not any(color + 'hat' in frame_name for color in colors):
        # Save the base chef image without hats

        # Scale the chef image to 55x55 pixels
        scaled_chef = cropped_image.resize((55, 55), Image.LANCZOS)

        # Create a new image with the background
        chef_with_background = background_image.copy()
        chef_with_background.paste(scaled_chef, (4, 4), scaled_chef)  # Centering the chef on the background
        
        # Save the base chef image on the background without hats
        # chef_with_background.save(frame_name)

        # Add each hat to the chef and save the new image
        # for direction in directions:
        direction = 'SOUTH'
        for color in colors:
            hat_frame_name = f"{direction}-{color}hat.png"
            if hat_frame_name in data['frames']:
                hat_info = data['frames'][hat_frame_name]['frame']
                
                # Extract the hat image
                hat_image = base_image.crop((
                    hat_info['x'],
                    hat_info['y'],
                    hat_info['x'] + hat_info['w'],
                    hat_info['y'] + hat_info['h']
                ))
                
                # Create a copy of the chef image to add the hat
                # chef_with_hat = cropped_image.copy()
                # chef_with_hat.paste(hat_image, (0, 0), hat_image)

                # Scale the hat to 55x55 pixels
                scaled_hat = hat_image.resize((55, 55), Image.LANCZOS)
                
                # Create a copy of the background with the chef
                chef_with_hat = chef_with_background.copy()
                chef_with_hat.paste(scaled_hat, (4, 4), scaled_hat)  # Position hat on the chef
                
                # Save the combined chef and hat image
                chef_item = frame_name.split('.')[0]

                # Save the combined chef and hat image
                chef_item = frame_name.split('.')[0]
                chef_with_hat.save(f"./assets/chefs/chef-{chef_item}-{color}.png")
