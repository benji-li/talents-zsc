### Install Multi-Human Overcooked-AI


### Game introduction

The objective is to fulfill and deliver the displayed orders sequentially. Failing to complete the orders in their correct sequence will result in them persisting in the order list. You'll earn 10 points for completing the dish that stands first in the order list, 5 points if the dish is present in the list but not in its correct sequence, and no points if the dish isn't in the order list at all.

#### Recipes:

- Steak dish: meat + clean plate
- Steak with garnish dish: meat + chopped onions + clean plate

Steak takes 8 seconds to cook. 

To clean a plate, you must pick up a plate and place it in the sink. Then, you press on interact in front of the sink three times to scrub it thoroughly. For preparing the garnish, you pick up the onion and place it on the chopping board. Next, you chop twice by performing two interactions.

#### Keyboard controls

- For the blue-hat chef, you move with arrow keys and interact with objects using space.
- For the green-hat chef, you move with wasd keys and interact with the key f. We also include support for this chef to be controlled with a controller.


### Installation
To complete the installation after unzipping this folder, run the following commands:

```
conda create -n overcooked_ai python=3.8
conda activate overcooked_ai
```

The above two lines create a conda environment named `overcooked_ai` and activate it.

```
cd multi-human-overcooked/overcooked_ai
pip install -e .
pip install moviepy
```

### Participant ID:

Please decide on a unique participant ID for your pair of players. This ID will be used to seperate you from the rest of the students.

### Once done, run this command to play! 

To try out different layouts, you can use the following commands:

```bash
python -m src.steakhouse_userstudy --participant_id <put your particiapnt id here> --record_video
```

e.g. if your participant ID is 10203, you would run:

```bash
python -m src.steakhouse_userstudy --participant_id 10203 --record_video
```

### Post-Game Data-Collection

After playing the game 3-4 (or more) times, please navigate to the folder called ``user_study`` and zip this folder and submit it to us. 
This folder contains all the data that we need to analyze your gameplay.