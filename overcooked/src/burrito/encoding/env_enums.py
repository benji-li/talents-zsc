from enum import Enum


class CounterType(Enum):
    EMPTY = " "
    COUNTER = "X"
    DELIVERY = "S"


class IngredientType(Enum):
    NONE = " "
    MUSHROOM = "Z"
    TORTILLA = "T"
    MEAT = "M"
    RICE = "R"


class KitchenToolType(Enum):
    POT = "P"
    CUTTING_BOARD = "B"
    GRILL = "G"
    SINK = "W"
    TRASH = "U"
    DIRTY_DISHES = "D"


class HeldItemType(Enum):
    DIRTY_PLATE = "dirty_plate"
    CLEAN_PLATE = "clean_plate"
    TORTILLA = "tortilla"
    BOILED_RICE = "boiled_rice"

    MEAT = "meat"
    MUSHROOM = "mushroom"
    RICE = "rice"

    CHOPPED_MEAT = "chopped_meat"
    CHOPPED_STEAK = "chopped_steak"
    CHOPPED_MUSHROOM = "chopped_mushroom"
    FRIED_MUSHROOM = "fried_mushroom"

    BOILED_RICE_PLATE = "boiled_rice-plate"
    TORTILLA_PLATE = "tortilla-plate"
    FRIED_MUSHROOM_PLATE = "fried_mushroom-plate"
    CHOPPED_STEAK_PLATE = "chopped_steak-plate"

    BOILED_RICE_TORTILLA_PLATE = "boiled_rice-tortilla-plate"
    CHOPPED_STEAK_TORTILLA_PLATE = "chopped_steak-tortilla-plate"
    FRIED_MUSHROOM_TORTILLA_PLATE = "fried_mushroom-tortilla-plate"
    CHOPPED_STEAK_BOILED_RICE_PLATE = "chopped_steak-boiled_rice-plate"
    FRIED_MUSHROOM_BOILED_RICE_PLATE = "fried_mushroom-boiled_rice-plate"

    STEAK_BURRITO = "steak_burrito"
    MUSHROOM_BURRITO = "mushroom_burrito"

    CHARCOAL = "charcoal"
    CHARCOAL_PLATE = "charcoal-plate"
    FIRE_EXT = "fire_ext"


class CounterItemType(Enum):
    CHOPPED_MEAT = "chopped_meat"
    CHOPPED_MUSHROOM = "chopped_mushroom"
    FRIED_MUSHROOM = "fried_mushroom"
    BOILED_RICE_PLATE = "boiled_rice"
    CHOPPED_STEAK = "chopped_steak"
    FIRE_EXT = "fire_ext"
    DIRTY_PLATE = "dirty_plate"


class RecipeType(Enum):
    STEAK_BURRITO_DISH = "steak_burrito_dish"
    MUSHROOM_BURRITO_DISH = "mushroom_burrito_dish"


class OrientationType(Enum):
    NORTH = "NORTH"
    EAST = "EAST"
    SOUTH = "SOUTH"
    WEST = "WEST"


class ActionMappingType(Enum):
    NONE = "NONE"
    UP = "UP"
    RIGHT = "RIGHT"
    DOWN = "DOWN"
    LEFT = "LEFT"
    INTERACT = "INTERACT"


counter_mapping = {
    CounterType.EMPTY.value: 0,
    CounterType.COUNTER.value: 1,
    CounterType.DELIVERY.value: 2,
}

ingredient_mapping = {
    IngredientType.MEAT.value: 3,
    IngredientType.MUSHROOM.value: 4,
    IngredientType.TORTILLA.value: 5,
    IngredientType.RICE.value: 6,
}

kitchen_tool_mapping = {
    KitchenToolType.DIRTY_DISHES.value: 1,
    KitchenToolType.POT.value: 2,
    KitchenToolType.CUTTING_BOARD.value: 3,
    KitchenToolType.GRILL.value: 4,
    KitchenToolType.SINK.value: 5,
    KitchenToolType.TRASH.value: 6,
}

held_item_mapping = {
    HeldItemType.DIRTY_PLATE.value: 1,
    HeldItemType.CLEAN_PLATE.value: 2,
    HeldItemType.MEAT.value: 3,
    HeldItemType.MUSHROOM.value: 4,
    HeldItemType.TORTILLA.value: 5,
    HeldItemType.RICE.value: 6,
    HeldItemType.CHOPPED_MEAT.value: 7,
    HeldItemType.CHOPPED_MUSHROOM.value: 8,
    HeldItemType.BOILED_RICE_PLATE.value: 9,
    HeldItemType.CHOPPED_STEAK_PLATE.value: 10,
    HeldItemType.FRIED_MUSHROOM_PLATE.value: 11,
    HeldItemType.TORTILLA_PLATE.value: 12,
    HeldItemType.BOILED_RICE_TORTILLA_PLATE.value: 13,
    HeldItemType.CHOPPED_STEAK_TORTILLA_PLATE.value: 14,
    HeldItemType.FRIED_MUSHROOM_TORTILLA_PLATE.value: 15,
    HeldItemType.CHOPPED_STEAK_BOILED_RICE_PLATE.value: 16,
    HeldItemType.FRIED_MUSHROOM_BOILED_RICE_PLATE.value: 17,
    HeldItemType.STEAK_BURRITO.value: 18,
    HeldItemType.MUSHROOM_BURRITO.value: 19,
    HeldItemType.CHARCOAL.value: 20,
    HeldItemType.CHARCOAL_PLATE.value: 21,
    HeldItemType.CHOPPED_STEAK.value: 22,
    HeldItemType.CHOPPED_MUSHROOM.value: 23,
    HeldItemType.FRIED_MUSHROOM.value: 24,
    HeldItemType.BOILED_RICE.value: 25,
    HeldItemType.FIRE_EXT.value: 40,
}

recipe_mapping = {
    RecipeType.STEAK_BURRITO_DISH.value: 13,
    RecipeType.MUSHROOM_BURRITO_DISH.value: 14,
}

orientation_mapping = {
    OrientationType.NORTH.value: 1,
    OrientationType.EAST.value: 2,
    OrientationType.SOUTH.value: 3,
    OrientationType.WEST.value: 4,
}

action_mapping = {
    ActionMappingType.NONE.value: 0,
    ActionMappingType.UP.value: 1,
    ActionMappingType.RIGHT.value: 2,
    ActionMappingType.DOWN.value: 3,
    ActionMappingType.LEFT.value: 4,
    ActionMappingType.INTERACT.value: 5,
}
