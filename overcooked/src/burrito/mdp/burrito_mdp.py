import copy
import logging
import itertools
from collections import defaultdict
import random

import numpy as np
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState  # PlayerState,
from overcooked_ai_py.mdp.overcooked_mdp import (
    Action,
    Direction,
    ObjectState,
    OvercookedGridworld,
    Recipe,
    SoupState,
)
from .utils import read_layout_dict

from burrito.encoding.env_enums import (
    counter_mapping,
    ingredient_mapping,
    kitchen_tool_mapping,
    held_item_mapping,
    recipe_mapping,
    orientation_mapping,
    action_mapping,
    KitchenToolType,
)
logger = logging.getLogger(__name__)


class PlayerState:
    """State of a player in BurritoGridworld.

    position: (x, y) tuple representing the player's location.
    orientation: Direction.NORTH/SOUTH/EAST/WEST representing orientation.
    held_object: ObjectState representing the object held by the player, or
        None if there is no such object.
    num_ingre_held (int): Number of times the player has held an ingredient
        object (onion or tomato).
    num_plate_held (int): Number of times the player has held a plate
    num_served (int): Number of times the player has served food
    """

    def __init__(
        self,
        position,
        orientation,
        held_object=None,
        num_ingre_held=0,
        num_plate_held=0,
        num_served=0,
        active_log=[],
        stuck_log=[],
    ):
        self.position = tuple(position)
        self.orientation = tuple(orientation)
        self.held_object = held_object
        self.num_ingre_held = num_ingre_held
        self.num_plate_held = num_plate_held
        self.num_served = num_served
        self.active_log = active_log.copy()
        self.stuck_log = stuck_log.copy()

        assert self.orientation in Direction.ALL_DIRECTIONS
        if self.held_object is not None:
            assert isinstance(self.held_object, ObjectState)
            assert self.held_object.position == self.position

    @property
    def pos_and_or(self):
        return self.position, self.orientation

    def get_pos_and_or(self):
        return self.position, self.orientation

    def has_object(self):
        return self.held_object is not None

    def get_object(self):
        assert self.has_object()
        return self.held_object

    def set_object(self, obj):
        assert not self.has_object()
        obj.position = self.position
        self.held_object = obj

    def remove_object(self):
        assert self.has_object()
        obj = self.held_object
        self.held_object = None
        return obj

    def update_pos_and_or(self, new_position, new_orientation):
        self.position = new_position
        self.orientation = new_orientation
        if self.has_object():
            self.get_object().position = new_position

    def deepcopy(self):
        new_obj = None if self.held_object is None else self.held_object.deepcopy()
        return PlayerState(
            self.position,
            self.orientation,
            new_obj,
            self.num_ingre_held,
            self.num_plate_held,
            self.num_served,
            self.active_log,
            self.stuck_log,
        )

    def __eq__(self, other):
        return (
            isinstance(other, PlayerState)
            and self.position == other.position
            and self.orientation == other.orientation
            and self.held_object == other.held_object
        )

    def __hash__(self):
        return hash((self.position, self.orientation, self.held_object))

    def __repr__(self):
        return (
            f"{self.position} facing {self.orientation} holding "
            f"{str(self.held_object)}"
        )

    def to_dict(self):
        return {
            "position": self.position,
            "orientation": self.orientation,
            "held_object": (
                self.held_object.to_dict() if self.held_object is not None else None
            ),
        }

    def get_workload(
        self,
    ):
        return {
            "num_ingre_held": self.num_ingre_held,
            "num_plate_held": self.num_plate_held,
            "num_served": self.num_served,
        }

    def print_workload(
        self,
    ):
        logger.info(f"Number of ingredients held: {self.num_ingre_held}")
        logger.info(f"Number of plates held: {self.num_plate_held}")
        logger.info(f"Number of soup served: {self.num_served}")

    @staticmethod
    def from_dict(player_dict):
        player_dict = copy.deepcopy(player_dict)
        held_obj = player_dict["held_object"]
        if held_obj is not None:
            player_dict["held_object"] = ObjectState.from_dict(held_obj)
        return PlayerState(**player_dict)


class Burrito_Recipe(Recipe):
    MAX_NUM_INGREDIENTS = 3
    CHICKEN = "chicken"
    MEAT = "meat"
    ONION = "onion"
    MUSHROOM = "mushroom"
    CHOPPED_MEAT = "chopped_meat"
    CHOPPED_STEAK = "chopped_steak"
    CHOPPED_MUSHROOM = "chopped_mushroom"
    RICE = "rice"
    BOILED_RICE = "boiled_rice"
    TORTILLA = "tortilla"

    ALL_INGREDIENTS = [
        CHICKEN,
        MEAT,
        ONION,
        MUSHROOM,
        CHOPPED_STEAK,
        CHOPPED_MEAT,
        CHOPPED_MUSHROOM,
        RICE,
        BOILED_RICE,
        TORTILLA,
    ]
    STR_REP = {
        CHOPPED_MUSHROOM: "^",
        CHICKEN: "@",
        MEAT: "!",
        ONION: "ø",
        CHOPPED_MEAT: "%",
        RICE: "R",
        BOILED_RICE: "Y",
        TORTILLA: "T",
    }

    @classmethod
    def configure(cls, conf):
        cls._conf = conf
        cls._configured = True
        cls._computed = False
        cls.MAX_NUM_INGREDIENTS = conf.get("max_num_ingredients", 3)

        cls._cook_time = None
        cls.delivery_reward = None
        cls.in_order_delivery_reward = None
        cls._value_mapping = None
        cls._time_mapping = None
        cls._onion_value = None
        cls._steak_time = None
        cls._chicken_value = None
        cls._chicken_time = None
        cls._rice_time = None
        cls._mushroom_time = None
        cls._chopped_steak_time = None
        cls._burn_time = None

        ## Basic checks for validity ##

        # Mutual Exclusion
        if (
            ("chicken_time" in conf and "steak_time" not in conf)
            or ("steak_time" in conf and "chicken_time" not in conf)
            or ("mushroom_time" in conf and "steak_time" not in conf)
            or ("rice_time" in conf and "rice_time" not in conf)
        ):
            raise ValueError(
                "Must specify 'steak_time', 'chicken_time', and 'mushroom_time'"
            )
        if (
            ("chicken_value" in conf and "steak_value" not in conf)
            or ("steak_value" in conf and "chicken_value" not in conf)
            or ("mushroom_value" in conf and "steak_value" not in conf)
            or ("mushroom_value" in conf and "chicken_value" not in conf)
        ):
            raise ValueError(
                "Must specify 'steak_value', 'chicken_value', and 'mushroom_value'"
            )
        if "chicken_value" in conf and "delivery_reward" in conf:
            raise ValueError("'delivery_reward' incompatible with '<ingredient>_value'")
        if "chicken_value" in conf and "recipe_values" in conf:
            raise ValueError("'recipe_values' incompatible with '<ingredient>_value'")
        if "recipe_values" in conf and "delivery_reward" in conf:
            raise ValueError("'delivery_reward' incompatible with 'recipe_values'")
        if "chicken_time" in conf and "cook_time" in conf:
            raise ValueError("'cook_time' incompatible with '<ingredient>_time'")
        if "chicken_time" in conf and "recipe_times" in conf:
            raise ValueError("'recipe_times' incompatible with '<ingredient>_time'")
        if "recipe_times" in conf and "cook_time" in conf:
            raise ValueError("'cook_time' incompatible with 'recipe_times'")

        # recipe_ lists and orders compatibility
        if "recipe_values" in conf:
            if not "all_orders" in conf or not conf["all_orders"]:
                raise ValueError(
                    "Must specify 'all_orders' if 'recipe_values' specified"
                )
            if not len(conf["all_orders"]) == len(conf["recipe_values"]):
                raise ValueError(
                    "Number of recipes in 'all_orders' must be the same as number in 'recipe_values"
                )
        if "recipe_times" in conf:
            if not "all_orders" in conf or not conf["all_orders"]:
                raise ValueError(
                    "Must specify 'all_orders' if 'recipe_times' specified"
                )
            if not len(conf["all_orders"]) == len(conf["recipe_times"]):
                raise ValueError(
                    "Number of recipes in 'all_orders' must be the same as number in 'recipe_times"
                )

        ## Conifgure ##
        # print(conf)

        if "cook_time" in conf:
            cls._cook_time = conf["cook_time"]

        if "delivery_reward" in conf:
            cls.delivery_reward = conf["delivery_reward"]

        if "in_order_delivery_reward" in conf:
            cls.in_order_delivery_reward = conf["in_order_delivery_reward"]

        if "recipe_values" in conf:
            cls._value_mapping = {
                cls.from_dict(recipe): value
                for (recipe, value) in zip(conf["all_orders"], conf["recipe_values"])
            }

        if "recipe_times" in conf:
            cls._time_mapping = {
                cls.from_dict(recipe): time
                for (recipe, time) in zip(conf["all_orders"], conf["recipe_times"])
            }

        if "chicken_time" in conf:
            cls._chicken_time = conf["chicken_time"]

        if "steak_time" in conf:
            cls._steak_time = conf["steak_time"]

        if "mushroom_time" in conf:
            cls._steak_time = conf["mushroom_time"]

        if "rice_time" in conf:
            cls._rice_time = conf["rice_time"]

        if "chopped_steak_time" in conf:
            cls._chopped_steak_time = conf["chopped_steak_time"]

        if "chicken_value" in conf:
            cls._chicken_value = conf["chicken_value"]

        if "warn_time" in conf:
            cls._warn_time = conf["warn_time"]

        if "steak_value" in conf:
            cls._steak_value = conf["steak_value"]

        if "wash_time" in conf:
            cls._wash_time = conf["wash_time"]


class BurntObjectState(ObjectState):
    def __init__(self, id, name, position, extinguished=True):
        self.id = id
        self._is_extinguished = extinguished
        super(BurntObjectState, self).__init__(
            name=name,
            position=position,
        )

    def deepcopy(self):
        return BurntObjectState(self.id, self.name, self.position)

    def __eq__(self, other):
        return (
            isinstance(other, BurntObjectState)
            and self.name == other.name
            and self.position == other.position
        )

    @classmethod
    def from_dict(cls, obj_dict):
        obj_dict = copy.deepcopy(obj_dict)
        return BurntObjectState(**obj_dict)

    @property
    def is_ready(self):
        return True

    @property
    def is_burnt(self):
        return True

    @property
    def is_extinguished(self):
        return self._is_extinguished

    def extinguish(self):
        self._is_extinguished = True

    def to_dict(self):
        info_dict = super(BurntObjectState, self).to_dict()
        return info_dict

    def is_valid(self):
        # print("name", self.name)
        return self.name in [
            "charcoal",
            "charcoal-plate",
            "charcoal_pot",
            "charcoal_grill",
        ]


class IdObjectState(ObjectState):
    def __init__(self, id, name, position):
        self.id = id
        super(IdObjectState, self).__init__(
            name=name,
            position=position,
        )

    def deepcopy(self):
        return IdObjectState(self.id, self.name, self.position)

    def __eq__(self, other):
        return (
            isinstance(other, IdObjectState)
            and self.name == other.name
            and self.position == other.position
        )

    @classmethod
    def from_dict(cls, obj_dict):
        obj_dict = copy.deepcopy(obj_dict)
        return IdObjectState(**obj_dict)

    def is_valid(self):
        return self.name in [
            "dirty_plate",
            "clean_plate",
            "meat",
            "dish",
            "onion",
            "steak",
            "steak_onion",
            "mushroom",
            "chopped_meat",
            "chopped_mushroom",
            "rice",
            "boiled_rice",
            "boiled_rice-plate",
            "tortilla",
            "fried_mushroom-plate",
            "chopped_steak-plate",
            "chopped_steak-boiled_rice-plate",
            "fried_mushroom-boiled_rice-plate",
            "mushroom_burrito",
            "steak_burrito",
            "tortilla-plate",
            "rice-plate",
            "chopped_steak-tortilla-plate",
            "fried_mushroom-tortilla-plate",
            "boiled_rice-tortilla-plate",
            "chopped_steak-boiled_rice-plate",
            "fried_mushroom-boiled_rice-plate",
            "charcoal",
            "fire_ext",
            "charcoal_pot",
            "charcoal_grill",
            "dishes",
        ]


class ChickenState(IdObjectState):
    def __init__(
        self,
        id,
        name,
        position,
        ingredients=[],
        cooking_tick=-1,
        cook_time=-1,
        **kwargs,
    ):
        """
        Represents a soup object. An object becomes a soup the instant it is placed in a pot. The
        soup's recipe is a list of ingredient names used to create it. A soup's recipe is undetermined
        until it has begun cooking.

        position (tupe): (x, y) coordinates in the grid
        ingrdients (list(ObjectState)): Objects that have been used to cook this soup. Determiens @property recipe
        cooking (int): How long the soup has been cooking for. -1 means cooking hasn't started yet
        cook_time(int): How long soup needs to be cooked, used only mostly for getting soup from dict with supplied cook_time, if None self.recipe.time is used
        """
        super(ChickenState, self).__init__(id, name, position)
        self._ingredients = ingredients
        self._cooking_tick = cooking_tick
        self._recipe = None
        self._cook_time = (
            cook_time if cook_time > 0 else Burrito_Recipe._chicken_time
        )

    def __eq__(self, other):
        return (
            isinstance(other, ChickenState)
            and self.name == other.name
            and self.position == other.position
            and self._cooking_tick == other._cooking_tick
            and all(
                [
                    this_i == other_i
                    for this_i, other_i in zip(self._ingredients, other._ingredients)
                ]
            )
        )

    def __hash__(self):
        ingredient_hash = hash(tuple([hash(i) for i in self._ingredients]))
        supercls_hash = super(ChickenState, self).__hash__()
        return hash((supercls_hash, self._cooking_tick, ingredient_hash))

    def __repr__(self):
        supercls_str = super(ChickenState, self).__repr__()
        ingredients_str = self._ingredients.__repr__()
        return "{}\nIngredients:\t{}\nCooking Tick:\t{}".format(
            supercls_str, ingredients_str, self._cooking_tick
        )

    def __str__(self):
        res = "{"
        for ingredient in sorted(self.ingredients):
            res += Burrito_Recipe.STR_REP[ingredient]
        if self.is_cooking:
            res += str(self._cooking_tick)
        elif self.is_ready:
            res += str("✓")
        return res

    @IdObjectState.position.setter
    def position(self, new_pos):
        self._position = new_pos
        for ingredient in self._ingredients:
            ingredient.position = new_pos

    @property
    def ingredients(self):
        return [ingredient.name for ingredient in self._ingredients]

    @property
    def is_cooking(self):
        return not self.is_idle and not self.is_ready

    @property
    def recipe(self):
        if self.is_idle:
            raise ValueError("Recipe is not determined until soup begins cooking")
        if not self._recipe:
            self._recipe = Burrito_Recipe(self.ingredients)
        return self._recipe

    @property
    def value(self):
        return self.recipe.value

    @property
    def cook_time(self):
        # used mostly when cook time is supplied by state dict
        if self._cook_time is not None:
            return self._cook_time
        else:
            return self.recipe.time

    @property
    def cook_time_remaining(self):
        return max(0, self.cook_time - self._cooking_tick)

    @property
    def is_ready(self):
        if self.is_idle:
            return False
        return self._cooking_tick >= self.cook_time

    @property
    def is_idle(self):
        return self._cooking_tick < 0

    @property
    def is_full(self):
        return (
            not self.is_idle
            or len(self.ingredients) == Burrito_Recipe.MAX_NUM_INGREDIENTS
        )

    def is_valid(self):
        if not all(
            [ingredient.position == self.position for ingredient in self._ingredients]
        ):
            return False
        if len(self.ingredients) > Burrito_Recipe.MAX_NUM_INGREDIENTS:
            return False
        return True

    def auto_finish(self):
        if len(self.ingredients) == 0:
            raise ValueError("Cannot finish chicken with no ingredients")
        self._cooking_tick = 0
        self._cooking_tick = self.cook_time

    def add_ingredient(self, ingredient):
        if not ingredient.name in Burrito_Recipe.ALL_INGREDIENTS:
            #print("Invalid ingredient", ingredient.name, "chicken State")
            raise ValueError("Invalid ingredient")
        if self.is_full:
            raise ValueError("Reached maximum number of ingredients in recipe")
        ingredient.position = self.position
        self._ingredients.append(ingredient)

    def add_ingredient_from_str(self, ingredient_str):
        ingredient_obj = IdObjectState(ingredient_str, self.position)
        self.add_ingredient(ingredient_obj)

    def pop_ingredient(self):
        if not self.is_idle:
            raise ValueError(
                "Cannot remove an ingredient from this Chicken at this time"
            )
        if len(self._ingredients) == 0:
            raise ValueError("No ingredient to remove")
        return self._ingredients.pop()

    def begin_cooking(self):
        if not self.is_idle:
            raise ValueError("Cannot begin cooking this chicken soup at this time")
        if len(self.ingredients) == 0:
            raise ValueError(
                "Must add at least one ingredient to chicken soup before you can begin cooking"
            )
        self._cooking_tick = 0

    def cook(self):
        if self.is_idle:
            raise ValueError("Must begin cooking before advancing cook tick")
        if self.is_ready:
            raise ValueError("Cannot cook a soup that is already done")
        self._cooking_tick += 1

    def deepcopy(self):
        return ChickenState(
            self.id,
            self.name,
            self.position,
            [ingredient.deepcopy() for ingredient in self._ingredients],
            self._cooking_tick,
            self._cook_time,
        )

    def to_dict(self):
        info_dict = super(ChickenState, self).to_dict()
        ingrdients_dict = [ingredient.to_dict() for ingredient in self._ingredients]
        info_dict["_ingredients"] = ingrdients_dict
        info_dict["cooking_tick"] = self._cooking_tick
        info_dict["is_cooking"] = self.is_cooking
        info_dict["is_ready"] = self.is_ready
        info_dict["is_idle"] = self.is_idle
        info_dict["cook_time"] = -1 if self.is_idle else self.cook_time

        # This is for backwards compatibility w/ overcooked-demo
        # Should be removed once overcooked-demo is updated to use 'cooking_tick' instead of '_cooking_tick'
        info_dict["_cooking_tick"] = self._cooking_tick
        return info_dict

    @classmethod
    def from_dict(cls, obj_dict):
        obj_dict = copy.deepcopy(obj_dict)
        if obj_dict["name"] != "soup":
            return super(ChickenState, cls).from_dict(obj_dict)

        if "state" in obj_dict:
            # Legacy soup representation
            ingredient, num_ingredient, time = obj_dict["state"]
            cooking_tick = -1 if time == 0 else time
            finished = time >= Burrito_Recipe._chicken_time
            if ingredient == Burrito_Recipe.CHICKEN:
                return ChickenState.get_soup(
                    obj_dict["position"],
                    num_chicken=num_ingredient,
                    cooking_tick=cooking_tick,
                    finished=finished,
                )
        ingredients_objs = [
            IdObjectState.from_dict(ing_dict) for ing_dict in obj_dict["_ingredients"]
        ]
        obj_dict["ingredients"] = ingredients_objs
        return cls(**obj_dict)

    @classmethod
    def get_chicken(
        cls, position, num_chicken=0, cooking_tick=-1, finished=False, **kwargs
    ):
        if num_chicken < 0:
            raise ValueError("Number of active ingredients must be positive")
        if num_chicken > Burrito_Recipe.MAX_NUM_INGREDIENTS:
            raise ValueError("Too many ingredients specified for this soup")
        if cooking_tick >= 0 and num_chicken == 0:
            raise ValueError("_cooking_tick must be -1 for empty soup")
        if finished and num_chicken == 0:
            raise ValueError("Empty soup cannot be finished")
        chicken = [
            IdObjectState(Burrito_Recipe.CHICKEN, position)
            for _ in range(num_chicken)
        ]
        ingredients = chicken
        soup = cls(position, ingredients, cooking_tick)
        if finished:
            soup.auto_finish()
        return soup


class CookableObjectState(IdObjectState):
    def __init__(
        self,
        id,
        name,
        position,
        ingredients=[],
        cooking_tick=-1,
        warning_tick=-1,
        waiting_tick=-1,
        cook_time=-1,
        warn_time=-1,
        waiting_time=-1,
        picked_up=False,
        is_extinguished=False,
        **kwargs,
    ):
        """
        Represents a soup object. An object becomes a soup the instant it is placed in a pot. The
        soup's recipe is a list of ingredient names used to create it. A soup's recipe is undetermined
        until it has begun cooking.

        position (tupe): (x, y) coordinates in the grid
        ingrdients (list(ObjectState)): Objects that have been used to cook this soup. Determiens @property recipe
        cooking (int): How long the soup has been cooking for. -1 means cooking hasn't started yet
        cook_time(int): How long soup needs to be cooked, used only mostly for getting soup from dict with supplied cook_time, if None self.recipe.time is used
        """
        super(CookableObjectState, self).__init__(id, name, position)
        self._ingredients = ingredients
        self._cooking_tick = cooking_tick
        self._warning_tick = warning_tick
        self._waiting_tick = waiting_tick
        self._recipe = None
        self._cook_time = cook_time if cook_time > 0 else Burrito_Recipe._rice_time
        self._warn_time = warn_time if warn_time > 0 else Burrito_Recipe._warn_time
        self._waiting_time = (
            waiting_time if waiting_time > 0 else Burrito_Recipe._rice_time
        )
        self.name = name
        self._picked_up = picked_up
        self._is_extinguished = is_extinguished

    def __eq__(self, other):
        return (
            isinstance(other, CookableObjectState)
            and self.name == other.name
            and self.position == other.position
            and self._cooking_tick == other._cooking_tick
            and self._warning_tick == other._warning_tick
            and self._waiting_tick == other._waiting_tick
            and self._picked_up == other._picked_up
            and self._is_extinguished == other._is_extinguished
            and all(
                [
                    this_i == other_i
                    for this_i, other_i in zip(self._ingredients, other._ingredients)
                ]
            )
        )

    def __hash__(self):
        ingredient_hash = hash(tuple([hash(i) for i in self._ingredients]))
        supercls_hash = super(CookableObjectState, self).__hash__()
        return hash(
            (supercls_hash, self._cooking_tick, self._warning_tick, ingredient_hash)
        )

    def __repr__(self):
        supercls_str = super(CookableObjectState, self).__repr__()
        ingredients_str = self._ingredients.__repr__()
        return "{}\nIngredients:\t{}\nCooking Tick:\t{}\tBurning Tick:{}\t{}Waiting Tick:\t{}Picked up:\t{}\tExtinguished{}".format(
            supercls_str,
            ingredients_str,
            self._cooking_tick,
            self._warning_tick,
            self._waiting_tick,
            self._picked_up,
            self._is_extinguished,
        )

    def __str__(self):
        res = "{"
        for ingredient in sorted(self.ingredients):
            res += Burrito_Recipe.STR_REP[ingredient]
        if self.is_cooking:
            res += str(self._cooking_tick)
        elif self.is_burnt:
            rest += str("x")
        elif self.is_ready or self.is_warning:
            res += str("✓")
        return res

    @IdObjectState.position.setter
    def position(self, new_pos):
        self._position = new_pos
        for ingredient in self._ingredients:
            ingredient.position = new_pos

    @property
    def ingredients(self):
        return [ingredient.name for ingredient in self._ingredients]

    @property
    def is_cooking(self):
        return not self.is_idle and not self.is_ready

    @property
    def is_warning(self):
        return self._waiting_tick >= self.wait_time and not self.is_burnt

    @property
    def is_picked_up(self):
        return self._picked_up

    def set_pick_up_true(self):
        self._picked_up = True

    @property
    def is_extinguished(self):
        return self._is_extinguished

    def set_extinguish_true(self):
        self._is_extinguished = True

    @property
    def is_waiting_for_pickup(self):
        return not self.is_idle and self.is_ready and not self.is_warning and not self.is_burnt

    @property
    def recipe(self):
        if self.is_idle:
            raise ValueError("Recipe is not determined until soup begins cooking")
        if not self._recipe:
            self._recipe = Burrito_Recipe(self.ingredients)
        return self._recipe

    @property
    def value(self):
        return self.recipe.value

    @property
    def cook_time(self):
        # used mostly when cook time is supplied by state dict
        if self._cook_time is not None:
            return self._cook_time
        else:
            return self.recipe.time

    @property
    def warn_time(self):
        # used mostly when cook time is supplied by state dict
        if self._warn_time is not None:
            return self._warn_time
        else:
            return self.recipe.time

    @property
    def wait_time(self):
        # used mostly when cook time is supplied by state dict
        if self._waiting_time is not None:
            return self._waiting_time
        else:
            return self.recipe.time

    @property
    def cook_time_remaining(self):
        return max(0, self.cook_time - self._cooking_tick)

    @property
    def burn_time_remaining(self):
        return max(0, self.burn_time - self._burning_tick)

    @property
    def is_ready(self):
        if self.is_idle:
            return False
        # print("is ready?", self._cooking_tick, self.cook_time)
        return self._cooking_tick >= self.cook_time

    @property
    def is_burnt(self):
        if self.is_idle or self.is_cooking:
            return False
        return self._warning_tick >= self.warn_time

    @property
    def is_idle(self):
        return self._cooking_tick < 0

    @property
    def is_full(self):
        return (
            not self.is_idle
            or len(self.ingredients) == Burrito_Recipe.MAX_NUM_INGREDIENTS
        )

    def is_valid(self):
        if not all(
            [ingredient.position == self.position for ingredient in self._ingredients]
        ):
            return False
        if len(self.ingredients) > Burrito_Recipe.MAX_NUM_INGREDIENTS:
            return False
        return True

    def auto_finish(self):
        if len(self.ingredients) == 0:
            raise ValueError("Cannot finish chicken with no ingredients")
        self._cooking_tick = 0
        self._cooking_tick = self.cook_time

    def add_ingredient(self, ingredient):
        if not ingredient.name in Burrito_Recipe.ALL_INGREDIENTS:
            #print("Invalid ingredient", ingredient.name, "chicken State")
            raise ValueError("Invalid ingredient")
        if self.is_full:
            raise ValueError("Reached maximum number of ingredients in recipe")
        ingredient.position = self.position
        self._ingredients.append(ingredient)

    def add_ingredient_from_str(self, ingredient_str):
        ingredient_obj = IdObjectState(ingredient_str, self.position)
        self.add_ingredient(ingredient_obj)

    def pop_ingredient(self):
        if not self.is_idle:
            raise ValueError(
                f"Cannot remove an ingredient from this {self.name} at this time"
            )
        if len(self._ingredients) == 0:
            raise ValueError("No ingredient to remove")
        return self._ingredients.pop()

    def begin_cooking(self):
        if not self.is_idle:
            raise ValueError("Cannot begin cooking this chicken soup at this time")
        if len(self.ingredients) == 0:
            raise ValueError(
                "Must add at least one ingredient to chicken soup before you can begin cooking"
            )
        self._cooking_tick = 0

    def begin_warning(self):
        if not self.is_ready:
            raise ValueError("Cannot begin rice warning at this time")
        self._warning_tick = 0

    def begin_waiting_for_pickup(self):
        if not self.is_ready:
            raise ValueError("Cannot begin burning this rice at this time")
        self._waiting_tick = 0

    def cook(self):
        # print("cooking rice: ", self._cooking_tick)
        if self.is_idle:
            raise ValueError("Must begin cooking before advancing cook tick")
        if self.is_ready:
            raise ValueError("Cannot cook a soup that is already done")
        if self.is_burnt:
            return
        self._cooking_tick += 1

    def warn(self):
        if self.is_idle:
            raise ValueError("Must begin cooking before advancing burn tick")
        if self.is_burnt:
            raise ValueError("Cannot burn a soup that is already burnt")
        self._warning_tick += 1

    def wait(self):
        # print("cooking rice: ", self._cooking_tick)
        if self.is_idle:
            raise ValueError("Must begin cooking before advancing cook tick")
        if self.is_burnt:
            raise ValueError("Cannot wait for a soup that is burning")
        self._waiting_tick += 1

    def deepcopy(self):
        return CookableObjectState(
            self.id,
            self.name,
            self.position,
            [ingredient.deepcopy() for ingredient in self._ingredients],
            self._cooking_tick,
            self._warning_tick,
            self._waiting_tick,
            self._cook_time,
            self._warn_time,
            self._waiting_time,
            self._picked_up,
            self._is_extinguished,
        )

    def to_dict(self):
        info_dict = super(CookableObjectState, self).to_dict()
        ingrdients_dict = [ingredient.to_dict() for ingredient in self._ingredients]
        info_dict["_ingredients"] = ingrdients_dict
        info_dict["cooking_tick"] = self._cooking_tick
        info_dict["warning_tick"] = self._warning_tick
        info_dict["is_cooking"] = self.is_cooking
        info_dict["is_burnt"] = self.is_burnt
        info_dict["is_ready"] = self.is_ready
        info_dict["is_idle"] = self.is_idle
        info_dict["cook_time"] = -1 if self.is_idle else self.cook_time
        info_dict["warn_time"] = -1 if not self.is_ready else self.warn_time
        info_dict["waiting_time"] = -1 if not self.is_ready else self.wait_time

        # This is for backwards compatibility w/ overcooked-demo
        # Should be removed once overcooked-demo is updated to use 'cooking_tick' instead of '_cooking_tick'
        info_dict["_cooking_tick"] = self._cooking_tick
        return info_dict

    @classmethod
    def from_dict(cls, obj_dict):
        obj_dict = copy.deepcopy(obj_dict)
        if obj_dict["name"] != "soup":
            return super(CookableObjectState, cls).from_dict(obj_dict)

        if "state" in obj_dict:
            # Legacy soup representation
            ingredient, num_ingredient, time = obj_dict["state"]
            cooking_tick = -1 if time == 0 else time
            burning_tick = -1 if time == 0 else time
            finished = time >= Burrito_Recipe._chicken_time
            if ingredient == Burrito_Recipe.RICE:
                return CookableObjectState.get_soup(
                    obj_dict["position"],
                    num_chicken=num_ingredient,
                    cooking_tick=cooking_tick,
                    burning_tick=burning_tick,
                    finished=finished,
                )
        ingredients_objs = [
            IdObjectState.from_dict(ing_dict) for ing_dict in obj_dict["_ingredients"]
        ]
        obj_dict["ingredients"] = ingredients_objs
        return cls(**obj_dict)

    # TODO: fix this up to include the rice
    @classmethod
    def get_rice(
        cls,
        position,
        num_chicken=0,
        cooking_tick=-1,
        burning_tick=-1,
        finished=False,
        **kwargs,
    ):
        if num_chicken < 0:
            raise ValueError("Number of active ingredients must be positive")
        if num_chicken > Burrito_Recipe.MAX_NUM_INGREDIENTS:
            raise ValueError("Too many ingredients specified for this soup")
        if cooking_tick >= 0 and num_chicken == 0:
            raise ValueError("_cooking_tick must be -1 for empty soup")
        if finished and num_chicken == 0:
            raise ValueError("Empty soup cannot be finished")
        chicken = [
            IdObjectState(Burrito_Recipe.RICE, position) for _ in range(num_chicken)
        ]
        ingredients = chicken
        soup = cls(position, ingredients, cooking_tick)
        if finished:
            #print("AUTO FINISHING RICE")
            soup.auto_finish()
        return soup


class PlateState(IdObjectState):
    def __init__(self, id, name, position, rinse_total=3, rinse_count=-1, **kwargs):
        super(PlateState, self).__init__(id, name, position)
        self._cook_time = rinse_total
        self._cooking_tick = rinse_count

    def __eq__(self, other):
        return (
            isinstance(other, PlateState)
            and self.id == other.id
            and self.name == other.name
            and self.position == other.position
            and self._cooking_tick == other._cooking_tick
        )

    def __hash__(self):
        supercls_hash = super(PlateState, self).__hash__()
        return hash((supercls_hash, self._cooking_tick))

    def __repr__(self):
        supercls_str = super(PlateState, self).__repr__()
        return "{}\nRinse Count:\t{}".format(supercls_str, self._cooking_tick)

    def __str__(self):
        res = "{"
        if self.is_rinsing:
            res += str(self._cooking_tick)
        elif self.is_ready:
            res += str("✓")
        return res

    @ObjectState.position.setter
    def position(self, new_pos):
        self._position = new_pos

    @property
    def is_rinsing(self):
        return not self.is_idle and not self.is_ready

    @property
    def cook_time(self):
        # used mostly when cook time is supplied by state dict
        if self._cook_time is not None:
            return self._cook_time
        else:
            return 2

    def is_valid(self):
        return self.name in ["clean_plate", "dirty_plate"]

    @property
    def rinse_time_remaining(self):
        return max(0, self._cook_time - self._cooking_tick)

    @property
    def is_ready(self):
        if self.is_idle:
            return False
        return self._cooking_tick >= self._cook_time

    @property
    def is_idle(self):
        return self._cooking_tick < 0

    @property
    def is_full(self):
        return not self.is_idle

    def auto_finish(self):
        self._cooking_tick = 0
        self._cooking_tick = self.cook_time

    @IdObjectState.position.setter
    def position(self, new_pos):
        self._position = new_pos

    def begin_rinsing(self):
        if not self.is_idle:
            raise ValueError("Cannot begin rinse at this time")
        self._cooking_tick = 0

    def rinse(self):
        if self.is_idle:
            raise ValueError("Must begin rinsing before advancing rinse tick")
        if self.is_ready:
            raise ValueError("Cannot rinse a plate that is already done")
        self._cooking_tick += 1

    def deepcopy(self):
        return PlateState(
            self.id, self.name, self.position, self._cook_time, self._cooking_tick
        )

    def to_dict(self):
        info_dict = super(PlateState, self).to_dict()
        info_dict["rinse_count"] = self._cooking_tick
        info_dict["is_ready"] = self.is_ready
        info_dict["is_idle"] = self.is_idle
        info_dict["rinse_total"] = self._cook_time
        return info_dict

    @classmethod
    def from_dict(cls, obj_dict):
        obj_dict = copy.deepcopy(obj_dict)
        if obj_dict["name"] != "clean_plate" or obj_dict["name"] != "dirty_plate":
            return super(SoupState, cls).from_dict(obj_dict)

    # @classmethod
    # def get_plate(cls, position, rinse_total=2):
    #     return cls(position, rinse_total)


class SteakState(SoupState):
    def __init__(
        self, id, name, position, ingredients=[], cooking_tick=-1, cook_time=-1
    ):
        super(SteakState, self).__init__(position, ingredients)
        self.id = id
        self.name = name
        self._cooking_tick = cooking_tick
        self._cook_time = cook_time if cook_time > 0 else Burrito_Recipe._steak_time

    def __eq__(self, other):
        return (
            isinstance(other, SteakState)
            and self.name == other.name
            and self.position == other.position
            and self._cooking_tick == other._cooking_tick
            and all(
                [
                    this_i == other_i
                    for this_i, other_i in zip(self._ingredients, other._ingredients)
                ]
            )
        )

    def __hash__(self):
        ingredient_hash = hash(tuple([hash(i) for i in self._ingredients]))
        supercls_hash = super(SteakState, self).__hash__()
        return hash((supercls_hash, self._cooking_tick, ingredient_hash))

    def __repr__(self):
        supercls_str = super(SteakState, self).__repr__()
        ingredients_str = self._ingredients.__repr__()
        return "{}\nIngredients:\t{}\nCooking Tick:\t{}".format(
            supercls_str, ingredients_str, self._cooking_tick
        )

    def __str__(self):
        res = "{"
        # print("ingredients", self.ingredients)
        for ingredient in sorted(self.ingredients):
            res += Burrito_Recipe.STR_REP[ingredient]
        if self.is_cooking:
            res += str(self._cooking_tick)
        elif self.is_ready:
            res += str("✓")
        return res

    def is_valid(self):
        return self.name in ["steak", "chopped_steak", "fried_mushroom"]

    @IdObjectState.position.setter
    def position(self, new_pos):
        self._position = new_pos
        for ingredient in self._ingredients:
            ingredient.position = new_pos

    @property
    def ingredients(self):
        return [ingredient.name for ingredient in self._ingredients]

    @property
    def is_cooking(self):
        return not self.is_idle and not self.is_ready

    @property
    def cook_time(self):
        return self._cook_time

    @property
    def is_ready(self):
        if self.is_idle:
            return False
        return self._cooking_tick >= self._cook_time

    @property
    def is_idle(self):
        return self._cooking_tick < 0

    @property
    def is_full(self):
        return (
            not self.is_idle
            or len(self.ingredients) == Burrito_Recipe.MAX_NUM_INGREDIENTS
        )

    def auto_finish(self):
        if len(self.ingredients) == 0:
            raise ValueError("Cannot finish steak with no ingredients")
        self._cooking_tick = 0
        self._cooking_tick = self.cook_time

    def add_ingredient(self, ingredient):
        if not ingredient.name in Burrito_Recipe.ALL_INGREDIENTS:
            #print("Invalid ingredient", ingredient.name, "steak state")
            raise ValueError("Invalid ingredient")
        if self.is_full:
            raise ValueError("Reached maximum number of ingredients in recipe")
        ingredient.position = self.position
        self._ingredients.append(ingredient)

    def add_ingredient_from_str(self, ingredient_str):
        ingredient_obj = IdObjectState(None, ingredient_str, self.position)
        self.add_ingredient(ingredient_obj)

    def pop_ingredient(self):
        if not self.is_idle:
            raise ValueError("Cannot remove an ingredient from this steak at this time")
        if len(self._ingredients) == 0:
            raise ValueError("No ingredient to remove")
        return self._ingredients.pop()

    def begin_cooking(self):
        if not self.is_idle:
            raise ValueError("Cannot begin cooking this steak at this time")
        if len(self.ingredients) == 0:
            raise ValueError(
                "Must add at least one ingredient to steak before you can begin cooking"
            )
        self._cooking_tick = 0

    def cook(self):
        if self.is_idle:
            raise ValueError("Must begin cooking before advancing cook tick")
        if self.is_ready:
            raise ValueError("Cannot cook a soup that is already done")
        self._cooking_tick += 1

    @classmethod
    def get_steak(cls, position, num_meat=1, cooking_tick=-1, finished=False, **kwargs):
        if num_meat < 0:
            raise ValueError("Number of active ingredients must be positive")
        if num_meat > Recipe.MAX_NUM_INGREDIENTS:
            raise ValueError("Too many ingredients specified for steak")
        if cooking_tick >= 0 and num_meat == 0:
            raise ValueError("_cooking_tick must be -1 for empty grill")
        if finished and num_meat == 0:
            raise ValueError("Empty grill cannot be finished")
        meats = [ObjectState(Recipe.MEAT, position) for _ in range(num_meat)]
        ingredients = meats
        steak = cls(position, cooking_tick)
        if finished:
            steak.auto_finish()
        return steak

    def to_dict(self):
        info_dict = super(SteakState, self).to_dict()
        ingrdients_dict = [ingredient.to_dict() for ingredient in self._ingredients]
        info_dict["_ingredients"] = ingrdients_dict
        info_dict["cooking_tick"] = self._cooking_tick
        info_dict["is_cooking"] = self.is_cooking
        info_dict["is_ready"] = self.is_ready
        info_dict["is_idle"] = self.is_idle
        info_dict["cook_time"] = -1 if self.is_idle else self.cook_time

        # This is for backwards compatibility w/ overcooked-demo
        # Should be removed once overcooked-demo is updated to use 'cooking_tick' instead of '_cooking_tick'
        info_dict["_cooking_tick"] = self._cooking_tick
        return info_dict

    @classmethod
    def from_dict(cls, obj_dict):
        obj_dict = copy.deepcopy(obj_dict)
        if obj_dict["name"] != "steak":
            return super(SteakState, cls).from_dict(obj_dict)

        if "state" in obj_dict:
            # Legacy soup representation
            ingredient, num_ingredient, time = obj_dict["state"]
            cooking_tick = -1 if time == 0 else time
            finished = time >= Burrito_Recipe._steak_time
            return SteakState.get_steak(
                obj_dict["position"],
                num_meat=num_ingredient,
                cooking_tick=cooking_tick,
                finished=finished,
            )

        ingredients_objs = [
            IdObjectState.from_dict(ing_dict) for ing_dict in obj_dict["_ingredients"]
        ]
        obj_dict["ingredients"] = ingredients_objs
        return cls(**obj_dict)

    def deepcopy(self):
        return SteakState(
            self.id,
            self.name,
            self.position,
            [ingredient.deepcopy() for ingredient in self._ingredients],
            self._cooking_tick,
        )


class GarnishState(SoupState):
    def __init__(self, id, name, position, ingredients=[], chop_count=-1, chop_time=2):
        super(GarnishState, self).__init__(position, ingredients)
        self.id = id
        self.name = name
        self._cooking_tick = chop_count
        self._cook_time = chop_time

    def __eq__(self, other):
        return (
            isinstance(other, GarnishState)
            and self.name == other.name
            and self.position == other.position
            and self._cooking_tick == other._cooking_tick
            and all(
                [
                    this_i == other_i
                    for this_i, other_i in zip(self._ingredients, other._ingredients)
                ]
            )
        )

    def __hash__(self):
        ingredient_hash = hash(tuple([hash(i) for i in self._ingredients]))
        supercls_hash = super(GarnishState, self).__hash__()
        return hash((supercls_hash, self._cooking_tick, ingredient_hash))

    def __repr__(self):
        supercls_str = super(GarnishState, self).__repr__()
        ingredients_str = self._ingredients.__repr__()
        return "{}\nIngredients:\t{}\nCooking Tick:\t{}".format(
            supercls_str, ingredients_str, self._cooking_tick
        )

    def is_valid(self):
        return self.name in ["garnish"]

    def begin_chop(self):
        if not self.is_idle:
            raise ValueError("Cannot begin rinse at this time")
        self._cooking_tick = 0

    def chop(self):
        if self.is_ready:
            raise ValueError("Cannot cook a soup that is already done")
        self._cooking_tick += 1

    @IdObjectState.position.setter
    def position(self, new_pos):
        self._position = new_pos
        for ingredient in self._ingredients:
            ingredient.position = new_pos

    def add_ingredient_from_str(self, ingredient_str):
        ingredient_obj = IdObjectState(None, ingredient_str, self.position)
        self.add_ingredient(ingredient_obj)

    @classmethod
    def get_garnish(
        cls, position, num_onion=1, chop_count=-1, finished=False, **kwargs
    ):
        if num_onion < 0:
            raise ValueError("Number of active ingredients must be positive")
        if num_onion > Recipe.MAX_NUM_INGREDIENTS:
            raise ValueError("Too many ingredients specified for garnish")
        if chop_count >= 0 and num_onion == 0:
            raise ValueError("_chop_count must be -1 for empty board")
        if finished and num_onion == 0:
            raise ValueError("Empty board cannot be finished")
        # onions = [
        #     ObjectState(Recipe.ONION, position) for _ in range(num_onions)
        # ]
        # ingredients = onions
        garnish = cls(position, chop_count)
        if finished:
            garnish.auto_finish()
        return garnish

    def deepcopy(self):
        return GarnishState(
            self.id,
            self.name,
            self.position,
            [ingredient.deepcopy() for ingredient in self._ingredients],
            self._cooking_tick,
        )

    def to_dict(self):
        info_dict = super(GarnishState, self).to_dict()
        ingrdients_dict = [ingredient.to_dict() for ingredient in self._ingredients]
        info_dict["_ingredients"] = ingrdients_dict
        info_dict["cooking_tick"] = self._cooking_tick
        info_dict["is_cooking"] = self.is_cooking
        info_dict["is_ready"] = self.is_ready
        info_dict["is_idle"] = self.is_idle
        info_dict["cook_time"] = -1 if self.is_idle else self.cook_time

        # This is for backwards compatibility w/ overcooked-demo
        # Should be removed once overcooked-demo is updated to use 'cooking_tick' instead of '_cooking_tick'
        info_dict["_cooking_tick"] = self._cooking_tick
        return info_dict

    @classmethod
    def from_dict(cls, obj_dict):
        obj_dict = copy.deepcopy(obj_dict)
        if obj_dict["name"] != "garnish":
            return super(GarnishState, cls).from_dict(obj_dict)

        if "state" in obj_dict:
            # Legacy soup representation
            ingredient, num_ingredient, time = obj_dict["state"]
            cooking_tick = -1 if time == 0 else time
            finished = time >= 10
            return GarnishState.get_garnish(
                obj_dict["position"],
                num_onion=num_ingredient,
                cooking_tick=cooking_tick,
                finished=finished,
            )

        ingredients_objs = [
            IdObjectState.from_dict(ing_dict) for ing_dict in obj_dict["_ingredients"]
        ]
        obj_dict["ingredients"] = ingredients_objs
        return cls(**obj_dict)


class ChoppedSteakState(SoupState):
    def __init__(self, id, name, position, ingredients=[], chop_count=-1, chop_time=2):
        super(ChoppedSteakState, self).__init__(position, ingredients)
        self.id = id
        self.name = name
        self._cooking_tick = chop_count
        self._cook_time = chop_time

    def __eq__(self, other):
        return (
            isinstance(other, ChoppedSteakState)
            and self.name == other.name
            and self.position == other.position
            and self._cooking_tick == other._cooking_tick
            and all(
                [
                    this_i == other_i
                    for this_i, other_i in zip(self._ingredients, other._ingredients)
                ]
            )
        )

    def __hash__(self):
        ingredient_hash = hash(tuple([hash(i) for i in self._ingredients]))
        supercls_hash = super(ChoppedSteakState, self).__hash__()
        return hash((supercls_hash, self._cooking_tick, ingredient_hash))

    def __repr__(self):
        supercls_str = super(ChoppedSteakState, self).__repr__()
        ingredients_str = self._ingredients.__repr__()
        return "{}\nIngredients:\t{}\nCooking Tick:\t{}".format(
            supercls_str, ingredients_str, self._cooking_tick
        )

    def is_valid(self):
        return self.name in ["chopped_steak"]

    @IdObjectState.position.setter
    def position(self, new_pos):
        self._position = new_pos
        for ingredient in self._ingredients:
            ingredient.position = new_pos

    def add_ingredient_from_str(self, ingredient_str):
        ingredient_obj = IdObjectState(None, ingredient_str, self.position)
        self.add_ingredient(ingredient_obj)

    @classmethod
    def get_chopped_steak(
        cls, position, num_onion=1, chop_count=-1, finished=False, **kwargs
    ):
        if num_onion < 0:
            raise ValueError("Number of active ingredients must be positive")
        if num_onion > Recipe.MAX_NUM_INGREDIENTS:
            raise ValueError("Too many ingredients specified for garnish")
        if chop_count >= 0 and num_onion == 0:
            raise ValueError("_chop_count must be -1 for empty board")
        if finished and num_onion == 0:
            raise ValueError("Empty board cannot be finished")
        # onions = [
        #     ObjectState(Recipe.ONION, position) for _ in range(num_onions)
        # ]
        # ingredients = onions
        garnish = cls(position, chop_count)
        if finished:
            garnish.auto_finish()
        return garnish

    def deepcopy(self):
        return GarnishState(
            self.id,
            self.name,
            self.position,
            [ingredient.deepcopy() for ingredient in self._ingredients],
            self._cooking_tick,
        )

    def to_dict(self):
        info_dict = super(ChoppedSteakState, self).to_dict()
        ingrdients_dict = [ingredient.to_dict() for ingredient in self._ingredients]
        info_dict["_ingredients"] = ingrdients_dict
        info_dict["cooking_tick"] = self._cooking_tick
        info_dict["is_cooking"] = self.is_cooking
        info_dict["is_ready"] = self.is_ready
        info_dict["is_idle"] = self.is_idle
        info_dict["cook_time"] = -1 if self.is_idle else self.cook_time

        # This is for backwards compatibility w/ overcooked-demo
        # Should be removed once overcooked-demo is updated to use 'cooking_tick' instead of '_cooking_tick'
        info_dict["_cooking_tick"] = self._cooking_tick
        return info_dict

    @classmethod
    def from_dict(cls, obj_dict):
        obj_dict = copy.deepcopy(obj_dict)
        if obj_dict["name"] != "chopped_steak":
            return super(ChoppedSteakState, cls).from_dict(obj_dict)

        if "state" in obj_dict:
            # Legacy soup representation
            ingredient, num_ingredient, time = obj_dict["state"]
            cooking_tick = -1 if time == 0 else time
            finished = time >= 10
            return ChoppedSteakState.get_chopped_steak(
                obj_dict["position"],
                num_onion=num_ingredient,
                cooking_tick=cooking_tick,
                finished=finished,
            )

        ingredients_objs = [
            IdObjectState.from_dict(ing_dict) for ing_dict in obj_dict["_ingredients"]
        ]
        obj_dict["ingredients"] = ingredients_objs
        return cls(**obj_dict)


class ChoppedItemState(SoupState):
    def __init__(self, id, name, position, ingredients=[], chop_count=-1, chop_time=2):
        super(ChoppedItemState, self).__init__(position, ingredients)
        self.id = id
        self.name = name
        self._cooking_tick = chop_count
        self._cook_time = chop_time

    def __eq__(self, other):
        return (
            isinstance(other, ChoppedItemState)
            and self.name == other.name
            and self.position == other.position
            and self._cooking_tick == other._cooking_tick
            and all(
                [
                    this_i == other_i
                    for this_i, other_i in zip(self._ingredients, other._ingredients)
                ]
            )
        )

    def __hash__(self):
        ingredient_hash = hash(tuple([hash(i) for i in self._ingredients]))
        supercls_hash = super(ChoppedItemState, self).__hash__()
        return hash((supercls_hash, self._cooking_tick, ingredient_hash))

    def __repr__(self):
        supercls_str = super(ChoppedItemState, self).__repr__()
        ingredients_str = self._ingredients.__repr__()
        return "{}\nIngredients:\t{}\nCooking Tick:\t{}".format(
            supercls_str, ingredients_str, self._cooking_tick
        )

    def is_valid(self):
        return self.name in ["chopped_mushroom", "chopped_meat"]

    def begin_chop(self):
        if not self.is_idle:
            raise ValueError("Cannot begin rinse at this time")
        self._cooking_tick = 0

    def chop(self):
        if self.is_ready:
            raise ValueError("Cannot cook a soup that is already done")
        self._cooking_tick += 1

    @IdObjectState.position.setter
    def position(self, new_pos):
        self._position = new_pos
        for ingredient in self._ingredients:
            ingredient.position = new_pos

    def add_ingredient_from_str(self, ingredient_str):
        ingredient_obj = IdObjectState(None, ingredient_str, self.position)
        self.add_ingredient(ingredient_obj)

    @classmethod
    def get_mushroom(
        cls, position, num_onion=1, chop_count=-1, finished=False, **kwargs
    ):
        if num_onion < 0:
            raise ValueError("Number of active ingredients must be positive")
        if num_onion > Recipe.MAX_NUM_INGREDIENTS:
            raise ValueError("Too many ingredients specified for garnish")
        if chop_count >= 0 and num_onion == 0:
            raise ValueError("_chop_count must be -1 for empty board")
        if finished and num_onion == 0:
            raise ValueError("Empty board cannot be finished")
        # onions = [
        #     ObjectState(Recipe.ONION, position) for _ in range(num_onions)
        # ]
        # ingredients = onions
        garnish = cls(position, chop_count)
        if finished:
            garnish.auto_finish()
        return garnish

    def deepcopy(self):
        return ChoppedItemState(
            self.id,
            self.name,
            self.position,
            [ingredient.deepcopy() for ingredient in self._ingredients],
            self._cooking_tick,
        )

    def to_dict(self):
        info_dict = super(ChoppedItemState, self).to_dict()
        ingrdients_dict = [ingredient.to_dict() for ingredient in self._ingredients]
        info_dict["_ingredients"] = ingrdients_dict
        info_dict["cooking_tick"] = self._cooking_tick
        info_dict["is_cooking"] = self.is_cooking
        info_dict["is_ready"] = self.is_ready
        info_dict["is_idle"] = self.is_idle
        info_dict["cook_time"] = -1 if self.is_idle else self.cook_time

        # This is for backwards compatibility w/ overcooked-demo
        # Should be removed once overcooked-demo is updated to use 'cooking_tick' instead of '_cooking_tick'
        info_dict["_cooking_tick"] = self._cooking_tick
        return info_dict

    @classmethod
    def from_dict(cls, obj_dict):
        obj_dict = copy.deepcopy(obj_dict)
        if obj_dict["name"] != "chopped_mushroom" or obj_dict["name"] != "chopped_meat":
            return super(ChoppedItemState, cls).from_dict(obj_dict)

        if "state" in obj_dict:
            # Legacy soup representation
            ingredient, num_ingredient, time = obj_dict["state"]
            cooking_tick = -1 if time == 0 else time
            finished = time >= 10
            return ChoppedItemState.get_mushroom(
                obj_dict["position"],
                num_onion=num_ingredient,
                cooking_tick=cooking_tick,
                finished=finished,
            )

        ingredients_objs = [
            IdObjectState.from_dict(ing_dict) for ing_dict in obj_dict["_ingredients"]
        ]
        obj_dict["ingredients"] = ingredients_objs
        return cls(**obj_dict)


class BurritoState(OvercookedState):
    def __init__(
        self,
        players,
        objects,
        bonus_orders=[],
        all_orders=[],
        complete_orders=[],
        consecutive_orders=0,
        order_display_list=[],
        order_list=[],
        timestep=0,
        obj_count=0,
        max_plates=4,
        num_plates=0,
        next_recipe = 0,
        **kwargs,
    ):
        self.obj_count = obj_count
        all_orders = [Burrito_Recipe.from_dict(order) for order in all_orders]
        self._all_orders = all_orders
        # print("max_plates", max_plates)
        # print("num_plates", num_plates)
        for pos, obj in objects.items():
            assert obj.position == pos
        self.players = tuple(players)
        self.objects = objects
        self._bonus_orders = bonus_orders
        self.consecutive_orders = consecutive_orders
        self._complete_orders = complete_orders
        self._order_display_list = order_display_list
        self.order_list = order_list
        self.next_recipe = next_recipe
        self.timestep = timestep
        self.max_plates = max_plates
        self.num_plates = num_plates
        # assert len(set(self.bonus_orders)) == len(
        #     self.bonus_orders
        # ), "Bonus orders must not have duplicates"
        # assert len(set(self.all_orders)) == len(
        #     self.all_orders
        # ), "All orders must not have duplicates"
        # assert set(self.bonus_orders).issubset(
        #     set(self.all_orders)
        # ), "Bonus orders must be a subset of all orders"

    def deepcopy(self):
        return BurritoState(
            players=[player.deepcopy() for player in self.players],
            objects={pos: obj.deepcopy() for pos, obj in self.objects.items()},
            bonus_orders=[order for order in self._bonus_orders],
            all_orders=[order.to_dict() for order in self.all_orders],
            timestep=self.timestep,
            consecutive_orders=self.consecutive_orders,
            obj_count=self.obj_count,
            order_list=[order for order in self.order_list],
            order_display_list=[order for order in self._order_display_list],
            max_plates=4,
            num_plates=self.num_plates,
            next_recipe=self.next_recipe
        )

    def time_independent_equal(self, other):
        order_lists_equal = self.all_orders == other.all_orders

        return (
            isinstance(other, BurritoState)
            and self.players == other.players
            and set(self.objects.items()) == set(other.objects.items())
            and order_lists_equal
        )

    def to_dict(self):
        return {
            "players": [p.to_dict() for p in self.players],
            "objects": [obj.to_dict() for obj in self.objects.values()],
            "order_list": [order for order in self.order_list],
            "bonus_orders": [order for order in self.bonus_orders],
            "all_orders": [order.to_dict() for order in self.all_orders],
            "timestep": self.timestep,
            "num_plates": self.num_plates,
        }

    @property
    def all_orders(self):
        return (
            sorted(self._all_orders)
            if self._all_orders
            else sorted(Burrito_Recipe.ALL_RECIPES)
        )

    @property
    def curr_order(self):
        return self.order_list[0]

    @property
    def num_orders_remaining(self):
        return len(self.order_list)

    @classmethod
    def from_players_pos_and_or(
        cls,
        players_pos_and_or,
        bonus_orders=[],
        all_orders=[],
        order_list=[],
        order_display_list=[],
    ):
        """
        Make a dummy OvercookedState with no objects based on the passed in player
        positions and orientations and order list
        """
        return cls(
            [
                PlayerState(*player_pos_and_or)
                for player_pos_and_or in players_pos_and_or
            ],
            objects={},
            bonus_orders=bonus_orders,
            all_orders=all_orders,
            order_list=order_list,
            order_display_list=order_list,
        )

    @classmethod
    def from_player_positions(
        cls,
        player_positions,
        bonus_orders=[],
        all_orders=[],
        order_list=[],
        order_display_list=[],
    ):
        """
        Make a dummy OvercookedState with no objects and with players facing
        North based on the passed in player positions and order list
        """
        dummy_pos_and_or = [(pos, Direction.NORTH) for pos in player_positions]
        return cls.from_players_pos_and_or(
            dummy_pos_and_or, bonus_orders, all_orders, order_list, order_display_list
        )

    @staticmethod
    def from_dict(state_dict, obj_count=0):
        state_dict = copy.deepcopy(state_dict)

        state_dict["players"] = [
            PlayerState.from_dict(p) for p in state_dict["players"]
        ]
        object_list = [IdObjectState.from_dict(o) for o in state_dict["objects"]]
        state_dict["objects"] = {ob.position: ob for ob in object_list}

        return BurritoState(**state_dict, obj_count=obj_count)

    # TODO: probs delete this state stuff
    # state_dict = copy.deepcopy(state_dict)
    #     print("STATE DICT", state_dict)
    #     if state_dict.get("players"):
    #         state_dict["players"] = [
    #             PlayerState.from_dict(p) for p in state_dict["players"]
    #         ]
    #     if state_dict.get("objects"):
    #         object_list = [IdObjectState.from_dict(o) for o in state_dict["objects"]]
    #         state_dict["objects"] = {ob.position: ob for ob in object_list}
    #         obj_count = len(object_list)
    #     return BurritoState(**state_dict, obj_count=obj_count)

    # below methods ported from ICAROS qd-humans framework for RL agents
    # def print_player_workload(
    #     self,
    # ):
    #     for idx, player in enumerate(self.players):
    #         logger.info(f"Player {idx + 1}")
    #         player.print_workload()

    def get_player_workload(
        self,
    ):
        workloads = []
        for idx, player in enumerate(self.players):
            workloads.append(player.get_workload())
        return workloads

    def cal_concurrent_active_frequency(
        self,
    ):
        """Proportion of time in which both agents are active (in [0,1])"""
        concurrent_active_log = self.cal_concurrent_active_log()
        return np.mean(concurrent_active_log)

    def cal_concurrent_active_sum(
        self,
    ):
        concurrent_active_log = self.cal_concurrent_active_log()
        res = np.sum(concurrent_active_log)

        return res

    def cal_concurrent_active_log(
        self,
    ):
        active_logs = self.get_player_active_log()
        if len(active_logs[0]) == 0:
            return []

        return np.array(active_logs[0]) & np.array(active_logs[1])

    def get_player_active_log(
        self,
    ):
        active_log = []
        for idx, player in enumerate(self.players):
            active_log.append(player.active_log)
        return active_log

    def cal_mean_stuck_time(
        self,
    ):
        """Proportion of time in which both agents are stuck (in [0,1])"""
        stuck_logs = self.get_player_stuck_log()
        return np.mean(stuck_logs[0])

    def cal_total_stuck_time(
        self,
    ):
        stuck_logs = self.get_player_stuck_log()
        res = sum(stuck_logs[0])
        return res

    def get_player_stuck_log(
        self,
    ):
        stuck_log = []
        for idx, player in enumerate(self.players):
            stuck_log.append(player.stuck_log)
        return stuck_log

def get_multiplier(consecutive_orders, multiplier_params):
    """
    :param multiplier_params: [[a0, b0], [a1, b1], ..., [ak, bk]]
    :return:
        if consecutive_orders <= a0:    return b0
        elif consecutive_orders <= a1: return b1
        ...
        elif consecutive_orders <= ak: return bk (ak usually set to 100)
    
    default game multiplier_params: [[1,1], [4,2], [100,5]]
    
    good training multiplier_params: [[0,1], [1,10], [2,20], [3,30], [100,40]]
    """
    param_idx = 0
    while consecutive_orders > multiplier_params[param_idx][0]:
        param_idx += 1
    return multiplier_params[param_idx][1]

def generate_reward(order, consecutive_orders, multiplier_params):
    # Generate a reward for the current state
    # s0 = 2 + 20 * (n) + m * t
    # where 2 is the # of stations (pot + stove for each recipe)

    t = timeToTip(order[1], order[2])
    n = len(dishname2ingradient(order[0])["ingredients"])
    multipier = get_multiplier(consecutive_orders, multiplier_params)

    #print("CURRENT ORDER", order)

    #print("tip: ", t)
    #print("# ingredients: ", n)
    #print("consecutive_orders: ", consecutive_orders)
    #print("multipier: ", multipier)

    score = 20 + 20 * n + multipier * t

    #print()
    #print("score: ", score)

    return score, t, n, multipier

def timeToTip(time_remaining, total_time):
    if time_remaining >= total_time * 0.66:
        return 8
    elif time_remaining >= total_time * 0.33:
        return 5
    else:
        return 3

def dishname2ingradient(dish_name):
    # map dish_name to its ingredient, for example, steak_onion_dish to {"ingredients" : ["meat","onion"]},
    if dish_name == "steak_dish":
        return {"ingredients": ["meat"]}
    elif dish_name == "boiled_chicken_dish":
        return {"ingredients": ["chicken"]}
    elif dish_name == "steak_onion_dish":
        return {"ingredients": ["meat", "onion"]}
    elif dish_name == "boiled_chicken_onion_dish":
        return {"ingredients": ["chicken", "onion"]}
    elif dish_name == "steak_burrito_dish":
        return {"ingredients": ["chopped_steak", "rice", "tortilla"]}
    elif dish_name == "mushroom_burrito_dish":
        return {"ingredients": ["fried_mushroom", "rice", "tortilla"]}


def ingradient2dishname(ingradient):
    # map ingradient to its dish_name, for example, {"ingredients" : ["meat","onion"]} to steak_onion_dish
    if ingradient == ["meat"]:
        return "steak_dish"
    elif ingradient == ["chicken"]:
        return "boiled_chicken_dish"
    elif ingradient == ["meat", "onion"]:
        return "steak_onion_dish"
    elif ingradient == ["chicken", "onion"]:
        return "boiled_chicken_onion_dish"
    elif ingradient == ["chopped_steak", "rice", "tortilla"]:
        return "steak_burrito_dish"
    elif ingradient == ["fried_mushroom", "rice", "tortilla"]:
        return "mushroom_burrito_dish"


DISH_TYPES = [
    "steak_dish",
    "boiled_chicken_dish",
    "steak_onion_dish",
    "boiled_chicken_onion_dish",
    "steak_burrito_dish",
    "mushroom_burrito_dish",
]

EVENT_TYPES = [
    # Onion events
    "onion_pickup",
    "useful_onion_pickup",
    "onion_drop",
    "useful_onion_drop",
    "potting_onion",
    # Meat events
    "meat_pickup",
    "useful_meat_pickup",
    "meat_drop",
    "useful_meat_drop",
    # Beef events
    "chopped_meat_pickup",
    "useful_chopped_meat_pickup",
    "chopped_meat_drop",
    "useful_chopped_meat_drop",
    # chicken events,
    "chicken_pickup",
    "useful_chicken_pickup",
    "chicken_drop",
    "useful_chicken_drop",
    "potting_chicken",
    # mushroom events
    "mushroom_pickup",
    "useful_mushroom_pickup",
    "mushroom_drop",
    "useful_mushroom_drop",
    "potting_mushroom",
    # chopped mushroom events
    "chopped_mushroom_pickup",
    "useful_chopped_mushroom_pickup",
    "chopped_mushroom_drop",
    "useful_chopped_mushroom_drop",
    # rice events
    "rice_pickup",
    "useful_rice_pickup",
    "rice_drop",
    "useful_rice_drop",
    "potting_rice",
    # boiled rice events
    "boiled_rice-plate_pickup",
    "useful_boiled_rice-plate_pickup",
    "boiled_rice-plate_drop",
    "useful_boiled_rice-plate_drop",
    # tortilla events
    "tortilla_pickup",
    "useful_tortilla_pickup",
    "tortilla_drop",
    "useful_tortilla_drop",
    # tortilla-plate
    "tortilla-plate_pickup",
    "useful_tortilla-plate_pickup",
    "tortilla-plate_drop",
    "useful_tortilla-plate_drop",
    # fire_ext events
    "fire_ext_pickup",
    "useful_fire_ext_pickup",
    "fire_ext_drop",
    "useful_fire_ext_drop",
    "extinguish_fire",
    # charcoal events
    "charcoal_pickup",
    "useful_charcoal_pickup",
    "charcoal_drop",
    "useful_charcoal_drop",
    # charcoal-plate events
    "charcoal-plate_pickup",
    "useful_charcoal-plate_pickup",
    "charcoal-plate_drop",
    "useful_charcoal-plate_drop",
    # choped_steak-plate events
    "chopped_steak-plate_pickup",
    "useful_chopped_steak-plate_pickup",
    "chopped_steak-plate_drop",
    "useful_chopped_steak-plate_drop",
    # chopped_steak-rice-plate events
    "chopped_steak-boiled_rice-plate_pickup",
    "useful_chopped_steak-boiled_rice-plate_pickup",
    "chopped_steak-boiled_rice-plate_drop",
    "useful_chopped_steak-boiled_rice-plate_drop",
    # fried_mushroom-rice-plate events
    "fried_mushroom-boiled_rice-plate_pickup",
    "useful_fried_mushroom-boiled_rice-plate_pickup",
    "fried_mushroom-boiled_rice-plate_drop",
    "useful_fried_mushroom-boiled_rice-plate_drop",
    # boiled_rice-tortilla-plate events
    "boiled_rice-tortilla-plate_pickup",
    "useful_boiled_rice-tortilla-plate_pickup",
    "boiled_rice-tortilla-plate_drop",
    "useful_boiled_rice-tortilla-plate_drop",
    # cooked mushroom event
    "fried_mushroom-plate_pickup",
    "fried_useful_mushroom-plate_pickup",
    "fried_mushroom-plate_drop",
    "fried_useful_mushroom-plate_drop",
    # fried mushroom tortilla
    "fried_mushroom-tortilla-plate_pickup",
    "useful_fried_mushroom-tortilla-plate_pickup",
    "fried_mushroom-tortilla-plate_drop",
    "useful_fried_mushroom-tortilla-plate_drop",
    # chopped steak tortilla events
    "chopped_steak-tortilla-plate_pickup",
    "useful_chopped_steak-tortilla-plate_pickup",
    "chopped_steak-tortilla-plate_drop",
    "useful_chopped_steak-tortilla-plate_drop",
    # chopped steak events
    "chopped_steak_pickup",
    "useful_chopped_steak_pickup",
    "chopped_steak_drop",
    "useful_chopped_steak_drop",
    # Dish events
    "useful_steak_pickup",
    "useful_steak_drop",
    "steak_cooking",
    "fried_mushroom_cooking",
    "chopped_steak_cooking",
    "dish_pickup",
    "steak_pickup",
    "boiled_chicken_pickup",
    "boiled_chicken_drop",
    "useful_boiled_chicken_pickup",
    "useful_dish_pickup",
    "dish_drop",
    "steak_drop",
    "boiled_chicken_onion_drop",
    "mushroom_burrito_drop",
    "mushroom_burrito_pickup",
    "steak_burrito_drop",
    "steak_burrito_pickup",
    "useful_dish_drop",
    "useful_steak_drop",
    "useful_boiled_chicken_drop",
    "dish_delivery",
    "correct_dish_delivery",
    "in_order_dish_delivery",
    "steak_onion_pickup",
    "boiled_chicken_onion_pickup",
    "useful_steak_onion_pickup",
    "useful_boiled_chicken_onion_pickup",
    "steak_onion_drop",
    "boiled_onion_drop",
    "useful_steak_onion_drop",
    "useful_boiled_chicken_onion_drop",
    "mushroom_burrito_dish_pickup",
    "steak_burrito_dish_pickup",
    "steak_onion_dish_delivery",
    "boiled_chicken_onion_delivery",
    "mushroom_burrito_dish_delivery" "steak_burrito_dish_delivery",
    "steak_dish_delivery",
    "boiled_chicken_delivery",
    # Soup events
    "soup_pickup",
    "soup_delivery",
    "soup_drop",
    # Potting events
    "optimal_onion_potting",
    "optimal_tomato_potting",
    "viable_onion_potting",
    "viable_tomato_potting",
    "catastrophic_onion_potting",
    "catastrophic_tomato_potting",
    "useless_onion_potting",
    "useless_tomato_potting",
    # Chopping events
    "chop_onion",
    "onion_chopping",
    "meat_chopping",
    "mushroom_chopping",
    # Rinsing events
    "plate_rinsing",
    "dirty_plate_drop",
    "dirty_plate_pickup",
    "rinse_dirty_plate",
    "clean_plate_pickup",
    "useful_clean_plate_pickup",

    "object_in_trash",
    "fried_mushroom-plate_grill_pickup",
    "fried_mushroom-tortilla-plate_grill_pickup",
    "fried_mushroom-boiled_rice-plate_grill_pickup",
    "mushroom_burrito_grill_pickup",

    "chopped_steak-plate_grill_pickup",
    "chopped_steak-tortilla-plate_grill_pickup",
    "chopped_steak-boiled_rice-plate_grill_pickup",
    "steak_burrito_grill_pickup",

    "boiled_rice-plate_pot_pickup",
    "chopped_steak-boiled_rice-plate_pot_pickup",
    "fried_mushroom-boiled_rice-plate_pot_pickup",
    "boiled_rice-tortilla-plate_pot_pickup",
    "mushroom_burrito_pot_pickup",
    "steak_burrito_pot_pickup",
]

REW_SHAPING_PARAMS = {
    "place_tortilla_on_plate": 0, #[0,0], #2,
    "pickup_dish_from_sink": 0, #[0,0], #2,
    "pickup_chopped_ingredient_from_chopboard": [0,0], #2,
    "place_ingredient_on_grill": 1.5, #3,
    "pickup_cooked_ingredient_from_grill": 3, #10,
    "place_ingredient_on_pot": 1.5, #2,
    "pickup_cooked_ingredient_from_pot": 3, #10,
    "extinguish_fire": 1, #2,
    "place_charcoal_on_trash": 1, #2,
    "burn_ingredient": 0, #-0.01, #-0.1, # per game tick
    "wash_dish": 1, #2,
    "pickup_tortilla": 0,
    "pickup_rice": 0,
    "pickup_meat": 0,
    "pickup_mushroom": 0,
    "chop_ingredient": 1,
    "food_assembly": 5, #10
    "serve_dish": 10,
    "wrong_dish_discount": 1,
    "correct_dish_bonus": 1,
    "multiplier_params": [[0,1], [1,10], [2,20], [3,30], [100, 40]],
    "random_dish_delivery": 40
}

class BurritoGridworld(OvercookedGridworld):
    def __init__(
        self,
        terrain,
        start_player_positions,
        start_state=None,
        start_all_orders=None,
        order_list=None,
        all_recipes=None,
        bonus_list=None,
        consecutive_orders=0,
        cook_time=10,
        num_items_for_steak=1,
        num_items_for_chicken=1,
        num_items_for_burrito=3,
        num_items_for_soup=3,
        chop_time=3,
        in_order_delivery_reward=10,
        delivery_reward=5,
        rew_shaping_params=None,
        layout_name="unnamed_layout",
        object_id_dict=None,
        **kwargs,
    ):
        super().__init__(
            terrain=terrain,
            start_player_positions=start_player_positions,
            start_all_orders=start_all_orders,
            cook_time=cook_time,
            num_items_for_soup=num_items_for_soup,
            delivery_reward=delivery_reward,
            rew_shaping_params=rew_shaping_params,
            layout_name=layout_name,
        )
        self.steak_cook_time = cook_time
        self.chop_time = chop_time
        self.object_id_dict = object_id_dict
        self.num_items_for_steak = num_items_for_steak
        self.num_items_for_chicken = num_items_for_chicken
        self.num_items_for_burrito = num_items_for_burrito
        self.order_list = order_list
        self.next_order = -1
        self.consecutive_orders = consecutive_orders
        self.all_recipes = all_recipes
        self.delivery_reward = delivery_reward
        self.in_order_delivery_reward = in_order_delivery_reward
        self.start_state = start_state
        self._configure_burrito_recipes(
            start_all_orders, num_items_for_chicken, num_items_for_steak, **kwargs
        )
        self.start_all_orders = (
            [r.to_dict() for r in Burrito_Recipe.ALL_RECIPES]
            if not start_all_orders
            else start_all_orders
        )
        self.reward_shaping_params = (
            REW_SHAPING_PARAMS
            if rew_shaping_params is None
            else rew_shaping_params
        )
        for rew in self.reward_shaping_params:
            if type(self.reward_shaping_params[rew]) == list:
                # individual shaping terms
                continue
            if rew in ['multiplier_params', 'random_dish_delivery', 'burn_ingredient']:
                # global shaping terms
                continue
            else:
                # make compatible with individual shaping terms
                val = self.reward_shaping_params[rew]
                self.reward_shaping_params[rew] = [val for _ in range(self.num_players)]

    @staticmethod
    def from_layout_name(layout_name, **params_to_overwrite):
        """
        Generates a OvercookedGridworld instance from a layout file.

        One can overwrite the default mdp configuration using partial_mdp_config.
        """
        params_to_overwrite = params_to_overwrite.copy()
        base_layout_params = read_layout_dict(layout_name) # customized read layout dict, from our src/data/layout

        grid = base_layout_params["grid"]
        del base_layout_params["grid"]
        base_layout_params["layout_name"] = layout_name

        if "start_state" in base_layout_params:
            base_layout_params["start_state"] = BurritoState.from_dict(
                base_layout_params["start_state"]
            )

        # Clean grid
        grid = [layout_row.strip() for layout_row in grid.split("\n")]
        return BurritoGridworld.from_grid(
            grid, base_layout_params, params_to_overwrite
        )

    @staticmethod
    def _assert_valid_grid(grid):
        """Raises an AssertionError if the grid is invalid.

        grid:  A sequence of sequences of spaces, representing a grid of a
        certain height and width. grid[y][x] is the space at row y and column
        x. A space must be either 'X' (representing a counter), ' ' (an empty
        space), 'O' (onion supply), 'P' (pot), 'D' (dish supply), 'S' (serving
        location), '1' (player 1) and '2' (player 2).
        """
        height = len(grid)
        width = len(grid[0])

        # Make sure the grid is not ragged
        assert all(len(row) == width for row in grid), "Ragged grid"

        # Borders must not be free spaces
        def is_not_free(c):
            return c in "XOPURZDCWBSGT"

        # for y in range(height):
        #     assert is_not_free(grid[y][0]), "Left border must not be free"
        #     assert is_not_free(grid[y][-1]), "Right border must not be free"
        # for x in range(width):
        #     assert is_not_free(grid[0][x]), "Top border must not be free"
        #     assert is_not_free(grid[-1][x]), "Bottom border must not be free"

        all_elements = [element for row in grid for element in row]
        digits = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
        layout_digits = [e for e in all_elements if e in digits]
        num_players = len(layout_digits)
        assert num_players > 0, "No players (digits) in grid"
        layout_digits = list(sorted(map(int, layout_digits)))
        assert layout_digits == list(
            range(1, num_players + 1)
        ), "Some players were missing"
        # TODO: change this to allow more terrain, inherite.
        print_all_elements = "".join(all_elements)
        #print(print_all_elements)
        assert all(
            c in "XOJKPDRSUTWBMCGZ123456789 " for c in all_elements
        ), "Invalid character in grid"
        assert all_elements.count("1") == 1, "'1' must be present exactly once"
        # assert all_elements.count("D") >= 1, "'D' must be present at least once"
        # assert all_elements.count("S") >= 1, "'S' must be present at least once"
        # assert all_elements.count("P") >= 1, "'P' must be present at least once"
        # assert (
        #     all_elements.count("G") >= 1
        # ), "'G' must be present at least once"
        # assert (
        #     all_elements.count("M") >= 1
        # ), "'M' must be present at least once"

    @staticmethod
    def from_grid(
        layout_grid, base_layout_params={}, params_to_overwrite={}, debug=True
    ):
        """
        Returns instance of OvercookedGridworld with terrain and starting
        positions derived from layout_grid.
        One can override default configuration parameters of the mdp in
        partial_mdp_config.
        """
        mdp_config = copy.deepcopy(base_layout_params)

        layout_grid = [[c for c in row] for row in layout_grid]
        BurritoGridworld._assert_valid_grid(layout_grid)

        if "layout_name" not in mdp_config:
            layout_name = "|".join(["".join(line) for line in layout_grid])
            mdp_config["layout_name"] = layout_name

        player_positions = [None] * 9
        for y, row in enumerate(layout_grid):
            for x, c in enumerate(row):
                if c in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                    layout_grid[y][x] = " "

                    # -1 is to account for fact that player indexing starts from 1 rather than 0
                    assert (
                        player_positions[int(c) - 1] is None
                    ), "Duplicate player in grid"
                    player_positions[int(c) - 1] = (x, y)

        num_players = len([x for x in player_positions if x is not None])
        player_positions = player_positions[:num_players]

        # After removing player positions from grid we have a terrain mtx
        mdp_config["terrain"] = layout_grid
        mdp_config["start_player_positions"] = player_positions

        for k, v in params_to_overwrite.items():
            curr_val = mdp_config.get(k, None)
            if debug:
                print(
                    "Overwriting mdp layout standard config value {}:{} -> {}".format(
                        k, curr_val, v
                    )
                )
            mdp_config[k] = v

        return BurritoGridworld(**mdp_config)

    def _configure_burrito_recipes(
        self, start_all_orders, num_items_for_chicken, num_items_for_steak, **kwargs
    ):
        self.recipe_config = {
            "num_items_for_chicken": num_items_for_chicken,
            "num_items_for_steak": num_items_for_steak,
            "all_orders": start_all_orders,
            **kwargs,
        }
        Burrito_Recipe.configure(self.recipe_config)

    #####################
    # BASIC CLASS UTILS #
    #####################

    def __eq__(self, other):
        return (
            np.array_equal(self.terrain_mtx, other.terrain_mtx)
            and self.start_player_positions == other.start_player_positions
            and self.start_all_orders == other.start_all_orders
            and self.steak_cook_time == other.steak_cook_time
            and self.delivery_reward == other.delivery_reward
            and self.in_order_delivery_reward == other.in_order_delivery_reward
            and self.reward_shaping_params == other.reward_shaping_params
            and self.layout_name == other.layout_name
        )

    def copy(self):
        return BurritoGridworld(
            terrain=self.terrain_mtx.copy(),
            start_player_positions=self.start_player_positions,
            start_all_orders=(
                None if self.start_all_orders is None else list(self.start_all_orders)
            ),
            cook_time=self.steak_cook_time,
            delivery_reward=self.delivery_reward,
            in_order_delivery_reward=self.in_order_delivery_reward,
            rew_shaping_params=copy.deepcopy(self.reward_shaping_params),
            layout_name=self.layout_name,
            object_id_dict=copy.deepcopy(self.object_id_dict),
        )

    @property
    def mdp_params(self):
        return {
            "layout_name": self.layout_name,
            "terrain": self.terrain_mtx,
            "start_player_positions": self.start_player_positions,
            "start_all_orders": self.start_all_orders,
            #"cook_time": self.soup_cook_time,
            "delivery_reward": self.delivery_reward,
            "in_order_delivery_reward": self.in_order_delivery_reward,
            "rew_shaping_params": copy.deepcopy(self.reward_shaping_params),
        }

    ##############
    # GAME LOGIC #
    ##############
    def get_standard_start_state(self):
        if self.start_state:
            # self.start_state.players = players
            self.start_state.order_list = self.order_list
            self.start_state.all_recipes = self.all_recipes
            return self.start_state
        start_state = BurritoState.from_player_positions(
            self.start_player_positions,
            all_orders=self.start_all_orders,
            order_list=self.order_list,
        )

        return start_state
    
    def get_random_start_position_fn(self):
        """
        Randomize player position on initialization of each game

        Currently only swap agents initial positions
        """
        def start_state_fn():
            valid_positions = self.get_valid_joint_player_positions()
            #start_pos = valid_positions[np.random.choice(len(valid_positions))]
            start_pos = []
            for x in range(len(self.start_player_positions)):
                pos = valid_positions[np.random.choice(len(valid_positions))]
                start_pos.append(pos)
                valid_positions.remove(pos)    
            np.random.shuffle(start_pos)
            dummy_pos_and_or = [(pos, Direction.NORTH) for pos in start_pos]
            players = [PlayerState(*player_pos_and_or) for player_pos_and_or in dummy_pos_and_or]
            if self.start_state:
                start_state = self.start_state.deepcopy()
                start_state.players = players
            else:
                """
                If not using start_state loaded from layout, need to specify objects states such as dirty dishes...
                """
                start_state = BurritoState.from_player_positions(start_pos,
                                                                    all_orders=self.start_all_orders,
                                                                    order_list=self.order_list)
            return start_state
        return start_state_fn

    def get_valid_joint_player_positions(self):
        valid_positions = []
        for y in range(len(self.terrain_mtx)):
            for x in range(len(self.terrain_mtx[y])):
                if self.terrain_mtx[y][x] == ' ':
                    valid_positions.append((x,y))
        return valid_positions

    def get_state_transition(
        self, state, joint_action, display_phi=False, motion_planner=None
    ):
        """Gets information about possible transitions for the action.

        Returns the next state, sparse reward and reward shaping.
        Assumes all actions are deterministic.

        NOTE: Sparse reward is given only when soups are delivered,
        shaped reward is given only for completion of subgoals
        (not soup deliveries).
        """
        events_infos = {event: [False] * self.num_players for event in EVENT_TYPES}
        assert not self.is_terminal(
            state
        ), "Trying to find successor of a terminal state: {}".format(state)

        for action, action_set in zip(joint_action, self.get_actions(state)):
            if action not in action_set:
                raise ValueError("Illegal action %s in state %s" % (action, state))

        new_state = state.deepcopy()
        # Resolve interacts first
        (
            sparse_reward_by_agent,
            shaped_reward_by_agent,
        ) = self.resolve_interacts(new_state, joint_action, events_infos)
        assert new_state.player_positions == state.player_positions
        assert new_state.player_orientations == state.player_orientations

        # Resolve player movements
        self.resolve_movement(new_state, joint_action)

        # Finally, environment effects
        env_effect_rew = self.step_environment_effects(new_state)

        # NOTE: For now , this is just to include the burnt item penalty 
        for idx in range(len(shaped_reward_by_agent)):
            shaped_reward_by_agent[idx] += env_effect_rew

        # Additional dense reward logic
        # shaped_reward += self.calculate_distance_based_shaped_reward(state, new_state)
        infos = {
            "event_infos": events_infos,
            "sparse_reward_by_agent": sparse_reward_by_agent,
            "shaped_reward_by_agent": shaped_reward_by_agent,
        }
        if display_phi:
            assert (
                motion_planner is not None
            ), "motion planner must be defined if display_phi is true"
            infos["phi_s"] = self.potential_function(state, motion_planner)
            infos["phi_s_prime"] = self.potential_function(new_state, motion_planner)
        return new_state, infos

    def resolve_interacts(self, new_state, joint_action, events_infos, rollout=True):
        """
        Resolve any INTERACT actions, if present.

        Currently if two players both interact with a terrain, we resolve player 1's interact
        first and then player 2's, without doing anything like collision checking.
        """
        pot_states = self.get_pot_states(new_state)
        # We divide reward by agent to keep track of who contributed
        sparse_reward, shaped_reward = [0.0] * self.num_players, [0.0] * self.num_players

        for player_idx, (player, action) in enumerate(
            zip(new_state.players, joint_action)
        ):
            pos, o = player.position, player.orientation
            i_pos = Action.move_in_direction(pos, o)
            # penalize agents from crossing to the other side of the map
#            if player_idx == 0 and i_pos[0] > 5:
#                shaped_reward[player_idx] -= 0.1
#            elif player_idx == 1 and i_pos[0] < 5:
#                shaped_reward[player_idx] -= 0.1

            if action != Action.INTERACT:
                continue
            terrain_type = self.get_terrain_type_at_pos(i_pos)
            if not rollout:
                obj_count = len(self.object_id_dict)
            else:
                obj_count = new_state.obj_count
            # NOTE: we always log pickup/drop before performing it, as that's
            # what the logic of determining whether the pickup/drop is useful assumes
            if terrain_type == "X":
                if player.has_object() and not new_state.has_object(i_pos):
                    obj_name = player.get_object().name
                    self.log_object_drop(
                        events_infos, new_state, obj_name, pot_states, player_idx
                    )

                    # Drop object on counter
                    obj = player.remove_object()
                    new_state.add_object(obj, i_pos)

                elif not player.has_object() and new_state.has_object(i_pos):
                    obj_name = new_state.get_object(i_pos).name

                    self.log_object_pickup(
                        events_infos, new_state, obj_name, pot_states, player_idx
                    )

                    # Pick up object from counter
                    obj = new_state.remove_object(i_pos)
                    #print(obj)
                    player.set_object(obj)

                elif player.has_object() and new_state.has_object(i_pos):
                    # special cases:
                    obj_name = player.get_object().name
                    #print("player holding", obj_name)
                    #print("object at counter", new_state.get_object(i_pos).name)
                    object_actions = {
                        "clean_plate": {"tortilla": "tortilla-plate"},
                        "boiled_rice-plate": {"tortilla": "boiled_rice-tortilla-plate"},
                        "chopped_steak-boiled_rice-plate": {
                            "tortilla": "steak_burrito"
                        },
                        "fried_mushroom-boiled_rice-plate": {
                            "tortilla": "mushroom_burrito"
                        },
                        "tortilla": {"clean_plate": "tortilla-plate"},
                        "chopped_steak-plate": {
                            "tortilla": "chopped_steak-tortilla-plate"
                        },
                        "fried_mushroom-plate": {
                            "tortilla": "fried_mushroom-tortilla-plate"
                        },
                    }
                    added_to_dish = False
                    if obj_name in object_actions:
                        target_obj = new_state.get_object(i_pos).name
                        if target_obj in object_actions[obj_name]:
                            new_obj = IdObjectState(
                                obj_count, object_actions[obj_name][target_obj], i_pos
                            )
                            new_state.remove_object(i_pos)
                            player.remove_object()
                            player.set_object(new_obj)
                            added_to_dish = True

                    if not added_to_dish:
                        player_obj = player.remove_object()

                        # Pick up object from counter
                        self.log_object_pickup(
                            events_infos, new_state, obj_name, pot_states, player_idx
                        )
                        obj = new_state.remove_object(i_pos)
                        player.set_object(obj)

                        # Drop object on counter
                        self.log_object_drop(
                            events_infos, new_state, obj_name, pot_states, player_idx
                        )
                        new_state.add_object(player_obj, i_pos)

            elif terrain_type == "O" and player.held_object is None:
                # Onion pickup from dispenser
                self.log_object_pickup(
                    events_infos, new_state, "onion", pot_states, player_idx
                )
                new_o_id = obj_count
                o = IdObjectState(new_o_id, "onion", pos)
                if not rollout:
                    self.object_id_dict[new_o_id] = o
                obj_count += 1
                player.set_object(o)
                # player.num_ingre_held += 1

            elif terrain_type == "T":
                obj_actions = {
                    "boiled_rice-plate": "boiled_rice-tortilla-plate",
                    "chopped_steak-plate": "chopped_steak-tortilla-plate",
                    "fried_mushroom-plate": "fried_mushroom-tortilla-plate",
                    "chopped_steak-boiled_rice-plate": "steak_burrito",
                    "fried_mushroom-boiled_rice-plate": "mushroom_burrito",
                }
                if player.held_object is None:
                    # Tortilla pickup from dispenser
                    self.log_object_pickup(
                        events_infos, new_state, "tortilla", pot_states, player_idx
                    )
                    new_o_id = obj_count
                    o = IdObjectState(new_o_id, "tortilla", pos)
                    if not rollout:
                        self.object_id_dict[new_o_id] = o
                    obj_count += 1
                    player.set_object(o)
                    if "pickup_tortilla" in self.reward_shaping_params:
                        shaped_reward[player_idx] += self.reward_shaping_params["pickup_tortilla"][player_idx]
                else:
                    #print("Player has object: ", player.get_object().name)
                    held_obj = player.get_object().name
                    new_obj_name = None
                    if held_obj == "clean_plate":
                        new_obj_name = "tortilla-plate"
                    elif held_obj in ["fried_mushroom", "chopped_steak"]:
                        new_obj_name = f"{held_obj}-tortilla-plate"
                    elif held_obj in obj_actions:
                        new_obj_name = obj_actions[held_obj]
                    if new_obj_name:
                        player.remove_object()
                        new_obj = IdObjectState(obj_count, new_obj_name, pos)
                        if not rollout:
                            self.object_id_dict[new_o_id] = o
                        obj_count += 1
                        player.set_object(new_obj)
                        # TODO: update these rewards
                        #shaped_reward[player_idx] += self.reward_shaping_params[
                        #    "SOUP_PICKUP_REWARD"
                        #]
                        if "place_tortilla_on_plate" in self.reward_shaping_params:
                            shaped_reward[player_idx] += self.reward_shaping_params["place_tortilla_on_plate"][player_idx]

            elif terrain_type == "Z" and player.held_object is None:
                # mushroom pickup from dispenser
                self.log_object_pickup(
                    events_infos, new_state, "mushroom", pot_states, player_idx
                )
                new_o_id = obj_count
                o = IdObjectState(new_o_id, "mushroom", pos)
                if not rollout:
                    self.object_id_dict[new_o_id] = o
                obj_count += 1
                player.set_object(o)
                if "pickup_mushroom" in self.reward_shaping_params:
                    shaped_reward[player_idx] += self.reward_shaping_params["pickup_mushroom"][player_idx]

            elif terrain_type == "M" and player.held_object is None:
                # meat pickup from dispenser
                self.log_object_pickup(
                    events_infos, new_state, "meat", pot_states, player_idx
                )
                new_o_id = obj_count
                o = IdObjectState(new_o_id, "meat", pos)
                if not rollout:
                    self.object_id_dict[new_o_id] = o
                obj_count += 1
                player.set_object(o)
                # player.num_ingre_held += 1
                if "pickup_meat" in self.reward_shaping_params:
                    shaped_reward[player_idx] += self.reward_shaping_params["pickup_meat"][player_idx]

            elif (
                terrain_type == "C" and player.held_object is None
            ):  # chicken pickup from dispenser
                self.log_object_pickup(
                    events_infos, new_state, "chicken", pot_states, player_idx
                )

                new_o_id = obj_count
                o = IdObjectState(new_o_id, "chicken", pos)
                if not rollout:
                    self.object_id_dict[new_o_id] = o
                obj_count += 1
                player.set_object(o)
                # player.num_ingre_held += 1
            elif (
                terrain_type == "R" and player.held_object is None
            ):  # chicken pickup from dispenser
                self.log_object_pickup(
                    events_infos, new_state, "rice", pot_states, player_idx
                )

                new_o_id = obj_count
                o = IdObjectState(new_o_id, "rice", pos)
                if not rollout:
                    self.object_id_dict[new_o_id] = o
                obj_count += 1
                player.set_object(o)
                # player.num_ingre_held += 1
                if "pickup_rice" in self.reward_shaping_params:
                    shaped_reward[player_idx] += self.reward_shaping_params["pickup_rice"][player_idx]

            elif terrain_type == "D" and player.held_object is None:
                self.log_object_pickup(
                    events_infos, new_state, "dirty_plate", pot_states, player_idx
                )
                # player.num_dirty_plate_held += 1

                # Give shaped reward if pickup is useful
                # if self.is_dirty_plate_pickup_useful(new_state, pot_states):
                #     shaped_reward[player_idx] += self.reward_shaping_params[
                #         "DIRTY_PLATE_PICKUP_REWARD"]

                # Perform dirty plate pickup from dispenser
                if new_state.num_plates > 0:
                    new_o_id = obj_count
                    o = PlateState(new_o_id, "dirty_plate", pos)
                    if not rollout:
                        self.object_id_dict[new_o_id] = o
                    obj_count += 1
                    player.set_object(o)
                    new_state.num_plates -= 1

            elif terrain_type == "W":
                if player.held_object is None:
                    # pick up clean plates
                    if self.plate_clean_at_location(new_state, i_pos):
                        self.log_object_pickup(
                            events_infos,
                            new_state,
                            "clean_plate",
                            pot_states,
                            player_idx,
                        )
                        obj = new_state.remove_object(i_pos)
                        player.set_object(obj)
                        # Give shaped reward if pickup is useful
                        # if self.is_dirty_plate_pickup_useful(new_state, pot_states):
                        # shaped_reward[player_idx] += self.reward_shaping_params["DIRTY_PLATE_PICKUP_REWARD"]

                        # player.num_dirty_plate_held += 1
                        if "pickup_dish_from_sink" in self.reward_shaping_params:
                            shaped_reward[player_idx] += self.reward_shaping_params["pickup_dish_from_sink"][player_idx]
                    # rinse dirty plates
                    else:
                        if new_state.has_object(i_pos):
                            obj = new_state.get_object(i_pos)
                            if not obj.is_ready:
                                obj.rinse()
                                if "wash_dish" in self.reward_shaping_params:
                                    shaped_reward[player_idx] += self.reward_shaping_params["wash_dish"][player_idx]

                                events_infos["plate_rinsing"][player_idx] = True
                                if obj.is_ready:
                                    new_state.remove_object(i_pos)
                                    obj_count += 1
                                    new_o_id = obj_count
                                    obj = IdObjectState(new_o_id, "clean_plate", i_pos)
                                    new_state.add_object(obj, i_pos)


                else:  # sink is empty and put dirty plate
                    if (
                        player.get_object().name == "dirty_plate"
                        and not new_state.has_object(i_pos)
                    ):
                        obj_name = player.get_object().name
                        self.log_object_drop(
                            events_infos, new_state, obj_name, pot_states, player_idx
                        )

                        # Drop object on counter
                        obj = player.remove_object()

                        new_o_id = obj_count
                        obj = PlateState(new_o_id, "dirty_plate", i_pos)
                        if not rollout:
                            self.object_id_dict[new_o_id] = obj
                        obj_count += 1
                        obj.begin_rinsing()
                        # print("begin rinsing", new_obj,new_state)
                        new_state.add_object(obj, i_pos)  # rinse time = 0

            elif terrain_type == "U" and player.held_object is not None:  # trash can
                player_obj = player.get_object().name
                ingredients = {
                    "meat": True,
                    "chopped_meat": True,
                    "chopped_mushroom": True,
                    "mushroom": True,
                    "tortilla": True,
                    "rice": True,
                    "fire_ext": False,
                    "charcoal": True,
                }
                plates = {"clean_plate": True, "dirty_plate": True}
                if player_obj not in ingredients and player_obj not in plates:
                    new_o_id = obj_count
                    new_obj = IdObjectState(new_o_id, "dirty_plate", i_pos)
                    if not rollout:
                        self.object_id_dict[new_o_id] = new_obj
                    obj_count += 1
                    player.remove_object()
                    player.set_object(new_obj)
                    events_infos["object_in_trash"][
                                        player_idx
                                    ] = True

                elif player_obj not in plates:
                    if player_obj == "charcoal":
                        if "place_charcoal_on_trash" in self.reward_shaping_params:
                            shaped_reward[player_idx] += self.reward_shaping_params["place_charcoal_on_trash"][player_idx]
                    if ingredients[player_obj]: # prevent throwing away fire ext
                        self.log_object_drop(
                            events_infos, new_state, player_obj, pot_states, player_idx
                        )
                        player.remove_object()
                        events_infos["object_in_trash"][
                                            player_idx
                                        ] = True


            elif terrain_type == "P":
                object_actions = {
                    "clean_plate": "boiled_rice-plate",
                    "tortilla-plate": "boiled_rice-tortilla-plate",
                    "fried_mushroom-tortilla-plate": "mushroom_burrito",
                    "chopped_steak-tortilla-plate": "steak_burrito",
                    "chopped_steak-plate": "chopped_steak-boiled_rice-plate",
                    "fried_mushroom-plate": "fried_mushroom-boiled_rice-plate",
                }
                # ready to pickup item from pot
                ready, burnt, is_extinguished, ready_name = self.pot_ready_at_location(
                    new_state, i_pos
                )
                if player.has_object():
                    player_obj = player.get_object().name
                    if player_obj == "fire_ext" and burnt and not is_extinguished:
                        obj = new_state.remove_object(i_pos)
                        new_o_id = obj_count
                        new_obj = BurntObjectState(new_o_id, "charcoal", i_pos)
                        obj_count += 1
                        new_state.add_object(new_obj)
                        if "extinguish_fire" in self.reward_shaping_params:
                            shaped_reward[player_idx] += self.reward_shaping_params["extinguish_fire"][player_idx]
                        events_infos["extinguish_fire"][
                                            player_idx
                                        ] = True

#                    elif player_obj in object_actions and burnt and is_extinguished:
#                        if player_obj == "clean_plate":
#                            new_state.remove_object(i_pos)  # remove charcoal
#                            player.remove_object()
#                            new_o_id = obj_count
#                            obj_count += 1
#                            new_obj = IdObjectState(new_o_id, "charcoal-plate", pos)
#                            player.set_object(new_obj)
#
                    elif player_obj in object_actions and ready and not burnt:
                        # new_state.remove_object(i_pos)  # Remove the chopped_object
                        new_obj_name = None
                        new_state.remove_object(i_pos)  # Remove the chopped_object
                        new_obj_name = object_actions[player_obj]
                        if new_obj_name:
                            new_o_id = obj_count
                            new_obj = IdObjectState(new_o_id, new_obj_name, pos)
                            player.remove_object()  # Remove the clean plate
#                            self.log_object_pickup(
#                                events_infos,
#                                new_state,
#                                new_obj_name,
#                                pot_states,
#                                player_idx,
#                            )
                            events_infos[f"{new_obj_name}_pot_pickup"][
                                        player_idx
                                    ] = True
                            player.set_object(new_obj)
                            if "pickup_cooked_ingredient_from_pot" in self.reward_shaping_params:
                                shaped_reward[player_idx] += self.reward_shaping_params["pickup_cooked_ingredient_from_pot"][player_idx]

                            if player_obj != "clean_plate":
                                if "food_assembly" in self.reward_shaping_params:
                                    shaped_reward[player_idx] += self.reward_shaping_params["food_assembly"][player_idx]

                    elif (
                        player.get_object().name in Burrito_Recipe.ALL_INGREDIENTS
                        and (not burnt or is_extinguished)
                    ):
                        item_type = player.get_object().name
                        if item_type != "rice":
                            break
                        if not new_state.has_object(i_pos) or new_state.get_object(i_pos).name == "charcoal":
                            obj = player.remove_object()
                            # Pot was empty, add boiled_rice to it
                            new_o_id = obj_count
                            new_obj = None
                            new_obj = CookableObjectState(
                                new_o_id, "boiled_rice", i_pos, []
                            )
                            if not rollout:
                                self.object_id_dict[new_o_id] = new_obj
                            obj_count += 1
                            
                            # enable swap ingred with charcoal
                            if new_state.has_object(i_pos) and new_state.get_object(i_pos).name == "charcoal":
                                charcoal = new_state.remove_object(i_pos)
                                player.set_object(charcoal)

                            new_obj.add_ingredient(obj)
                            new_obj.begin_cooking()
                            new_state.add_object(new_obj, i_pos)
                            events_infos["potting_rice"][player_idx] = True
                            if "place_ingredient_on_pot" in self.reward_shaping_params:
                                shaped_reward[player_idx] += self.reward_shaping_params["place_ingredient_on_pot"][player_idx]
#                        soup = new_state.get_object(i_pos)
#                        if not soup.is_full:
#                            obj = player.remove_object()
#                            soup.add_ingredient(obj)
#                            soup.begin_cooking()
#                            #shaped_reward[player_idx] += self.reward_shaping_params[
#                            #    "PLACEMENT_IN_POT_REW"
#                            #]
#                            if obj.name == Burrito_Recipe.CHICKEN:
#                                events_infos["potting_chicken"][player_idx] = True
#                            if obj.name == Burrito_Recipe.RICE:
#                                events_infos["potting_rice"][player_idx] = True
 
                elif player.held_object is None and is_extinguished:
                    obj = new_state.get_object(i_pos).name
                    if obj == "charcoal":
                        new_obj = new_state.remove_object(i_pos)  # remove charcoal
                        # new_o_id = obj_count
                        # obj_count += 1
                        # new_obj = IdObjectState(new_o_id, "charcoal", pos)
                        player.set_object(new_obj)

            elif terrain_type == "G":
                object_actions = {
                    "clean_plate": {
                        "fried_mushroom": "fried_mushroom-plate",
                        "chopped_steak": "chopped_steak-plate",
                    },
                    "boiled_rice-tortilla-plate": {
                        "chopped_steak": "steak_burrito",
                        "fried_mushroom": "mushroom_burrito",
                    },
                    "tortilla-plate": {
                        "chopped_steak": "chopped_steak-tortilla-plate",
                        "fried_mushroom": "fried_mushroom-tortilla-plate",
                    },
                    "boiled_rice-plate": {
                        "chopped_steak": "chopped_steak-boiled_rice-plate",
                        "fried_mushroom": "fried_mushroom-boiled_rice-plate",
                    },
                }

                ready, burnt, is_extinguished, ready_name = (
                    self.steak_ready_at_location(new_state, i_pos)
                )
                if player.has_object():
                    player_obj = player.get_object().name
                    if player_obj == "fire_ext" and burnt and not is_extinguished:
                        obj = new_state.remove_object(i_pos)
                        new_o_id = obj_count
                        new_obj = BurntObjectState(new_o_id, "charcoal", i_pos, True)
                        obj_count += 1
                        new_state.add_object(new_obj)
                        if "extinguish_fire" in self.reward_shaping_params:
                            shaped_reward[player_idx] += self.reward_shaping_params["extinguish_fire"][player_idx]
                        events_infos["extinguish_fire"][
                                            player_idx
                                        ] = True

               
#                    elif player_obj in object_actions and burnt and is_extinguished:
#                        if player_obj == "clean_plate":
#                            new_state.remove_object(i_pos)  # remove charcoal
#                            player.remove_object()
#                            new_o_id = obj_count
#                            obj_count += 1
#                            new_obj = IdObjectState(new_o_id, "charcoal-plate", pos)
#                            player.set_object(new_obj)
#
                    elif player_obj in object_actions and ready and not burnt:
                        new_obj_name = None
                        new_state.remove_object(i_pos)  # Remove the chopped_object
                        new_obj_name = object_actions[player_obj][ready_name]
                        #print("new_obj_name", new_obj_name)
                        if new_obj_name:
                            new_o_id = obj_count
                            new_obj = IdObjectState(new_o_id, new_obj_name, pos)
                            player.remove_object()  # Remove the clean plate
#                            self.log_object_pickup(
#                                events_infos,
#                                new_state,
#                                new_obj_name,
#                                pot_states,
#                                player_idx,
#                            )
                            events_infos[f"{new_obj_name}_grill_pickup"][
                                        player_idx
                                    ] = True
                            player.set_object(new_obj)
                            if "pickup_cooked_ingredient_from_grill" in self.reward_shaping_params:
                                shaped_reward[player_idx] += self.reward_shaping_params["pickup_cooked_ingredient_from_grill"][player_idx]
                            if player_obj != 'clean_plate':
                                if "food_assembly" in self.reward_shaping_params:
                                    shaped_reward[player_idx] += self.reward_shaping_params["food_assembly"][player_idx]

                    elif (
                        player.get_object().name in Burrito_Recipe.ALL_INGREDIENTS
                        and (not burnt or is_extinguished)
                    ):
                        item_type = player.get_object().name
                        if item_type in ["chopped_meat", "chopped_mushroom"]:
                            if not new_state.has_object(i_pos) or new_state.get_object(i_pos).name == "charcoal":
                                obj = player.remove_object()
                                new_o_id = obj_count
                                new_obj_name = (
                                    "chopped_steak"
                                    if item_type == "chopped_meat"
                                    else "fried_mushroom"
                                )
                                new_obj = CookableObjectState(
                                    new_o_id, new_obj_name, i_pos, []
                                )
                                if not rollout:
                                    self.object_id_dict[new_o_id] = new_obj
                                obj_count += 1

                                # enable swap ingred with charcoal
                                if new_state.has_object(i_pos) and new_state.get_object(i_pos).name == "charcoal":
                                    charcoal = new_state.remove_object(i_pos)
                                    player.set_object(charcoal)

                                new_obj.add_ingredient(obj)
                                new_obj.begin_cooking()
                                new_state.add_object(new_obj, i_pos)

                                #shaped_reward[player_idx] += self.reward_shaping_params[
                                #    "PLACEMENT_IN_POT_REW"
                                #]
                                events_infos[f"{new_obj_name}_cooking"][
                                    player_idx
                                ] = True
                                if "place_ingredient_on_grill" in self.reward_shaping_params:
                                    shaped_reward[player_idx] += self.reward_shaping_params["place_ingredient_on_grill"][player_idx]
 
                elif player.held_object is None and is_extinguished:
                    obj = new_state.get_object(i_pos).name
                    if obj == "charcoal":
                        new_obj = new_state.remove_object(i_pos)  # remove charcoal
                        # new_o_id = obj_count
                        # obj_count += 1
                        # new_obj = IdObjectState(new_o_id, "charcoal", pos)
                        player.set_object(new_obj)

            elif terrain_type == "S" and player.has_object():
                obj = player.get_object()
                dish_name = obj.name + "_dish"
                # if (dish_name in new_state.order_list) or (dish_name in new_state._complete_orders):
                if dish_name in DISH_TYPES:
                    new_state, delivery_rew, delivery_type = self.deliver_dish(new_state, player, obj)
                    for idx, _ in enumerate(new_state.players):
                        sparse_reward[idx] += (delivery_rew/2)
                    # sparse_reward[player_idx] += delivery_rew
                    discount = 1
                    if delivery_type == "in order":
                        if "correct_dish_bonus" in self.reward_shaping_params:
                            discount = self.reward_shaping_params["correct_dish_bonus"][player_idx]
                    else:
                        if "wrong_dish_discount" in self.reward_shaping_params:
                            discount = self.reward_shaping_params["wrong_dish_discount"][player_idx]
                    if "serve_dish" in self.reward_shaping_params:
                        shaped_reward[player_idx] += self.reward_shaping_params["serve_dish"][player_idx] * discount

                    if new_state.num_plates < new_state.max_plates:
                        new_state.num_plates += 1

                    # Log dish delivery
                    events_infos["dish_delivery"][player_idx] = True
                    if delivery_type == "in order":
                        events_infos["in_order_dish_delivery"][player_idx] = True
                    if delivery_type != "random":
                        events_infos["correct_dish_delivery"][player_idx] = True

            elif terrain_type == "B":
                if player.held_object is None:
                    if new_state.has_object(i_pos):
                        obj = new_state.get_object(i_pos)
                        assert (
                            obj.name == "garnish"
                            or obj.name == "chopped_meat"
                            or obj.name == "chopped_mushroom"
                        ), f"Object on chopping board was not garnish or chopped beef: was {obj.name}"
                        ready, ready_name = self.chopped_item_ready_at_location(
                            new_state, i_pos
                        )
                        if not ready:
                            obj.chop()
                            # shaped_reward[
                            #     player_idx] += self.reward_shaping_params[
                            #         "CHOPPING_ONION_REW"]

                            # Log onion chopping
                            events_infos["onion_chopping"][player_idx] = True
                            if "chop_ingredient" in self.reward_shaping_params:
                                shaped_reward[player_idx] += self.reward_shaping_params["chop_ingredient"][player_idx]
                        else:
                            new_state.remove_object(i_pos)  # Remove the chopped_object
                            picked_up_obj = IdObjectState(
                                i_pos, ready_name, pos
                            )  # Get steak or mushroom
                            self.log_object_pickup(
                                events_infos,
                                new_state,
                                ready_name,
                                pot_states,
                                player_idx,
                            )
                            # Pick up steak
                            #print("pick up ", picked_up_obj)
                            player.set_object(picked_up_obj)
                            if "pickup_chopped_ingredient_from_chopboard" in self.reward_shaping_params:
                                shaped_reward[player_idx] += self.reward_shaping_params["pickup_chopped_ingredient_from_chopboard"][player_idx]

                elif player.get_object().name == "onion" and not new_state.has_object(
                    i_pos
                ):
                    # Chopping board was empty, add onion to it
                    obj = player.remove_object()
                    new_o_id = obj_count
                    new_obj = GarnishState(new_o_id, "garnish", i_pos, [])
                    if not rollout:
                        self.object_id_dict[new_o_id] = new_obj
                    obj_count += 1
                    new_obj.add_ingredient(obj)
                    new_obj.begin_chop()
                    new_state.add_object(new_obj, i_pos)
                    # shaped_reward[
                    # player_idx] += self.reward_shaping_params[
                    # "PLACEMENT_ON_BOARD_REW"]

                    # Log onion potting
                    events_infos["onion_chopping"][player_idx] = True

                # Pick up garnish
                elif player.get_object().name == "meat" and not new_state.has_object(
                    i_pos
                ):
                    # Chopping board was empty, add chopped beef
                    obj = player.remove_object()
                    new_o_id = obj_count
                    new_obj = ChoppedItemState(new_o_id, "chopped_meat", i_pos, [])
                    if not rollout:
                        self.object_id_dict[new_o_id] = new_obj
                    obj_count += 1
                    new_obj.begin_chop()
                    new_state.add_object(new_obj, i_pos)
                    # Log beeef potting
                    events_infos["meat_chopping"][player_idx] = True
                elif (
                    player.get_object().name == "mushroom"
                    and not new_state.has_object(i_pos)
                ):
                    # Chopping board was empty, add chopped mushroom
                    obj = player.remove_object()
                    new_o_id = obj_count
                    new_obj = ChoppedItemState(new_o_id, "chopped_mushroom", i_pos, [])
                    if not rollout:
                        self.object_id_dict[new_o_id] = new_obj
                    obj_count += 1
                    new_obj.begin_chop()
                    new_state.add_object(new_obj, i_pos)
                    # Log beeef potting
                    events_infos["mushroom_chopping"][player_idx] = True
                # Pick up garnish
                elif (
                    player.get_object().name == "steak"
                    and self.garnish_ready_at_location(new_state, i_pos)
                ):
                    player.remove_object()  # Remove the clean plate
                    self.log_object_pickup(
                        events_infos, new_state, "steak", pot_states, player_idx
                    )

                    _ = new_state.remove_object(i_pos)  # Get steak
                    new_o_id = obj_count
                    new_obj = IdObjectState(new_o_id, "steak_onion", pos)
                    if not rollout:
                        self.object_id_dict[new_o_id] = new_obj
                    obj_count += 1
                    player.set_object(new_obj)
                    # shaped_reward[player_idx] += self.reward_shaping_params[
                    #     "GARNISH_STEAK_REWARD"]
                # Pick up garnish
                elif (
                    player.get_object().name == "boiled_chicken"
                    and self.garnish_ready_at_location(new_state, i_pos)
                ):
                    player.remove_object()  # Remove the clean plate
                    self.log_object_pickup(
                        events_infos,
                        new_state,
                        "boiled_chicken",
                        pot_states,
                        player_idx,
                    )

                    _ = new_state.remove_object(i_pos)  # Get steak
                    new_o_id = obj_count
                    new_obj = IdObjectState(new_o_id, "boiled_chicken_onion", pos)
                    if not rollout:
                        self.object_id_dict[new_o_id] = new_obj
                    obj_count += 1
                    player.set_object(new_obj)
                    # shaped_reward[player_idx] += self.reward_shaping_params[
                    #     "GARNISH_STEAK_REWARD"]
            else:
                continue

            new_state.obj_count = obj_count

        return sparse_reward, shaped_reward

    """Override overcooked_mdp implementation so collisions don't freeze all agents"""
    def resolve_movement(self, state, joint_action):
        """Resolve player movement and deal with possible collisions"""
        (
            new_positions,
            new_orientations,
        ) = self.compute_new_positions_and_orientations(
                state.players, joint_action
        )
        for player_state, new_pos, new_o in zip(
            state.players, new_positions, new_orientations
        ):
            player_state.update_pos_and_or(new_pos, new_o)

    def compute_new_positions_and_orientations(
            self, old_player_states, joint_action
    ):
        """Compute new positions and orientations ignoring collisions"""
        new_positions, new_orientations = list(
            zip(
                *[
                    self._move_if_direction(p.position, p.orientation, a)
                    for p, a in zip(old_player_states, joint_action)
                ]
            )
        )
        old_positions = tuple(p.position for p in old_player_states)
        if self.num_players == 1:
            return new_positions, new_orientations
        new_positions = self._handle_collisions(old_positions, new_positions)
        # print("old positions: ", old_positions)
        # print("new positions: ", new_positions)
        return new_positions, new_orientations

    
    def _handle_collisions(self, current_positions, desired_positions):
        """
        Randomized brute-force: try subsets of movers (largest first),
        shuffle within each size, and return the first collision-free result,
        but skip pure 2-agent swaps.  Longer cycles or other moves are allowed.
        """
        n = len(current_positions)
        final = tuple(current_positions)
        
        # agents that actually want to move
        movers = [i for i in range(n) 
                if desired_positions[i] != current_positions[i]]
        
        # helper to detect a pure swap
        def has_swap_pair(subset):
            # return True if any two agents in subset would swap
            for a, b in itertools.combinations(subset, 2):
                if (desired_positions[a] == current_positions[b] and
                    desired_positions[b] == current_positions[a]):
                    return True
            return False
        
        # try to move as many as possible: from all movers down to 1
        for k in range(len(movers), 0, -1):
            subsets = list(itertools.combinations(movers, k))
            random.shuffle(subsets)
            
            for subset in subsets:
                # skip pure swaps
                if has_swap_pair(subset):
                    continue
                
                # build candidate positions
                cand = list(current_positions)
                for i in subset:
                    cand[i] = desired_positions[i]
                
                # accept if collision‑free
                if len(set(cand)) == n:
                    return tuple(cand)
        
        # fallback: no one moves
        return final


    def deliver_dish(self, state, player, dish_obj):
        """
        Deliver the steak, and get reward if there is no order list
        or if the type of the delivered steak matches the next order.
        """
        player.remove_object()

        # TODO: Clean up this function, a lot of this stuff can be deleted

        #if state.order_list is None:
        #    return state, self._delivery_reward

        # If the delivered soup is the one currently required
        # assert not self.is_terminal(state)
        current_order = state.order_list[0]
        all_active_orders = [order[0] for order in state.order_list]
        dish = dish_obj.name + "_dish"

        # dish is the first order in the order list
        if dish in current_order:
            state.consecutive_orders += 1  # dish served in order
            completed_order = state.order_list.pop(0)
            state.order_list.append(self._generate_next_order(state))
            state._order_display_list = state.order_list  # + state._complete_orders
            reward, tip, number_ingredients, multipier = generate_reward(
                completed_order, state.consecutive_orders, self.reward_shaping_params["multiplier_params"]
            )
            #print("COMPLETED IN ORDER:", completed_order)
            state._bonus_orders.append(
                [
                    dish + "_tick",
                    completed_order[2] - completed_order[1],
                    tip,
                    number_ingredients,
                    multipier,
                ]
            )
            state._complete_orders.append(dish + "_tick")
            return state, reward, "in order"
        # dish served not in order, but in order list
        elif dish in all_active_orders:
            state.consecutive_orders = 0
            for i, order in enumerate(state.order_list):
                if order[0] == dish:
                    completed_order = state.order_list.pop(i)
                    reward, tip, number_ingredients, multipier = generate_reward(
                        completed_order, state.consecutive_orders, self.reward_shaping_params["multiplier_params"]
                    )
                    state._bonus_orders.append(
                        [
                            dish + "_tick",
                            completed_order[2] - completed_order[1],
                            tip,
                            number_ingredients,
                            multipier,
                        ]
                    )
                    #print("COMPLETED OUT OF ORDER:", completed_order)
                    state.order_list.append(self._generate_next_order(state))
                    state._order_display_list = (
                        state.order_list
                    )  # + state._complete_orders
                    # print("bonus orders",state._bonus_orders)
                    return state, reward, "out of order" # reward
        # dish is not in order list
        else:
            #print("COMPLETED RANDOM DISH:", dish)
            state.consecutive_orders = 0
            # NOTE: Give a smaller fixed reward 
            #
            # dish served not in order list
            #state._bonus_orders.append([dish + "_tick", completed_order[1]])
            #state.order_list.append(self._generate_next_order())
            # print("bonus orders",state._bonus_orders)
            return state, self.reward_shaping_params["random_dish_delivery"], "random" # 40

    def _generate_next_order(self, state):
        state.next_recipe = (state.next_recipe + 1) % len(self.all_recipes)
        next = self.all_recipes[state.next_recipe]
        return next

    def step_environment_effects(self, state):
        state.timestep += 1
        # Update the state of the timers on each dish

        state.order_list = [
            [order[0], max(0, order[1] - 2), order[2]] for order in state.order_list
        ]
        initial_order_count = len(state.order_list)
        state.order_list = [order for order in state.order_list if order[1] > 0]

        if len(state.order_list) < initial_order_count:
            state.consecutive_orders = 0
            state.score =- 15 # penalty for not delivering on time

        if state.timestep > 550:
            while len(state.order_list) < 3:
                state.order_list.append(self._generate_next_order(state))

        if state.timestep > 700:
            while len(state.order_list) < 4:
                state.order_list.append(self._generate_next_order(state))
        
        env_reward = 0.0 
        for obj in state.objects.values():
            if (
                obj.name in ["chopped_steak", "fried_mushroom", "boiled_rice"]
                and not obj.is_picked_up
            ):
                # automatically starts cooking when the pot has 1 ingredients
                if (
                    not obj.is_cooking
                    and not obj.is_ready
                    and len(obj.ingredients) == 1
                ):
                    obj.begin_cooking()
                #  print("begin cooking", obj)
                elif obj.is_cooking:
                    obj.cook()
                elif obj.is_waiting_for_pickup and not obj.is_burnt:
                    obj.wait()
                elif not obj.is_burnt:
                    obj.warn()
                elif obj.is_burnt:
                    env_reward += self.reward_shaping_params["burn_ingredient"]
        return env_reward

    def is_terminal(self, state):
        # There is a finite horizon, handled by the environment.
        # TODO: better terminal state? is it even necessary
        # if len(state.order_list) <= 0:
        #     return True
        return False

    #######################
    # LAYOUT / STATE INFO #
    #######################

    def get_chopping_board_locations(self):
        return list(self.terrain_pos_dict["B"])

    def get_meat_dispenser_locations(self):
        return list(self.terrain_pos_dict["M"])

    def get_chicken_dispenser_locations(self):
        return list(self.terrain_pos_dict["C"])

    def get_sink_locations(self):
        return list(self.terrain_pos_dict["W"])

    def get_dirty_plate_locations(self):
        return list(self.terrain_pos_dict["D"])

    def get_grill_locations(self):
        return list(self.terrain_pos_dict["G"])

    def get_key_objects_locations(self):
        return (
            self.mdp.get_onion_dispenser_locations()
            + self.mdp.get_chopping_board_locations()
            + self.mdp.get_meat_dispenser_locations()
            + self.mdp.get_grill_locations()
            + self.mdp.get_pot_locations()
            + self.mdp.get_dirty_plate_dispenser_locations()
            + self.mdp.get_sink_locations()
        )

    def get_pot_states(self, state, pots_states_dict=None, valid_pos=None):
        """Returns dict with structure:
        {
         empty: [ObjStates]
         onion: {
            'x_items': [soup objects with x items],
            'cooking': [ready soup objs]
            'ready': [ready soup objs],
            'partially_full': [all non-empty and non-full soups]
            }
         tomato: same dict structure as above
        }
        """
        if pots_states_dict is None:
            pots_states_dict = defaultdict(list)

        get_pot_info = []
        if valid_pos is not None:
            for pot_pos in self.get_pot_locations():
                if pot_pos in valid_pos:
                    get_pot_info.append(pot_pos)
        else:
            get_pot_info = self.get_pot_locations()

        for pot_pos in get_pot_info:
            if not state.has_object(pot_pos):
                pots_states_dict["empty"].append(pot_pos)
            else:
                soup = state.get_object(pot_pos)
                assert soup.name == "soup" or "chicken", (
                    "soup at "
                    + str(pot_pos)
                    + " is not a chicken/soup but a "
                    + soup.name
                )
                if soup.is_ready:
                    pots_states_dict["ready"].append(pot_pos)
                elif soup.is_cooking:
                    pots_states_dict["cooking"].append(pot_pos)
                else:
                    num_ingredients = len(soup.ingredients)
                    pots_states_dict["{}_items".format(num_ingredients)].append(pot_pos)

        return pots_states_dict

    def get_grill_states(self, state, grills_states_dict=None, valid_pos=None):
        """Returns dict with structure:
        {
         empty: [positions of empty pots]
        'x_items': [grill objects with x items that have yet to start grilling],
        'cooking': [grill objs that are grilling but not ready]
        'ready': [ready grill objs],
        }
        NOTE: all returned grills are just grill positions
        """
        if grills_states_dict is None:
            grills_states_dict = defaultdict(list)

        get_grill_info = []
        if valid_pos is not None:
            for grill_pos in self.get_grill_locations():
                if grill_pos in valid_pos:
                    get_grill_info.append(grill_pos)
        else:
            get_grill_info = self.get_grill_locations()

        for grill_pos in get_grill_info:
            if not state.has_object(grill_pos):
                grills_states_dict["empty"].append(grill_pos)
            else:
                steak = state.get_object(grill_pos)
                assert steak.name == "steak", (
                    "steak at " + grill_pos + " is not a steak but a " + steak.name
                )
                if steak.is_ready:
                    grills_states_dict["ready"].append(grill_pos)
                else:  # steak is_cooking
                    grills_states_dict["cooking"].append(grill_pos)

        return grills_states_dict

    def get_ready_grills(self, grill_states):
        return grill_states["ready"]

    def get_cooking_grills(self, grill_states):
        return grill_states["cooking"]

    def get_sink_states(self, state):
        empty_sink = []
        full_sink = []
        ready_sink = []
        sink_locations = self.get_sink_locations()
        for loc in sink_locations:
            if not state.has_object(loc):  # board is empty
                empty_sink.append(loc)
            else:
                obj = state.get_object(loc)
                if obj.is_ready:
                    ready_sink.append(loc)
                else:
                    full_sink.append(loc)
        return {"empty": empty_sink, "full": full_sink, "ready": ready_sink}

    def get_chopping_board_states(self, state):
        empty_board = []
        full_board = []
        ready_board = []
        board_locations = self.get_chopping_board_locations()
        for loc in board_locations:
            if not state.has_object(loc):  # board is empty
                empty_board.append(loc)
            else:
                obj = state.get_object(loc)
                if obj.is_ready:
                    ready_board.append(loc)
                else:
                    full_board.append(loc)
        return {"empty": empty_board, "full": full_board, "ready": ready_board}

    def steak_ready_at_location(self, state, pos):
        obj_name = ["steak", "chopped_steak", "fried_mushroom", "charcoal"]
        if not state.has_object(pos):
            return False, False, False, None
        obj = state.get_object(pos)
        assert obj.name in obj_name, "Object at location was not {}".format(obj_name)
        return obj.is_ready, obj.is_burnt, obj.is_extinguished, obj.name

    def steak_to_be_cooked_at_location(self, state, pos):
        if not state.has_object(pos):
            return False
        obj = state.get_object(pos)
        return obj.name == "steak" and not obj.is_cooking and not obj.is_ready

    def plate_clean_at_location(self, state, pos):
        if not state.has_object(pos):
            return False
        obj = state.get_object(pos)
        if obj.name == "dirty_plate":
            return False
        return obj.name == "clean_plate"

    def garnish_ready_at_location(self, state, pos):
        if not state.has_object(pos):
            return False
        obj = state.get_object(pos)
        assert obj.name == "garnish", "Object on chopping board was not garnish"
        prep_time = obj._cooking_tick
        return prep_time >= obj._cook_time

    def chopped_item_ready_at_location(self, state, pos):
        if not state.has_object(pos):
            return False
        obj = state.get_object(pos)
        assert (
            obj.name == "chopped_meat" or obj.name == "chopped_mushroom"
        ), "Object on chopping board was not chopped_meat or chopped_mushroom"
        prep_time = obj._cooking_tick
        return prep_time >= obj._cook_time, obj.name

    # TODO: change above objectname_ready_at_location to object_ready_at_location
    def pot_ready_at_location(self, state, pos):
        obj_name = ["boiled_chicken", "boiled_rice", "charcoal", "charcoal_pot"]
        if not state.has_object(pos):
            return False, False, False, None
        obj = state.get_object(pos)
        assert obj.name in obj_name, "Object at location was not {}".format(obj_name)
        return obj.is_ready, obj.is_burnt, obj.is_extinguished, obj.name

    ################################
    # EVENT LOGGING HELPER METHODS #
    ################################

    def log_object_drop(self, events_infos, state, obj_name, pot_states, player_index):
        """Player dropped the object on a counter"""
        obj_drop_key = obj_name + "_drop"
        
        # Only set the event if it exists in events_infos (from EVENT_TYPES)
        if obj_drop_key in events_infos:
            events_infos[obj_drop_key][player_index] = True

    def is_potting_optimal(self, state, old_soup, new_soup):
        """
        True if the highest valued soup possible is the same before and after the potting
        """
        old_recipe = (
            Burrito_Recipe(old_soup.ingredients) if old_soup.ingredients else None
        )
        new_recipe = Burrito_Recipe(new_soup.ingredients)
        old_val = self.get_recipe_value(
            state, self.get_optimal_possible_recipe(state, old_recipe)
        )
        new_val = self.get_recipe_value(
            state, self.get_optimal_possible_recipe(state, new_recipe)
        )
        return old_val == new_val

    def is_potting_viable(self, state, old_soup, new_soup):
        """
        True if there exists a non-zero reward soup possible from new ingredients
        """
        new_recipe = Burrito_Recipe(new_soup.ingredients)
        new_val = self.get_recipe_value(
            state, self.get_optimal_possible_recipe(state, new_recipe)
        )
        return new_val > 0

    def is_potting_catastrophic(self, state, old_soup, new_soup):
        """
        True if no non-zero reward soup is possible from new ingredients
        """
        old_recipe = (
            Burrito_Recipe(old_soup.ingredients) if old_soup.ingredients else None
        )
        new_recipe = Burrito_Recipe(new_soup.ingredients)
        old_val = self.get_recipe_value(
            state, self.get_optimal_possible_recipe(state, old_recipe)
        )
        new_val = self.get_recipe_value(
            state, self.get_optimal_possible_recipe(state, new_recipe)
        )
        return old_val > 0 and new_val == 0

    def is_potting_useless(self, state, old_soup, new_soup):
        """
        True if ingredient added to a soup that was already gauranteed to be worth at most 0 points
        """
        old_recipe = (
            Burrito_Recipe(old_soup.ingredients) if old_soup.ingredients else None
        )
        old_val = self.get_recipe_value(
            state, self.get_optimal_possible_recipe(state, old_recipe)
        )
        return old_val == 0

    #####################
    # TERMINAL GRAPHICS #
    #####################

    def state_string(self, state):
        """String representation of the current state"""
        # TODO
        return ""

    ###################
    # STATE ENCODINGS #
    ###################

    def _convert_orientation(self, orientation):
        return (
            "SOUTH"
            if orientation[1] == 1
            else (
                "NORTH"
                if orientation[1] == -1
                else ("EAST" if orientation[0] == 1 else "WEST")
            )
        )


    def _convert_to_layer(self, current_env_grid:np.ndarray, burrito_state:BurritoState):
        ############### encode fixed environment assets ####################3
        def _handle_cell(row, col, cell):
            current_env_grid[row, col, 0] = counter_mapping.get(cell, 0)
            current_env_grid[row, col, 1] = ingredient_mapping.get(cell, 0)
            current_env_grid[row, col, 3] = kitchen_tool_mapping.get(cell, 0) 
        for row_idx, row in enumerate(self.terrain_mtx):
            for col_idx, cell in enumerate(row):
                _handle_cell(row_idx, col_idx, cell)
        
        ############ encode dynamic objects ######################
        def _handle_dynamic_objects(dynamic_objects):
            for obj in dynamic_objects.values():
                row, col = obj.position
                name = obj.name
                is_burnt = getattr(obj, "is_burnt", False)
                _is_extinguished = getattr(obj, "_is_extinguished", False)
                _cooking_tick = getattr(obj, "_cooking_tick", None)
                _warning_tick = getattr(obj, "_warning_tick", None)
                _waiting_tick = getattr(obj, "_waiting_tick", None)

                # print("Object Name", name)
                if _cooking_tick:
                    # print("Cooking Tick", _cooking_tick)
                    current_env_grid[col, row, 4] = _cooking_tick
                if _waiting_tick:
                    # print("Waiting Tick", _waiting_tick)
                    current_env_grid[col, row, 5] = _waiting_tick
                if _warning_tick:
                    # print("Warning Tick", _warning_tick)
                    current_env_grid[col, row, 6] = _warning_tick

                existing = current_env_grid[col, row, 1]

                current_env_grid[col, row, 2] = (
                    held_item_mapping.get(name, None) or existing
                )

                current_env_grid[col, row, 7] = (
                    1 if is_burnt else (2 if _is_extinguished else 0)
                )
        _handle_dynamic_objects(burrito_state.objects)

        ###################### encode player states ###################
        def _handle_players(players:list[PlayerState]):
            for i, player in enumerate(players):
                index = i + 1
                row, col = player.position
                held_object = getattr(player.held_object, "name", "None")
                orientation = self._convert_orientation(player.orientation)
                current_env_grid[col, row, 8] = index  # PLAYER POSITION
                obj_map = held_item_mapping.get(held_object)
                # print("Held Object", held_object)
                # print("Object Map", obj_map)
                current_env_grid[col, row, 9] = held_item_mapping.get(
                    held_object, 0
                )  # PLAYER HELD OBJECT - can be held item or ingredient
                current_env_grid[col, row, 10] = orientation_mapping.get(
                    orientation, 0
                )  # PLAYER ORIENTATION
        _handle_players(burrito_state.players)


    def lossless_state_encoding(self, burrito_state:BurritoState) -> list[np.ndarray]:
        """
        Convert a burrito state object into a numpy array encoding. The output is a global observation
        :param burrito_state: A burrito object that will be encoded
        :return: the encoded numpy array map, shape (height, wide, 11), for each agent
            - Layer 0: positions of counter
            - Layer 1: positions of ingredient
            - Layer 2: positions of agent-held items
            - Layer 3: positions of kitchen tools
            - Layer 4~6: timers
            - Layer 7: ingredient burnt or extinguished status
            - Layer 8: position of agents
            - Layer 9: positions of agent-held items
            - Layer 10: orientations of agents
        """
        # The observation grid
        current_env_grid = np.zeros(
            (self.height, self.width, 11), dtype=np.int32
        )
        self._convert_to_layer(current_env_grid, burrito_state)
        # assert np.all(current_env_grid >= 0)
        return [current_env_grid.copy() for _ in range(self.num_players)]

    def nopos_state_encoding(self, burrito_state:BurritoState):
        """
        Encode state into vectors that do not contain spatial information. Not the full state

        Dims:
            - 0 ~ 6: kitchen tool status. 0 for nothing, 1 for occupied
            - 7 ~ 13: kitchen tool status. 0 for nothing, 1 for is ready
            - 14 ~ 20: kitchen tool status. 0 for nothing, 1 for is waiting for pickup
            - 21 ~ 27: kitchen tool status. 0 for nothing, 1 for warning
            - 28 ~ 34: kitchen tool status. 0 for nothing, 1 for burning
            - 35 ~ 41: kitchen tool status. 0 for nothing, 1 for is extinguished
            - 42 ~ 66: dynamic object status. 0 for nothing, 1 for on counter
            - 67 ~ 91: dynamic object status. 0 for nothing, 1 for on 'ego' agent hand
            - 92 ~ 116: dynamic object status. 0 for nothing, 1 for on other agent hand
            # - 117: passing counter status. 0 for unavailable, 1 for available
            # - 118: other counter status. 0 for unavailable, 1 for available
        """
        # X, S, M, Z, T, R, D, P , B, G, W, U 12 terrain layers
        terrain_types = [key for key in list(kitchen_tool_mapping.keys())]
        terrain_types.remove('U')
        terrain_types.remove('D')
        terrain_features = []
        for terrain_type in terrain_types:
            terrain_list = self.terrain_pos_dict[terrain_type]
            for i, (row, col) in enumerate(terrain_list):
                terrain_features.append(f'{terrain_type}_{i}')
        object_features = [key for key, value in held_item_mapping.items()]
        status_features = ["is_ready", "is_waiting_for_pickup", "is_warning", "is_burnt", "is_extinguished"]
        DIMS = terrain_features + [f"{terrain_feature}_{status}" for terrain_feature in terrain_features for status in status_features] + object_features + [f'{object_feature}_passingcounter' for object_feature in object_features] + [f'{object_feature}_ego' for object_feature in object_features] + [f'{object_feature}_other' for object_feature in object_features] # + [f'{object_feature}_order' for object_feature in object_features]
        state_dict = [{k: 0 for k in DIMS} for _ in range(self.num_players)]
        
        for terrain_type in terrain_types:
            terrain_list = self.terrain_pos_dict[terrain_type]
            for i, pos in enumerate(terrain_list):
                if burrito_state.has_object(pos):
                    for j in range(self.num_players):
                        state_dict[j][f'{terrain_type}_{i}'] = 1
                    obj = burrito_state.get_object(pos)
                    for status in status_features:
                        if getattr(obj, status, False):
                            for j in range(self.num_players):
                                state_dict[j][f'{terrain_type}_{i}_{status}'] = 1
        
        for obj in burrito_state.objects.values():
            name = obj.name
            row, col = obj.position
            if self.terrain_mtx[col][row] != ' ':
                for j in range(self.num_players):
                    state_dict[j][name] += 1
                    if row== 5 and col in [1,2,3]:
                        state_dict[j][f'{name}_passingcounter'] += 1
            else:
                for j in range(self.num_players):
                    if (row, col) == burrito_state.players[j].position:
                        state_dict[j][f'{name}_ego'] = 1
                        for k in range(self.num_players):
                            if k != j:
                                state_dict[k][f'{name}_other'] = 1

        state_vec = []
        for i in range(self.num_players):
            state_vec.append(np.stack([val for val in state_dict[i].values()]).astype(np.int32))
        return state_vec
#    def nopos_state_encoding(self, burrito_state:BurritoState):
#        """
#        Encode state into vectors that do not contain spatial information. Not the full state
#
#        Dims:
#            - 0 ~ 6: kitchen tool status. 0 for nothing, 1 for occupied
#            - 7 ~ 13: kitchen tool status. 0 for nothing, 1 for is ready
#            - 14 ~ 20: kitchen tool status. 0 for nothing, 1 for is waiting for pickup
#            - 21 ~ 27: kitchen tool status. 0 for nothing, 1 for warning
#            - 28 ~ 34: kitchen tool status. 0 for nothing, 1 for burning
#            - 35 ~ 41: kitchen tool status. 0 for nothing, 1 for is extinguished
#            - 42 ~ 66: dynamic object status. 0 for nothing, 1 for on counter
#            - 67 ~ 91: dynamic object status. 0 for nothing, 1 for on 'ego' agent hand
#            - 92 ~ 116: dynamic object status. 0 for nothing, 1 for on other agent hand
#            # - 117: passing counter status. 0 for unavailable, 1 for available
#            # - 118: other counter status. 0 for unavailable, 1 for available
#        """
#        # X, S, M, Z, T, R, D, P , B, G, W, U 12 terrain layers
#        terrain_types = [key for key in list(kitchen_tool_mapping.keys())]
#        terrain_types.remove('U')
#        terrain_types.remove('D')
#        terrain_features = []
#        for terrain_type in terrain_types:
#            terrain_list = self.terrain_pos_dict[terrain_type]
#            for i, (row, col) in enumerate(terrain_list):
#                terrain_features.append(f'{terrain_type}_{i}')
#        object_features = [key for key, value in held_item_mapping.items()]
#        status_features = ["is_ready", "is_waiting_for_pickup", "is_warning", "is_burnt", "is_extinguished"]
#        DIMS = terrain_features + [f"{terrain_feature}_{status}" for terrain_feature in terrain_features for status in status_features] + object_features + [f'{object_feature}_ego' for object_feature in object_features] + [f'{object_feature}_other' for object_feature in object_features] # + [f'{object_feature}_order' for object_feature in object_features]
#        state_dict = [{k: 0 for k in DIMS} for _ in range(self.num_players)]
#        
#        for terrain_type in terrain_types:
#            terrain_list = self.terrain_pos_dict[terrain_type]
#            for i, pos in enumerate(terrain_list):
#                if burrito_state.has_object(pos):
#                    for j in range(self.num_players):
#                        state_dict[j][f'{terrain_type}_{i}'] = 1
#                    obj = burrito_state.get_object(pos)
#                    for status in status_features:
#                        if getattr(obj, status, False):
#                            for j in range(self.num_players):
#                                state_dict[j][f'{terrain_type}_{i}_{status}'] = 1
#        
#        for obj in burrito_state.objects.values():
#            name = obj.name
#            row, col = obj.position
#            if self.terrain_mtx[col][row] != ' ':
#                for j in range(self.num_players):
#                    state_dict[j][name] = 1
#            else:
#                for j in range(self.num_players):
#                    if (row, col) == burrito_state.players[j].position:
#                        state_dict[j][f'{name}_ego'] = 1
#                        for k in range(self.num_players):
#                            if k != j:
#                                state_dict[k][f'{name}_other'] = 1
#
#        state_vec = []
#        for i in range(self.num_players):
#            state_vec.append(np.stack([val for val in state_dict[i].values()]).astype(np.int32))
#        return state_vec
#
#
    def onehot_state_encoding(self, burrito_state:BurritoState, debug=False):
        # X, S, M, Z, T, R, D, P , B, G, W, U 12 terrain layers
        terrain_features = [key for key in list(counter_mapping.keys())+list(ingredient_mapping.keys())+list(kitchen_tool_mapping.keys())]
        try:
            terrain_features.remove(' ')
        except Exception as e:
            pass
        # 25 dynamic object layers
        object_features = [key for key, value in held_item_mapping.items()]
        # 3 timer layers, not one-hot encoding
        timer_features = ["is_ready", "is_warning", "is_waiting_for_pickup"]
        # 2 burning layers
        burning_features = ["is_burnt", "is_extinguished"]
        # 5 position (1 layer) and orientation (4 layers) layers
        player_features = ["player_loc"] + ["player_orientation_{}".format(Direction.DIRECTION_TO_INDEX[d]) for d in Direction.ALL_DIRECTIONS]
        # 47 layers
        LAYERS = player_features + terrain_features + object_features + timer_features + burning_features
        state_mask_dict = {k:np.zeros((self.height, self.width)) for k in LAYERS}
        # terrain layers
        for terrain_type in terrain_features:
            terrain_list = self.terrain_pos_dict[terrain_type]
            for row, col in terrain_list:
                state_mask_dict[terrain_type][col, row] = 1
        # object and tick, burning feature layers
        for obj in burrito_state.objects.values():
            row, col = obj.position
            name = obj.name
            state_mask_dict[name][col, row] = 1
            for status_type in timer_features+burning_features:
                obj_status = getattr(obj, status_type, 0)
                state_mask_dict[status_type][col, row] = int(obj_status)
        # player layers
        for i, player in enumerate(burrito_state.players):
            row, col = player.position
            or_idx = Direction.DIRECTION_TO_INDEX[player.orientation]
            state_mask_dict["player_loc"][col, row] = 1
            state_mask_dict["player_orientation_{}".format(or_idx)][col, row] = 1
        #convert into numpy array
        state_mask_stack = np.stack([state_mask_layer for state_mask_layer in state_mask_dict.values()], axis=0).astype(np.int32)
        state_mask_stack = state_mask_stack.transpose(1,2,0)
        if debug:
            print("terrain----")
            print(np.array(self.terrain_mtx))
            print("-----------")
            print(len(LAYERS))
            print(len(state_mask_dict))
            for k, v in state_mask_dict.items():
                print(k)
                print(np.transpose(v, (0, 1)))
        return [state_mask_stack.copy() for _ in range(self.num_players)]
