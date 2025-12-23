import math
import torch

# Factory that returns a distance function with fixed bounds
def make_distance_to_bounds(lower_bound=-math.inf, upper_bound=math.inf, none_distance = math.inf):
    def distance(value):
        if value is None:
            return none_distance
        if value < lower_bound:
            return lower_bound - value
        if value > upper_bound:
            return value - upper_bound
        return 0
    return distance

# Factory that returns a distance function with fixed bounds
def make_distance_to_bounds_parallel(lower_bound=-math.inf, upper_bound=math.inf, none_distance = float('inf')):
    def cal_distance(values):
        dist = torch.zeros_like(values)

        # NaN → inf
        dist = torch.where(torch.isnan(values), torch.full_like(values, none_distance), dist)

        # value < lower_bound → lower_bound - value
        dist = torch.where(values < lower_bound, lower_bound - values, dist)

        # value > upper_bound → value - upper_bound
        dist = torch.where(values > upper_bound, values - upper_bound, dist)

        return dist
    return cal_distance

def compare_hierarchical(list1, list2):
    for a, b in zip(list1, list2):
        if a < b:
            return -1
        elif a > b:
            return 1
    # If all elements are the same, compare lengths
    return 0