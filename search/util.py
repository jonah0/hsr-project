import heapq
import pathlib
from typing import List
from geopy.distance import great_circle, geodesic
from shapely.geometry import Point
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd


cwd = pathlib.Path(__file__).parent.resolve()
test_data = pd.read_csv(cwd.joinpath('../data/algo_testing_data.csv'))

# --------------------------------------------------- Methods/Functionality ------------------------------------------ #


class HSRSearchState():
    """
    A class to represent a state in our HSR search problem.
    A state is represented as a DataFrame of actions, where each action is a single intercity
    rail corridor built.
    """

    def __init__(self, railSegments: pd.DataFrame, colsForHash: List[str] = ['Origin', 'Dest']) -> None:
        self.railSegments = railSegments
        self.colsForHash = colsForHash
        self.cost: int

    def computeCost(self):
        """
        Compute the cost of this state. Lower numbers indicate more desirable states.

        NOTE: This refers to the UCS/search problem notion of "cost", NOT the monetary construction cost.
        """
        if self.railSegments.empty:
            return 0

        weight_pop = -0
        weight_time = +100
        weight_emissions = -0.4 * (1/1e8)

        score = 0
        score += weight_pop * (self.railSegments['pop_origin'] + self.railSegments['pop_dest'])
        score += weight_time * (self.railSegments['hsr_travel_time_hr'] -
                                self.railSegments['plane_travel_time_hr'] + 3)  # add 3 hrs for security, etc
        score += weight_emissions * self.railSegments['co2_g']
        return score.sum()

    def getRailSegments(self):
        return self.railSegments

    def getCost(self):
        if self.cost is None:
            self.cost = self.computeCost()
        return self.cost

    def getSuccessor(self, action: pd.Series):
        # transpose action from series into a single-row dataframe so that we can use pd.concat()
        action_df = action.to_frame().T
        newRailSegments = pd.concat([self.railSegments, action_df], axis='index')
        return HSRSearchState(newRailSegments)

    def getHash(self) -> frozenset:
        """
        Returns a `frozenset` representation of the given state, where each element of the frozenset is a rail segment.

        This turns the DataFrame into a hashable type that can be added to a `set` or compared easily.
        """
        return frozenset(self.railSegments[self.colsForHash].itertuples(name='rail', index=False))


class BetterPriorityQueue:
    """
      Implements a BETTER priority queue data structure designed for my UCS implementation. Each inserted item
      has a priority associated with it and the client is usually interested
      in quick retrieval of the lowest-priority item in the queue. This
      data structure allows O(1) access to the lowest-priority item.
    """

    def __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, oldItem, newItem, newPriority):
        # If item not in priority queue, do the same thing as self.push
        # If item already in priority queue with equal or lower priority, do nothing
        # If item already in priority queue, replace it with the new item with the new priority
        for index, (p, c, i) in enumerate(self.heap):
            if i == oldItem:
                if p <= newPriority:
                    break
                del self.heap[index]
                self.heap.append((newPriority, c, newItem))
                heapq.heapify(self.heap)
                break
        else:
            self.push(newItem, newPriority)

    def findNodeWithState(self, state):
        for index, (p, c, i) in enumerate(self.heap):
            if i[0] == state:
                return i
        return None


class PriorityQueue:
    """
      Implements a priority queue data structure. Each inserted item
      has a priority associated with it and the client is usually interested
      in quick retrieval of the lowest-priority item in the queue. This
      data structure allows O(1) access to the lowest-priority item.
    """

    def __init__(self):
        self.heap = []
        self.count = 0
        self.hashes = set()

    def push(self, item, priority):
        entry = (priority, self.count, item)
        self.hashes.add(item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        self.hashes.remove(item)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def contains(self, item) -> bool:
        return item in self.hashes

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)
            self.hashes.add(item)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def great_circle_two_points(pt1: Point, pt2: Point):
    return great_circle((pt1.y, pt1.x), (pt2.y, pt2.x))


def getRow(city1, city2) -> pd.Series:
    return test_data.loc[(test_data['Origin'] == city1) & (test_data['Dest'] == city2)].iloc[0]


def getPassServed(city1, city2) -> float:
    row = getRow(city1, city2)
    pop_orign = row['pop_origin']
    pop_dest = row['pop_dest']
    return pop_orign + pop_dest


def getEmissionsSaved(city1, city2) -> float:
    row = getRow(city1, city2)
    return row['co2_g']


def getRailCost(city1, city2) -> float:
    row = getRow(city1, city2)
    return row['construction_cost_usd']


def getTimeSaved(city1, city2) -> float:
    row = getRow(city1, city2)
    time_hsr = row['hsr_travel_time_hr']
    time_plane = row['plane_travel_time_hr']
    return time_plane - time_hsr


# sample_city1 = 'LAX'
# sample_city2 = 'BOS'
#
# pass_served = getPassServed(sample_city1, sample_city2)
# emissions_saved = getEmissionsSaved(sample_city1, sample_city2)
# rail_cost = getRailCost(sample_city1, sample_city2)
# time_saved = getTimeSaved(sample_city1, sample_city2)
#
# print(f"Passengers Served: {pass_served}")
# print(f"Emissions Saved: {emissions_saved}")
# print(f"Rail Cost: {rail_cost}")
# print(f"Time Saved: {time_saved}")
