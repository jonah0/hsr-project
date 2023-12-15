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

    def getRailSegments(self):
        return self.railSegments

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
        self.hashes: dict = {}

    def push(self, item, priority):
        entry = (priority, self.count, item)
        self.hashes[item] = priority
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        del self.hashes[item]
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def contains(self, item) -> bool:
        return item in self.hashes

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        if self.contains(item) and self.hashes[item] > priority:
            for index, (p, c, i) in enumerate(self.heap):
                if i == item:
                    if p <= priority:
                        break
                    del self.heap[index]
                    self.heap.append((priority, c, item))
                    heapq.heapify(self.heap)
                    break
            self.hashes[item] = priority

        if not self.contains(item):
            self.push(item, priority)
            self.hashes[item] = priority


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0
