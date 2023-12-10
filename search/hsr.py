from algo import HSRSearch
import util


# TODO: define the HighSpeedRailProblem class, figure out parameters, decide on a budget
# TODO: How to define a goal state

def isGoalState(state, budget):
    budget -= util.getConstructionCost(state)
    return budget >= 0


class HighSpeedRailProblem:
    """
    A search problem intended to explore what an optimal national HSR network
    could look like for the US, given various performance metrics and budgetary constraints.
    - The state space is all possible combinations of intercity rail segments.
    A rail segment is represented as a pair of two cities.
    - The action space is simply placing any rail segment that is not present.
    """

    def getStartState(self):
        pass

    def isGoalState(self, state):
        pass

    def getSuccessors(self, state):
        pass

    def getCostOfActions(self, actions):
        pass


class HSRProblem1(HighSpeedRailProblem):
    """
    Problem 1 is exploring the following question:
    Given a fixed construction budget, what is the optimal selection of intercity rail corridors to build?
    - The cost of building a certain rail segment is the negative of the "benefit"
    gained from the segment, as defined by the metrics we are interested in.
    - The goal state is any state in which the total monetary costs of constructing
    all present rail segments is equal to the predetermined budget.
    """
    pass


class HSRProblem2(HighSpeedRailProblem):
    """
    Problem 2 is exploring the following question:
    Given a goal of improving national metrics (emissions, population served by rail, etc.)
    by a specified amount, what is the least-cost investment necessary to achieve this,
    and which arrangement of intercity rail corridors would satisfy this?
    - The cost of building a certain rail segment is the construction cost.
    - The goal state is any state in which the total benefit gained by all rail segments
    is at least the predetermined threshold.
    """
    pass


def main():
    # a state in the search problem is a set of rail segments.
    # each segment should be a tuple of cities
    # todo: decide: each city is either represented by its name or its IATA airport code (BOS, LAX...)

    hsr = HighSpeedRailProblem()
    # performing the search
    solution = HSRSearch(hsr)
    print(solution)


if __name__ == '__main__':
    main()
