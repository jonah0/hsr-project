from algo import HSRSearch
import util


# TODO: define the HighSpeedRailProblem class, figure out parameters, decide on a budget
# TODO: How to define a goal state

def isGoalState(state, budget):
    budget -= util.getConstructionCost(state)
    return budget >= 0


class HighSpeedRailProblem:

    def getStartState(self):
        pass

    def isGoalState(self, state):
        pass

    def getSuccessors(self, state):
        pass

    def getCostOfActions(self, actions):
        pass


def main():
    # a state in the search problem is a set of rail segments.
    # each segment should be a tuple of cities
    # todo: decide: each city is either represented by its name or its IATA airport code (BOS, LAX...)
    initial_state = set()  # blank map

    hsr = HighSpeedRailProblem()
    # performing the search
    solution = HSRSearch(hsr)
    print(solution)


if __name__ == '__main__':
    main()
