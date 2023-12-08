from algo import HSRSearch
import util


# TODO: define the HighSpeedRailProblem class, figure out parameters, decide on a budget
# TODO: How to define a goal state

def isGoalState(state, budget):
    budget -= util.getStateCost(state)
    budget >= 0


class HighSpeedRailProblem:

    def getStartState(self):
        pass

    def getSuccessors(self, state):
        pass

    def getCostOfActions(self, actions):
        pass


def main():
    initial_state = ...  # blank map
    heuristic = 0
    hsr = HighSpeedRailProblem()
    # performing the search
    solution = HSRSearch(hsr, heuristic)
    print(solution)


if __name__ == '__main__':
    main()
