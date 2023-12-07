from algo import search
import util


# TODO: define the HighSpeedRailProblem class, figure out parameters, decide on a budget
# How to define a goal state

class HighSpeedRailProblem:
    def __init__(self, initial_state, budget):
        # initializing parameters
        self.initial_state = initial_state
        self.budget = budget
        pass

    def evaluate(self, sol):
        # eval logic that should return a score
        # execution of search should return the sol with the highest score
        # % of budget to some perceived value of the solution ratio

        # if sol.cost <= self.budget:

        pass


class Solution:
    def __init__(self, path, cost):
        self.path = path  # path to sol
        self.cost = cost  # cost of sol


def main():
    initial_state = ...  # blank map
    budget = ...  # TBD

    # instance of the problem
    hsr = HighSpeedRailProblem(initial_state, budget)
    # performing the search
    solution = search(hsr)

    hsr.evaluate(solution)
    print(solution)


if __name__ == '__main__':
    main()
