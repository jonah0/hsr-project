from typing import Dict, Set, List, Any
import pathlib
import numpy as np
import pandas as pd
import util
import cProfile

cwd = pathlib.Path(__file__).parent.resolve()


class HighSpeedRailProblem:
    """
    A search problem intended to explore what an optimal national HSR network
    could look like for the US, given various performance metrics and budgetary constraints.
    - The state space is all possible combinations of intercity rail segments.
    A rail segment is represented as a pair of two cities.
    - The action space is simply placing any rail segment that is not present.
    """

    def __init__(self, od_matrix: pd.DataFrame) -> None:
        # od_matrix is origin-destination matrix; DataFrame of all city pairs and their associated metrics
        self.od_matrix = od_matrix
        x = self.od_matrix['Origin']
        y = self.od_matrix['Dest']
        xy = pd.merge(x, y, how='cross')
        self.xy = list(zip(xy['Origin'], xy['Dest']))

    def getStartState(self):
        return set()

    def isGoalState(self, state):
        pass

    def getSuccessors(self, state: list):
        successors = []

        for (origin, dest) in self.xy:
            if (origin, dest) not in state:
                nextState = state.copy().append((origin, dest))
                cost = evaluate_hsr(origin, dest)
                successors.append((nextState, (origin, dest), cost))

        return successors

    def getCostOfActions(self, actions):
        pass


class HighSpeedRailProblemPandas:
    """
    A search problem intended to explore what an optimal national HSR network
    could look like for the US, given various performance metrics and budgetary constraints.
    - The state space is all possible combinations of intercity rail segments.
    A rail segment is represented as a pair of two cities.
    - The action space is simply placing any rail segment that is not present.
    """

    def __init__(self, od_matrix: pd.DataFrame, colsForHash: List[str] = ['Origin', 'Dest']) -> None:
        # od_matrix is origin-destination matrix; DataFrame of all city pairs and their associated metrics
        self.od_matrix = od_matrix
        self.colsForHash = colsForHash

    def getStartState(self):
        return util.HSRSearchState(pd.DataFrame(columns=self.od_matrix.columns))

    def isGoalState(self, state: util.HSRSearchState):
        pass

    def getValidActions(self, state: util.HSRSearchState) -> pd.DataFrame:
        """
        Returns a DataFrame of valid "actions" that can be taken from the given state.
        An action is placing a new intercity rail segment, represented by a row in the returned DataFrame.
        """
        cols = self.od_matrix.columns
        merged = self.od_matrix.merge(
            state.getRailSegments(), how='left', indicator=True,
            on=['Origin', 'Dest'], suffixes=('', '_state')
        )
        mask = merged['_merge'] == 'left_only'
        actions = merged[mask]
        return actions[cols]

    def getSuccessors(self, state: util.HSRSearchState) -> List[tuple[util.HSRSearchState, float]]:
        actions = self.getValidActions(state)
        nextStatesAndCosts = actions.apply(lambda row: self.computeNextStateAndCost(state, row), axis='columns')
        return nextStatesAndCosts.tolist()

    def getOnlySuccessors(self, state: util.HSRSearchState) -> pd.Series[util.HSRSearchState]:
        actions = self.getValidActions(state)
        nextStates = actions.apply(lambda action: state.getSuccessor(action), axis='columns')  # type: ignore
        return nextStates

    def computeNextStateAndCost(self, currentState: util.HSRSearchState, action: pd.Series) -> tuple:
        newState = currentState.getSuccessor(action)
        newCost = self.getCostOfState(newState)
        return (newState, newCost)

    def getCostOfState(self, state: util.HSRSearchState):
        """
        Compute the cost of the given state. Lower numbers indicate more desirable states.

        NOTE: This refers to the UCS/search problem notion of "cost", NOT the monetary construction cost.
        """
        railSegments = state.getRailSegments()
        if railSegments.empty:
            return 0

        weight_pop = -0
        weight_time = +100
        weight_emissions = -0.4 * (1/1e8)

        score = 0
        score += weight_pop * (railSegments['pop_origin'] + railSegments['pop_dest'])
        score += weight_time * (railSegments['hsr_travel_time_hr'] -
                                railSegments['plane_travel_time_hr'] + 3)  # add 3 hrs for security, etc
        score += weight_emissions * railSegments['co2_g']

        return score.sum()

    def hashState(self, state: pd.DataFrame) -> frozenset:
        """
        Returns a `frozenset` representation of the given state, where each element of the frozenset is a rail segment.

        This turns the DataFrame into a hashable type that can be added to a `set` or compared easily.
        """
        # todo pass the frozenset to hash()?
        return frozenset(state[self.colsForHash].itertuples(name='rail', index=False))


class HSRProblem1(HighSpeedRailProblem):
    """
    Problem 1 is exploring the following question:
    Given a fixed construction budget, what is the optimal selection of intercity rail corridors to build?
    - The cost of building a certain rail segment is the negative of the "benefit"
    gained from the segment, as defined by the metrics we are interested in.
    - The goal state is any state in which the total monetary costs of constructing
    all present rail segments is equal to the predetermined budget.
    """

    def __init__(self, budget: float = 0) -> None:
        super()
        self.budget = budget

    def isGoalState(self, state):
        # todo reimplement with pandas
        constructionCost = 0
        for (origin, dest) in state:
            constructionCost += util.getRailCost(origin, dest)
        # todo this goes over budget but will change later
        return constructionCost >= self.budget

    def getCostOfActions(self, actions):
        # todo reimplement with pandas
        # todo costs are positive here
        cost = 0
        for (origin, dest) in actions:
            cost += evaluate_hsr(origin, dest)
        return cost


class HSRProblem1Pandas(HighSpeedRailProblemPandas):
    """
    Problem 1 is exploring the following question:
    Given a fixed construction budget, what is the optimal selection of intercity rail corridors to build?
    - The cost of building a certain rail segment is the negative of the "benefit"
    gained from the segment, as defined by the metrics we are interested in.
    - The goal state is any state in which the total monetary costs of constructing
    all present rail segments is equal to the predetermined budget.
    """

    def __init__(self, od_matrix: pd.DataFrame, colsForHash: List[str] = ['Origin', 'Dest'], budget: float = 0) -> None:
        super().__init__(od_matrix=od_matrix, colsForHash=colsForHash)
        self.budget = budget

    def isGoalState(self, state: pd.DataFrame):
        totalConstructionCost = state['construction_cost_usd'].sum()
        return totalConstructionCost >= self.budget


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


def HSRSearch(problem: HighSpeedRailProblem, heuristic=util.nullHeuristic):
    startState = problem.getStartState()
    # a node is a tuple of: (state, path-so-far, cost)
    # todo path-so-far should be hashable set?
    startNode = (startState, [], 0)
    # explored is set of states (each state is set of rail segments)
    explored = set()
    frontier = util.BetterPriorityQueue()
    frontier.push(startNode, 0)

    while not frontier.isEmpty():
        currentNode = frontier.pop()
        explored.add(currentNode[0])

        if problem.isGoalState(currentNode[0]):
            return currentNode[1]

        successors = problem.getSuccessors(currentNode[0])
        for (nextState, action, cost) in successors:
            pathSoFar = currentNode[1].copy()
            pathSoFar.append(action)

            # will be value of priority in the queue
            nextCost = currentNode[2] + cost
            nextNode = (nextState, pathSoFar, nextCost)
            nextPriority = nextCost + heuristic(nextState, problem)

            if nextState not in explored:
                nodeWithNextState = frontier.findNodeWithState(nextState)
                frontier.update(nodeWithNextState, nextNode, nextPriority)

    return False  # if frontier empty, no solution


def HSRSearchPandas(problem: HighSpeedRailProblemPandas, heuristic=util.nullHeuristic):
    # dict of state hashes to states
    hashToState: Dict[frozenset, util.HSRSearchState] = {}
    # dict of state hashes to state costs (eliminates need for frontier to hold costs)
    hashToCost: Dict[frozenset, float] = {}
    # explored is set of (hashed) states
    explored: Set[frozenset] = set()

    # a state is a dataframe of all intercity rail segments
    # cost is defined by the cost function of the problem
    startState = problem.getStartState()
    startStateHash = startState.getHash()
    hashToState[startStateHash] = startState
    hashToCost[startStateHash] = 0

    frontier = util.PriorityQueue()
    frontier.push(startStateHash, 0)

    while not frontier.isEmpty():
        currentStateHash = frontier.pop()
        currentState = hashToState[currentStateHash]
        # currentCost = hashToCost[currentStateHash] # todo needed?
        explored.add(currentStateHash)

        if problem.isGoalState(currentState):
            return currentState

        successors = problem.getOnlySuccessors(currentState)
        for nextState in successors:
            nextStateHash = nextState.getHash()
            if nextStateHash not in hashToState:
                hashToState[nextStateHash] = nextState
            # compute cost only if we've never encountered this state (hash) before
            if nextStateHash not in hashToCost:
                hashToCost[nextStateHash] = nextState.computeCost()

            nextCost = hashToCost[nextStateHash]
            nextPriority = nextCost + heuristic(nextState, problem)

            if nextStateHash not in explored:
                frontier.update(nextStateHash, nextPriority)

    return False  # if frontier empty, no solution


def evaluate_hsr(city1, city2) -> float:
    weight_pass = -0.4
    weight_time = -0.3
    # weight_cost = -0.3
    weight_emissions = -0.4

    score = 0

    row = util.getRow(city1, city2)

    pop_orign = row['pop_origin']
    pop_dest = row['pop_dest']
    score += weight_pass * (pop_orign + pop_dest)

    time_hsr = row['hsr_travel_time_hr']
    time_plane = row['plane_travel_time_hr']
    score += weight_time * (time_plane - time_hsr)

    score += weight_emissions * (row['co2_g'])
    return score


def main():
    # to cut down on the search space, only consider cities with a minimum population
    MIN_POP = 400e3
    MAX_POP = 1e99
    BUDGET_USD = 50e9

    # todo: decide: each city is either represented by its name or its IATA airport code (BOS, LAX...)
    od_matrix = pd.read_csv(cwd.joinpath('../data/algo_testing_data.csv'))
    mask = (od_matrix['pop_origin'] >= MIN_POP) &\
        (od_matrix['pop_origin'] <= MAX_POP) &\
        (od_matrix['pop_dest'] >= MIN_POP) &\
        (od_matrix['pop_dest'] <= MAX_POP)
    filtered_od = od_matrix[mask]

    hsr = HSRProblem1Pandas(od_matrix=filtered_od, budget=BUDGET_USD)

    solution = HSRSearchPandas(hsr)
    print('solution found! exporting to csv...')
    solution.to_csv(cwd.joinpath('../out/solution.csv'), index=False)


if __name__ == '__main__':
    # cProfile.run('main()', sort='cumulative') # for time analysis
    main()
