from hsr import HighSpeedRailProblem
import util

# defining the search algorithm


def HSRSearch(problem: HighSpeedRailProblem, heuristic=util.nullHeuristic):
    startState = problem.getStartState()
    startNode = (startState, [])
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


def evaluate_hsr(city1, city2):
    weight_pass = 0.4
    weight_time = 0.3
    # weight_cost = -0.3
    weight_emissions = 0.4

    score = 0
    score += weight_pass * util.getPassServed(city1, city2)
    score += weight_emissions * util.getEmissionsSaved(city1, city2)
