# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        # Evaluation function
        # Eval(s) = w1f1(s) + w2f2(s) + ... + wnfn(s)
        # What features of f(s) to use?

        # f1: Number of food pellets left (less = good -> inverse)
        # f2: Distance to the closest food pellet (closer = good -> inverse)
        # f3: Distance to closest ghost (further = good unless ghost is scared)
        w1 = 100
        w2 = 10
        w3 = 1
        score = 0

        if action == 'Stop':
            score -= 50

        if (successorGameState.getNumFood() > 0):
            f1 = 1/(successorGameState.getNumFood())
        else:
            f1 = 0


        foodPos = newFood.asList()
        if len(foodPos) > 1:
            if min([manhattanDistance(newPos, food) for food in foodPos]) > 0:
                f2 = 1/(min([manhattanDistance(newPos, food) for food in foodPos]))
        elif len(foodPos) == 1:
            f2 = 1/manhattanDistance(newPos, foodPos[0])
        else:
            f2 = 0
        
        ghostPos = [ghostState.getPosition() for ghostState in newGhostStates]
        f3 = min([manhattanDistance(newPos, ghost) for ghost in ghostPos])
        if any(time > 0 for time in newScaredTimes):
            w3 = 0
            # print("w3 is 0")
        else:
            if f3 < 5:
                w3 = 5
                # print("w3 is 10")
            elif f3 < 10:
                w3 = 3
                # print("w3 is 5")
            else:
                w3 = 1
                # print("w3 is 1")


        # print(f"sucessorGameState.getScore() {successorGameState.getScore()}")
        # print(f"w1: {w1}, f1: {f1}, w1*f1: {w1 * f1}")
        # print(f"w2: {w2}, f2: {f2}, w2*f2: {w2 * f2}")
        # print(f"w3: {w3}, f3: {f3}, w3*f3: {w3 * f3}")
        return successorGameState.getScore() + score + (w1 * f1) + (w2 * f2) + (w3 * f3)
        
        

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        return self.value(gameState, 0, 0)[0] # [action, value]

    def value(self, gameState, agentIndex, depth):
        # if the state is a terminal state, return the state's utility
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return None, self.evaluationFunction(gameState)
        
        # if the next agent is MAX (Pacman), return maxValue(state)
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth)
        # if the next agent is MIN (Ghost), return minValue(state)
        else:
            return self.minValue(gameState, agentIndex, depth)

    def maxValue(self, gameState, agentIndex, depth):
        max_action = None
        max_value = float('-inf')

        for action in gameState.getLegalActions(agentIndex):
            value = self.value(gameState.generateSuccessor(agentIndex, action), 
                               self.getNextAgent(agentIndex, gameState), 
                               self.getNextDepth(agentIndex, depth, gameState))[1]

            if value > max_value:
                max_action = action
                max_value = value

        return max_action, max_value
    
    def minValue(self, gameState, agentIndex, depth):
        min_action = None
        min_value = float('+inf')

        for action in gameState.getLegalActions(agentIndex):
            value = self.value(gameState.generateSuccessor(agentIndex, action), 
                                       self.getNextAgent(agentIndex, gameState), 
                                       self.getNextDepth(agentIndex, depth, gameState))[1]

            if value < min_value:
                min_action = action
                min_value = value

        return min_action, min_value
    
    # Helper function to determine the next agent's index
    def getNextAgent(self, agentIndex, gameState):
        if agentIndex == gameState.getNumAgents() - 1:
            return 0
        return agentIndex + 1

    # Helper function to determine if we need to go a level deeper
    # (after all agents have moved, which is after the last ghost's turn)
    def getNextDepth(self, agentIndex, depth, gameState):
        if agentIndex == gameState.getNumAgents() - 1:
            return depth + 1
        return depth

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        return self.value(gameState, 0, 0, float('-inf'), float('+inf'))[0] # [action, value]

    def value(self, gameState, agentIndex, depth, alpha, beta):
        # if the state is a terminal state, return the state's utility
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return None, self.evaluationFunction(gameState)
        
        # if the next agent is MAX (Pacman), return maxValue(state)
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth, alpha, beta)
        # if the next agent is MIN (Ghost), return minValue(state)
        else:
            return self.minValue(gameState, agentIndex, depth, alpha, beta)

    def maxValue(self, gameState, agentIndex, depth, alpha, beta):
        max_action = None
        max_value = float('-inf')

        for action in gameState.getLegalActions(agentIndex):
            value = self.value(gameState.generateSuccessor(agentIndex, action), 
                               self.getNextAgent(agentIndex, gameState), 
                               self.getNextDepth(agentIndex, depth, gameState),
                               alpha,
                               beta)[1]

            if value > max_value:
                max_action = action
                max_value = value

            if beta < max_value:
                return max_action, max_value
            
            alpha = max(alpha, max_value)

        return max_action, max_value
    
    def minValue(self, gameState, agentIndex, depth, alpha, beta):
        min_action = None
        min_value = float('+inf')

        for action in gameState.getLegalActions(agentIndex):
            value = self.value(gameState.generateSuccessor(agentIndex, action), 
                                       self.getNextAgent(agentIndex, gameState), 
                                       self.getNextDepth(agentIndex, depth, gameState),
                                       alpha,
                                       beta)[1]

            if value < min_value:
                min_action = action
                min_value = value
            
            if alpha > min_value:
                return min_action, min_value
        
            beta = min(beta, min_value)

        return min_action, min_value
    
    # Helper function to determine the next agent's index
    def getNextAgent(self, agentIndex, gameState):
        if agentIndex == gameState.getNumAgents() - 1:
            return 0
        return agentIndex + 1

    # Helper function to determine if we need to go a level deeper
    # (after all agents have moved, which is after the last ghost's turn)
    def getNextDepth(self, agentIndex, depth, gameState):
        if agentIndex == gameState.getNumAgents() - 1:
            return depth + 1
        return depth

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        return self.value(gameState, 0, 0)[0] # [action, value]

    def value(self, gameState, agentIndex, depth):
        # if the state is a terminal state, return the state's utility
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return None, self.evaluationFunction(gameState)
        
        # if the next agent is MAX (Pacman), return maxValue(state)
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth)
        # if the next agent is MIN (Ghost), return minValue(state)
        else:
            return self.chanceValue(gameState, agentIndex, depth)

    def maxValue(self, gameState, agentIndex, depth):
        max_action = None
        max_value = float('-inf')

        for action in gameState.getLegalActions(agentIndex):
            value = self.value(gameState.generateSuccessor(agentIndex, action), 
                               self.getNextAgent(agentIndex, gameState), 
                               self.getNextDepth(agentIndex, depth, gameState))[1]

            if value > max_value:
                max_action = action
                max_value = value

        return max_action, max_value
    
    def chanceValue(self, gameState, agentIndex, depth):
        chance_action = None
        chance_value = 0

        for action in gameState.getLegalActions(agentIndex):
            value = self.value(gameState.generateSuccessor(agentIndex, action), 
                                       self.getNextAgent(agentIndex, gameState), 
                                       self.getNextDepth(agentIndex, depth, gameState))[1]

            chance_value += value * (1.0 / len(gameState.getLegalActions(agentIndex)))
        return chance_action, chance_value
    
    # Helper function to determine the next agent's index
    def getNextAgent(self, agentIndex, gameState):
        if agentIndex == gameState.getNumAgents() - 1:
            return 0
        return agentIndex + 1

    # Helper function to determine if we need to go a level deeper
    # (after all agents have moved, which is after the last ghost's turn)
    def getNextDepth(self, agentIndex, depth, gameState):
        if agentIndex == gameState.getNumAgents() - 1:
            return depth + 1
        return depth

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
