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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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

        score = successorGameState.getScore()
        FOOD_PROXIMITY = 10
        ATE_FOOD = 250

        newFoodList = newFood.asList()

        #reward for eating food
        if len(newFoodList) < len(currentGameState.getFood().asList()):
            score += ATE_FOOD
            
        #find that foodd
        if newFoodList:
            min_food_dist = 1000000
            for foodPos in newFoodList:
                dist = manhattanDistance(newPos, foodPos)
                if dist < min_food_dist:
                    min_food_dist = dist
            
            score += FOOD_PROXIMITY / min_food_dist
            
        return score

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        bestAction = None
        max_value = -100000
        
        
        for act in gameState.getLegalActions(0):
            #next is ghost agent move
            score = self.get_value(gameState.generateSuccessor(0, act), 1, 0)
            
            if score > max_value:
                max_value = score
                bestAction = act
                
        return bestAction
    
    def get_value(self, state: GameState, index, depth):
        
        # checks terminal state or maximum search depth
        if depth == self.depth or state.isLose() or state.isWin():
            return self.evaluationFunction(state)
            
        # search function. 
        # is_max_node for Pacman - agent 0, False for ghosts agent != 0.
        is_max_node = (index == 0)
        return self.search_value(is_max_node, state, index, depth)
    
    def search_value(self, is_max_node, state: GameState, index, depth):
        #initial best score
        if is_max_node:
            bestScore = -100000
        else:
            bestScore = 100000
        
        #determine next agent and depth
        if index == state.getNumAgents() - 1:
            nextIndex = 0
            nextDepth = depth + 1
        else:
            nextIndex = index + 1
            nextDepth = depth
        
        for action in state.getLegalActions(index):
            #recursive call
            score = self.get_value(state.generateSuccessor(index, action), nextIndex, nextDepth)
        
            if is_max_node:
                if score > bestScore:
                    bestScore = score
            else:
                if score < bestScore:
                    bestScore = score
            
        return bestScore


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        bestScore = -100000
        bestAction = None
        alpha = -100000
        beta = 100000
        
        #initial call for max node - pacman
        for act in gameState.getLegalActions(0):   
            #next is ghost agent move         
            score = self.alpha_beta(alpha, beta, gameState.generateSuccessor(0, act), 0, 1)
            
            if score > bestScore:
                bestScore = score
                bestAction = act
            alpha = max(alpha, bestScore)
            
        return bestAction
    
    def alpha_beta(self, alpha, beta, state: GameState, depth, index):  

        #determine next agent and depth      
        nextAgent = (index + 1) % state.getNumAgents()
        nextDepth = depth
        if nextAgent == 0:
            nextDepth += 1  
            
        # checks terminal state or maximum search depth
        if depth == self.depth or state.isLose() or state.isWin():
            return self.evaluationFunction(state)
        
        # search function. 
        # is_max_node for Pacman - agent 0, False for ghosts agent != 0.
        return self.prune_get_val(alpha, beta, state, depth, index, nextDepth, nextAgent)

    def prune_get_val(self, alpha, beta, state: GameState, depth, index, nextDepth, nextAgent):
        is_max_agent = (index == 0)

        if is_max_agent:
            minmax = -100000
        else:
            minmax = 100000
        
        for action in state.getLegalActions(index):
            #recursive call
            prune_get_val = self.alpha_beta(alpha, beta, state.generateSuccessor(index, action), nextDepth, nextAgent)

            if is_max_agent:
                if prune_get_val > minmax:
                    #best score for max node
                    minmax = prune_get_val 
                if minmax > beta:
                    #prune
                    return minmax
                #beta cut-off
                alpha = max(alpha, minmax)
            else:
                if prune_get_val < minmax:
                    #best score for min node
                    minmax = prune_get_val
                if minmax < alpha:
                    #prune
                    return minmax
                #alpha cut-off
                beta = min(beta, minmax)
        return minmax

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        bestAction = None
        bestValue = -float('inf')

        for action in gameState.getLegalActions(0):
            succ = gameState.generateSuccessor(0, action)
            # agent 1 (first ghost), still at search depth 0 
            value = self.expectimax_value(succ, 1, 0)
            if value > bestValue:
                bestValue = value
                bestAction = action

        return bestAction

    def expectimax_value(self, state: GameState, index: int, depth: int):
        # return heuristic value if we've reached the depth limit
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        numAgents = state.getNumAgents()

        if index == 0:
            best = -float('inf')
            for action in state.getLegalActions(0):
                succ = state.generateSuccessor(0, action)
                # if there is more than one agent, the next agent after pm
                # a ghost. if only pm then nextIndex wraps back to 0 and we advance the ply.
                nextIndex = 1 if numAgents > 1 else 0
                nextDepth = depth
                if nextIndex == 0:
                    # increment, wrapped back to pm
                    nextDepth = depth + 1
                val = self.expectimax_value(succ, nextIndex, nextDepth)
                if val > best:
                    best = val
            return best
        else:
            # expectimax value of a chance node is the avg of its successors.
            actions = state.getLegalActions(index)
            if len(actions) == 0:
                return self.evaluationFunction(state)

            total = 0.0
            for action in actions:
                succ = state.generateSuccessor(index, action)
                # if this ghost is the last agent, the next will be
                # pm -> we've completed one ply cycle + increment depth
                if index == numAgents - 1:
                    nextIndex = 0
                    nextDepth = depth + 1
                else:
                    nextIndex = index + 1
                    nextDepth = depth

                val = self.expectimax_value(succ, nextIndex, nextDepth)
                total += val

            return total / len(actions)

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
