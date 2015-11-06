# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

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
    some Directions.X for some X in the set {North, South, West, East, Stop}
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

    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    foodDistance = []
    currentFood = currentGameState.getFood().asList()
    pacmanPos = list(successorGameState.getPacmanPosition())

    for ghostState in newGhostStates:
        if ghostState.getPosition() == tuple(pacmanPos) and ghostState.scaredTimer is 0:
            return -float("inf") 

    for food in currentFood:
        x = -1*abs(food[0] - pacmanPos[0])
        y = -1*abs(food[1] - pacmanPos[1])
        foodDistance.append(x+y) 
        #foodDistance.append(manhattanDistance(food, pacmanPos))
    
    return max(foodDistance)
 
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
    
    def maxValue(self, state, depth, agentIndex):
      value = float("-inf") 
      if state.isWin() or state.isLose():
        return self.evaluationFunction(state) #utilidade das folhas

      for action in state.getLegalActions(agentIndex):
        successor = state.generateSuccessor(agentIndex, action)
          
        temp = self.minValue(successor, depth, 1)
        if temp > value:
          value = temp
          maxAction = action

      if depth == 1:
        return maxAction
      else:
        return value

    def minValue(self, state, depth, agentIndex):
      value= float("inf")
      numAgents = state.getNumAgents()

      if state.isWin() or state.isLose():
        return self.evaluationFunction(state)
      for action in state.getLegalActions(agentIndex):
        successor = state.generateSuccessor(agentIndex, action)
        
        if agentIndex == numAgents - 1:
          if depth == self.depth:
            temp = self.evaluationFunction(successor)
          else:
            temp = self.maxValue(successor, depth+1, 0)
        else:
          temp = self.minValue(successor, depth, agentIndex+1)

        if temp < value:
          value = temp

      return value

    def getAction(self, state):

        maxAction = self.maxValue(state, 1, 0)
        return maxAction
   
class AlphaBetaAgent(MultiAgentSearchAgent):
  
  def maxValue(self, state, depth, agentIndex, alpha, beta):
      
      value= float("-inf")
      if state.isWin() or state.isLose():
        return self.evaluationFunction(state)

      for action in state.getLegalActions(agentIndex):
        successor = state.generateSuccessor(agentIndex, action)
        
        temp = self.minValue(successor, depth, 1, alpha, beta)

        if temp > beta:
          return temp

        if temp > value:
          value = temp
          maxAction = action

        alpha = max(alpha, value)

      if depth == 1:
        return maxAction
      else:
        return value

  def minValue(self, state, depth, agentIndex, alpha, beta):
      value= float("inf")

      numAgents = state.getNumAgents()

      if state.isWin() or state.isLose():
        return self.evaluationFunction(state)

      for action in state.getLegalActions(agentIndex):
        successor = state.generateSuccessor(agentIndex, action)
      
        if agentIndex == numAgents - 1:
          if depth == self.depth:
            temp = self.evaluationFunction(successor)
          else:
            temp = self.maxValue(successor, depth+1, 0, alpha, beta)

        else:
          temp = self.minValue(successor, depth, agentIndex+1, alpha, beta)

        if temp < alpha:
          return temp
        if temp < value:
          value = temp
          minAction = action

        beta = min(beta, value)
      return value

  def getAction(self, state):
      
        maxAction = self.maxValue(state, 1, 0, float("-inf"), float("inf"))
        return maxAction
  
class ExpectimaxAgent(MultiAgentSearchAgent):
  
  def maxValue(self, state, depth, agentIndex):
      value= float("-inf")
      if state.isWin() or state.isLose():
        return self.evaluationFunction(state)


      for action in state.getLegalActions(0):
        successor = state.generateSuccessor(0, action)
        
        temp = self.minValue(successor, depth, 1)
        if temp > value:
          value = temp
          maxAction = action

      if depth == 1:
        return maxAction
      else:
        return value

  def minValue(self, state, depth, agentIndex):

      value= 0
      numAgents = state.getNumAgents()
      
      if state.isWin() or state.isLose():
        return self.evaluationFunction(state)
      legalActions = state.getLegalActions(agentIndex)
      
      prob = 1/len(legalActions)
      for action in legalActions:
        successor = state.generateSuccessor(agentIndex, action)
        
        if agentIndex == numAgents - 1:
          if depth == self.depth:
            temp = self.evaluationFunction(successor)
          else:
            temp = self.maxValue(successor, depth+1, 0)
        else:
          temp = self.minValue(successor, depth, agentIndex+1)

        value += temp * prob

      return value

  def getAction(self, state):

        return self.maxValue(state, 1, 0)

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

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()
