from __future__ import print_function
# bustersAgents.py
# ----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from builtins import range
from builtins import object
import util
from game import Agent
from game import Directions
from keyboardAgents import KeyboardAgent
import inference
import busters
import numpy as np
import os.path


class NullGraphics(object):
    "Placeholder for graphics"

    def initialize(self, state, isBlue=False):
        pass

    def update(self, state):
        pass

    def pause(self):
        pass

    def draw(self, state):
        pass

    def updateDistributions(self, dist):
        pass

    def finish(self):
        pass


class KeyboardInference(inference.InferenceModule):
    """
    Basic inference module for use with the keyboard.
    """

    def initializeUniformly(self, gameState):
        "Begin with a uniform distribution over ghost positions."
        self.beliefs = util.Counter()
        for p in self.legalPositions: self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observe(self, observation, gameState):
        noisyDistance = observation
        emissionModel = busters.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getPacmanPosition()
        allPossible = util.Counter()
        for p in self.legalPositions:
            trueDistance = util.manhattanDistance(p, pacmanPosition)
            if emissionModel[trueDistance] > 0:
                allPossible[p] = 1.0
        allPossible.normalize()
        self.beliefs = allPossible

    def elapseTime(self, gameState):
        pass

    def getBeliefDistribution(self):
        return self.beliefs


class BustersAgent(object):
    "An agent that tracks and displays its beliefs about ghost positions."

    def __init__(self, index=0, inference="ExactInference", ghostAgents=None, observeEnable=True,
                 elapseTimeEnable=True):
        inferenceType = util.lookup(inference, globals())
        self.inferenceModules = [inferenceType(a) for a in ghostAgents]
        self.observeEnable = observeEnable
        self.elapseTimeEnable = elapseTimeEnable

    def registerInitialState(self, gameState):
        "Initializes beliefs and inference modules"
        import __main__
        self.display = __main__._display
        for inference in self.inferenceModules:
            inference.initialize(gameState)
        self.ghostBeliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
        self.firstMove = True

    def observationFunction(self, gameState):
        "Removes the ghost states from the gameState"
        agents = gameState.data.agentStates
        gameState.data.agentStates = [agents[0]] + [None for i in range(1, len(agents))]
        return gameState

    def getAction(self, gameState):
        "Updates beliefs, then chooses an action based on updated beliefs."
        # for index, inf in enumerate(self.inferenceModules):
        #    if not self.firstMove and self.elapseTimeEnable:
        #        inf.elapseTime(gameState)
        #    self.firstMove = False
        #    if self.observeEnable:
        #        inf.observeState(gameState)
        #    self.ghostBeliefs[index] = inf.getBeliefDistribution()
        # self.display.updateDistributions(self.ghostBeliefs)
        return self.chooseAction(gameState)

    def chooseAction(self, gameState):
        "By default, a BustersAgent just stops.  This should be overridden."
        return Directions.STOP


class BustersKeyboardAgent(BustersAgent, KeyboardAgent):
    "An agent controlled by the keyboard that displays beliefs about ghost positions."

    def __init__(self, index=0, inference="KeyboardInference", ghostAgents=None):
        KeyboardAgent.__init__(self, index)
        BustersAgent.__init__(self, index, inference, ghostAgents)

    def getAction(self, gameState):
        return BustersAgent.getAction(self, gameState)

    def chooseAction(self, gameState):
        return KeyboardAgent.getAction(self, gameState)


from distanceCalculator import Distancer
from game import Actions
from game import Directions
import random, sys

'''Random PacMan Agent'''


class RandomPAgent(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    ''' Example of counting something'''

    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if (height == True):
                    food = food + 1
        return food

    ''' Print the layout'''

    def printGrid(self, gameState):
        table = ""
        ##print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table

    def chooseAction(self, gameState):
        move = Directions.STOP
        legal = gameState.getLegalActions(0)  ##Legal position from the pacman
        move_random = random.randint(0, 3)
        if (move_random == 0) and Directions.WEST in legal:  move = Directions.WEST
        if (move_random == 1) and Directions.EAST in legal: move = Directions.EAST
        if (move_random == 2) and Directions.NORTH in legal:   move = Directions.NORTH
        if (move_random == 3) and Directions.SOUTH in legal: move = Directions.SOUTH
        return move


class GreedyBustersAgent(BustersAgent):
    "An agent that charges the closest ghost."

    def registerInitialState(self, gameState):
        "Pre-computes the distance between every two points."
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    def chooseAction(self, gameState):
        """
        First computes the most likely position of each ghost that has
        not yet been captured, then chooses an action that brings
        Pacman closer to the closest ghost (according to mazeDistance!).

        To find the mazeDistance between any two positions, use:
          self.distancer.getDistance(pos1, pos2)

        To find the successor position of a position after an action:
          successorPosition = Actions.getSuccessor(position, action)

        livingGhostPositionDistributions, defined below, is a list of
        util.Counter objects equal to the position belief
        distributions for each of the ghosts that are still alive.  It
        is defined based on (these are implementation details about
        which you need not be concerned):

          1) gameState.getLivingGhosts(), a list of booleans, one for each
             agent, indicating whether or not the agent is alive.  Note
             that pacman is always agent 0, so the ghosts are agents 1,
             onwards (just as before).

          2) self.ghostBeliefs, the list of belief distributions for each
             of the ghosts (including ghosts that are not alive).  The
             indices into this list should be 1 less than indices into the
             gameState.getLivingGhosts() list.
        """
        pacmanPosition = gameState.getPacmanPosition()
        legal = [a for a in gameState.getLegalPacmanActions()]
        livingGhosts = gameState.getLivingGhosts()
        livingGhostPositionDistributions = \
            [beliefs for i, beliefs in enumerate(self.ghostBeliefs)
             if livingGhosts[i + 1]]
        return Directions.EAST


class BasicAgentAA(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        self.countActions = 0

    ''' Example of counting something'''

    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if (height == True):
                    food = food + 1
        return food

    ''' Print the layout'''

    def printGrid(self, gameState):
        table = ""
        # print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table

    def printInfo(self, gameState):
        print("---------------- TICK ", self.countActions, " --------------------------")
        # Map size
        width, height = gameState.data.layout.width, gameState.data.layout.height
        print("Width: ", width, " Height: ", height)
        # Pacman position
        print("Pacman position: ", gameState.getPacmanPosition())
        # Legal actions for Pacman in current position
        print("Legal actions: ", gameState.getLegalPacmanActions())
        # Pacman direction
        print("Pacman direction: ", gameState.data.agentStates[0].getDirection())
        # Number of ghosts
        print("Number of ghosts: ", gameState.getNumAgents() - 1)
        # Alive ghosts (index 0 corresponds to Pacman and is always false)
        print("Living ghosts: ", gameState.getLivingGhosts())
        # Ghosts positions
        print("Ghosts positions: ", gameState.getGhostPositions())
        # Ghosts directions
        print("Ghosts directions: ",
              [gameState.getGhostDirections().get(i) for i in range(0, gameState.getNumAgents() - 1)])
        # Manhattan distance to ghosts
        print("Ghosts distances: ", gameState.data.ghostDistances)
        # Pending pac dots
        print("Pac dots: ", gameState.getNumFood())
        # Manhattan distance to the closest pac dot
        print("Distance nearest pac dots: ", gameState.getDistanceNearestFood())
        # Map walls
        print("Map:")
        print(gameState.getWalls())
        # Score
        print("Score: ", gameState.getScore())

    def chooseAction(self, gameState):
        self.countActions = self.countActions + 1
        self.printInfo(gameState)
        move = Directions.STOP
        legal = gameState.getLegalActions(0)  ##Legal position from the pacman
        move_random = random.randint(0, 3)
        if (move_random == 0) and Directions.WEST in legal:  move = Directions.WEST
        if (move_random == 1) and Directions.EAST in legal: move = Directions.EAST
        if (move_random == 2) and Directions.NORTH in legal:   move = Directions.NORTH
        if (move_random == 3) and Directions.SOUTH in legal: move = Directions.SOUTH
        return move

    def printLineData(self, gameState):
        return "XXXXXXXXXX"


class QLearningAgent(BustersAgent):

    # Initialization
    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        self.epsilon = 0.05
        self.alpha = 0.8
        self.discount = 0.3
        self.actions = {"North": 0, "East": 1, "South": 2, "West": 3}
        if os.path.exists("qtable.txt"):
            self.table_file = open("qtable.txt", "r+")
            self.q_table = self.readQtable()
        else:
            self.table_file = open("qtable.txt", "w+")
            # "*** CHECK: NUMBER OF ROWS IN QTABLE DEPENDS ON THE NUMBER OF STATES ***"
            self.initializeQtable(9)

    def initializeQtable(self, nrows):
        "Initialize qtable"
        self.q_table = np.zeros((nrows, len(self.actions)))

    def readQtable(self):
        "Read qtable from disc"
        table = self.table_file.readlines()
        q_table = []

        for i, line in enumerate(table):
            row = line.split()
            row = [float(x) for x in row]
            q_table.append(row)

        return q_table

    def writeQtable(self):
        "Write qtable to disc"
        self.table_file.seek(0)
        self.table_file.truncate()

        for line in self.q_table:
            for item in line:
                self.table_file.write(str(item) + " ")
            self.table_file.write("\n")

    def printQtable(self):
        "Print qtable"
        for line in self.q_table:
            print(line)
        print("\n")

    def __del__(self):
        "Destructor. Invokation at the end of each episode"
        self.writeQtable()
        self.table_file.close()

    def computePosition(self, state):
        """
        Compute the row of the qtable for a given state.
        """

        "*** YOUR CODE HERE ***"
        closestXY = []
        value = 100
        aux = 0
        for i in range(len(state.data.ghostDistances)):
            if state.data.ghostDistances[i] == None:
                aux = i
            elif state.data.ghostDistances[i] < value:
                value = state.data.ghostDistances[i]
                aux = i
        closestXY.append(value)
        closestX = state.getGhostPositions()[aux][0]
        closestY = state.getGhostPositions()[aux][1]

        if closestX == state.getPacmanPosition()[0] and closestY < state.getPacmanPosition()[1]:
            closestXY.append(0)  # down
        elif closestX == state.getPacmanPosition()[0] and closestY > state.getPacmanPosition()[1]:
            closestXY.append(1)  # up
        elif closestY == state.getPacmanPosition()[1] and closestX < state.getPacmanPosition()[0]:
            closestXY.append(2)  # left
        elif closestY == state.getPacmanPosition()[1] and closestX > state.getPacmanPosition()[0]:
            closestXY.append(3)  # right
        elif closestX > state.getPacmanPosition()[0] and closestY > state.getPacmanPosition()[1]:
            closestXY.append(4)  # upright
        elif closestX > state.getPacmanPosition()[0] and closestY < state.getPacmanPosition()[1]:
            closestXY.append(5)  # downright
        elif closestX < state.getPacmanPosition()[0] and closestY > state.getPacmanPosition()[1]:
            closestXY.append(6)  # upleft
        elif closestX < state.getPacmanPosition()[0] and closestY < state.getPacmanPosition()[1]:
            closestXY.append(7)  # downleft
        else:
            closestXY.append(8)  # terminal

        legal = None

        if state.getLegalPacmanActions() == ["North", "Stop"]:
            legal = 0
        elif state.getLegalPacmanActions() == ["South", "Stop"]:
            legal = 1
        elif state.getLegalPacmanActions() == ["East", "Stop"]:
            legal = 2
        elif state.getLegalPacmanActions() == ["West", "Stop"]:
            legal = 3
        elif state.getLegalPacmanActions() == ["North", "South", "Stop"]:
            legal = 4
        elif state.getLegalPacmanActions() == ["North", "East", "Stop"]:
            legal = 5
        elif state.getLegalPacmanActions() == ["North", "West", "Stop"]:
            legal = 6
        elif state.getLegalPacmanActions() == ["South", "East", "Stop"]:
            legal = 7
        elif state.getLegalPacmanActions() == ["South", "West", "Stop"]:
            legal = 8
        elif state.getLegalPacmanActions() == ["East", "West", "Stop"]:
            legal = 9
        elif state.getLegalPacmanActions() == ["North", "South", "East", "Stop"]:
            legal = 10
        elif state.getLegalPacmanActions() == ["North", "South", "West", "Stop"]:
            legal = 11
        elif state.getLegalPacmanActions() == ["South", "East", "West", "Stop"]:
            legal = 12
        elif state.getLegalPacmanActions() == ["North", "East", "West", "Stop"]:
            legal = 13
        elif state.getLegalPacmanActions() == ["North", "South", "East", "West", "Stop"]:
            legal = 14
        elif state.getLegalPacmanActions() == ["Stop"]:
            legal = 15
        else:
            legal = 16

        return closestXY[1] * 16 + (legal)  # (closestXY[1]-1)*20 + closestXY[0]

    def getQValue(self, state, action):

        """
            Returns Q(state,action)
            Should return 0.0 if we have never seen a state
            or the Q node value otherwise
        """
        position = self.computePosition(state)
        action_column = self.actions[action]
        return self.q_table[position][action_column]

    def computeValueFromQValues(self, state):
        """
            Returns max_action Q(state,action)
            where the max is over legal actions.  Note that if
            there are no legal actions, which is the case at the
            terminal state, you should return a value of 0.0.
        """
        legalActions = state.getLegalPacmanActions()
        if 'Stop' in legalActions: legalActions.remove("Stop")
        if len(legalActions) == 0:
            return 0
        return max(self.q_table[self.computePosition(state)])

    def computeActionFromQValues(self, state):
        """
            Compute the best action to take in a state.  Note that if there
            are no legal actions, which is the case at the terminal state,
            you should return None.
        """
        legalActions = state.getLegalPacmanActions()
        if 'Stop' in legalActions: legalActions.remove("Stop")
        if len(legalActions) == 0:
            return None

        best_actions = [legalActions[0]]
        best_value = self.getQValue(state, legalActions[0])
        for action in legalActions:
            value = self.getQValue(state, action)
            if value == best_value:
                best_actions.append(action)
            if value > best_value:
                best_actions = [action]
                best_value = value

        return random.choice(best_actions)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
        """

        # Pick Action
        legalActions = state.getLegalPacmanActions()
        if 'Stop' in legalActions: legalActions.remove("Stop")
        action = None

        if len(legalActions) == 0:
            return action

        flip = util.flipCoin(self.epsilon)

        if flip:
            return random.choice(legalActions)
        return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
            The parent class calls this to observe a
            state = action => nextState and reward transition.
            You should do your Q-Value update here

        Q-Learning update:

        if terminal_state:
        Q(state,action) <- (1-self.alpha) Q(state,action) + self.alpha * (r + 0)
        else:
        Q(state,action) <- (1-self.alpha) Q(state,action) + self.alpha * (r + self.discount * max a' Q(nextState, a'))
        
        """

        "*** YOUR CODE HERE ***"
        if len(state.data.ghostDistances) == 0:
            q_value = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * (reward + 0)
        else:
            q_value = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * (
                        reward + self.discount * max(self.q_table[self.computePosition(nextState)]))

        self.q_table[self.computePosition(state)][self.actions[action]] = q_value

    def getPolicy(self, state):
        "Return the best action in the qtable for a given state"
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        "Return the highest q value for a given state"
        return self.computeValueFromQValues(state)

    def getReward(self, state, action, nextstate):
        "Return the obtained reward"

        # auxiliar needed variable
        closestXY = []
        value = 100
        aux = 0
        for i in range(len(state.data.ghostDistances)):
            if state.getLivingGhosts()[i + 1] == True:
                if state.data.ghostDistances[i] == None:
                    aux = i
                elif state.data.ghostDistances[i] < value:
                    value = state.data.ghostDistances[i]
                    aux = i
        closestXY.append(value)
        closestX = state.getGhostPositions()[aux][0]
        closestY = state.getGhostPositions()[aux][1]

        if closestX == state.getPacmanPosition()[0] and closestY < state.getPacmanPosition()[1]:
            closestXY.append(0)  # down
        elif closestX == state.getPacmanPosition()[0] and closestY > state.getPacmanPosition()[1]:
            closestXY.append(1)  # up
        elif closestY == state.getPacmanPosition()[1] and closestX < state.getPacmanPosition()[0]:
            closestXY.append(2)  # left
        elif closestY == state.getPacmanPosition()[1] and closestX > state.getPacmanPosition()[0]:
            closestXY.append(3)  # right
        elif closestX > state.getPacmanPosition()[0] and closestY > state.getPacmanPosition()[1]:
            closestXY.append(4)  # upright
        elif closestX > state.getPacmanPosition()[0] and closestY < state.getPacmanPosition()[1]:
            closestXY.append(5)  # downright
        elif closestX < state.getPacmanPosition()[0] and closestY > state.getPacmanPosition()[1]:
            closestXY.append(6)  # upleft
        elif closestX < state.getPacmanPosition()[0] and closestY < state.getPacmanPosition()[1]:
            closestXY.append(7)  # downleft
        else:
            closestXY.append(8)  # terminal

        ###############################################################

        ### auxiliary booleans
        bool1 = (closestXY[1] == 0) and ('South' not in state.getLegalPacmanActions())  # want to go straight down
        bool2 = (closestXY[1] == 1) and ('North' not in state.getLegalPacmanActions())  # want to go straight up
        # editar estos
        bool3 = closestXY[1] == 2 and ('West' not in state.getLegalPacmanActions())  # want to go straight left
        bool4 = closestXY[1] == 3 and ('East' not in state.getLegalPacmanActions())  # want to go straight right

        bool5 = (closestXY[1] == 5 or closestXY[1] == 7) and ('South' not in state.getLegalPacmanActions())  # want to go straight down
        bool6 = (closestXY[1] == 4 or closestXY[1] == 6) and ('North' not in state.getLegalPacmanActions())


        # auxiliar pacman next state positon and closest ghost next position
        pacmanx_next = nextstate.getPacmanPosition()[0]
        pacmany_next = nextstate.getPacmanPosition()[1]
        #closestx_next = nextstate.data.ghostDistances.index(min(filter(lambda x: x is not None, state.data.ghostDistances)))[0]
        #closesty_next = nextstate.data.ghostDistances.index(min(filter(lambda x: x is not None, state.data.ghostDistances)))[1]


        # reward function
        # if score goes down
        if state.data.score > nextstate.data.score:
            # if none of the "blocked" cases happen
            if not (bool1 or bool2 or bool3 or bool4):
                if state.data.ghostDistances.count(None) != len(
                        state.data.ghostDistances) and nextstate.data.ghostDistances.count(None) != len(
                        nextstate.data.ghostDistances):

                    # if we are getting near we reward
                    if min(filter(lambda x: x is not None, state.data.ghostDistances)) > min(
                            filter(lambda x: x is not None, nextstate.data.ghostDistances)):
                        reward = 0.5
                    elif min(filter(lambda x: x is not None, state.data.ghostDistances)) > min(
                            filter(lambda x: x is not None, nextstate.data.ghostDistances)):
                        reward = 0.1
                    # if we are going opposite direction we penalize
                    elif min(filter(lambda x: x is not None, state.data.ghostDistances)) < min(
                            filter(lambda x: x is not None, nextstate.data.ghostDistances)):
                        reward = -0.1
                    else:
                        reward = 0.0
            else:
                # if we are blocked either up/down
                # and next action X stays the same we are doing bad
                if (bool1 or bool2) and closestX == pacmanx_next:
                    reward = -0.5
                # if we are blocked either left/right and Y stays the same we are doing bad
                elif (bool3 or bool4) and closestY == pacmany_next:
                    reward = -0.5
                elif bool5 or bool6:
                    if closestX < state.getPacmanPosition()[0]:
                        if action == "West":
                            reward = 0.5
                        else:
                            reward = 0
                    if closestX > state.getPacmanPosition()[0]:
                        if action == "East":
                            reward = 0.5
                        else:
                            reward = 0
                else:
                    reward = 0

        else:
            # editar, alomejor no siempre uno
            # si se come una pelota de comida menos porque una vez se la come fuera
            reward = 1
        print(bool1)
        print(bool2)
        print(bool3)
        print(bool4)
        print(bool5)
        print(bool6)

        print(closestXY)
        print(reward)
        return reward
