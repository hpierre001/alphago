# -*- coding: utf-8 -*-
'''
'''

import Goban
from playerInterface import *

import numpy as np
import matplotlib.pyplot as plt

import random as rd

C_PUCT = 1

class Node():

    def __init__(self, move=None, parent=None, p=1, n=0, w=0):
        self._move = move
        self._parent = parent
        self._children = {}
        self._stats = {"p":p, "n":n, "w":w}

    def addChild(self, move):
        self._children[move] = Node(move, self)
        return self._children[move]

    def delChild(self, move):
        del self._children[move]

    def getChild(self, move):
        return self._children[move]

    def modifyStats(self, p=None, n=None, w=None):
        if p != None:
            self._stats["p"] = p

        if n != None:
            self._stats["n"] = n

        if w != None:
            self._stats["w"] = w

    def getStats(self):
        return self._stats

    def getStat(self, stat):
        return self._stats[stat]

    def getMove(self):
        return self._move

    def getParent(self):
        return self._parent

    def delParent(self):
        self._parent = None

    def getChildByMove(self, move):
        return self._children[move]

    def addStat(self, stat, val):
        self._stats[stat] += val
        return self._stats[stat]

    def isLeaf(self):
        return self._children == {}

    def Q(self):
        if self._stats["n"] == 0:
            return 0
        return self._stats["w"]/self._stats["n"]

    def U(self):
        return self._stats["p"] * np.sqrt(self._parent._stats["n"]) / (self._stats["n"] + 1) * C_PUCT

    def randChild(self):
        return rd.choices(list(self._children), wheights=[child.Q()+child.U() for child in self._children])[0]

    def _mostVisitedChild(self):
        max_child = None
        max_n = 0
        for move, child in self._children.items():
            if child._stats["n"] > max_n:
                max_child = child
                max_n = max_child._stats["n"]
        return max_child

    def getBetterChild(self, tau = 0):
        if tau == 0:
            return self._mostVisitedChild()
        N = 0
        for move, child in self._children.items():
            N += np.power(child._stats["n"], 1/tau)
        return self._children[rd.choices(list(self._children), weights=[child._stats["p"] * np.power(child._stats["n"], 1/tau)/N for child in self._children.values()])[0]]


class myPlayer(PlayerInterface):
    ''' Example of a random player for the go. The only tricky part is to be able to handle
    the internal representation of moves given by legal_moves() and used by push() and
    to translate them to the GO-move strings "A1", ..., "J8", "PASS". Easy!
    '''

    def __init__(self):
        self._board = Goban.Board()
        self._mycolor = None
        self._node = Node()

    def getPlayerName(self):
        return "mcts Player"

    def _rollout(self):
        count = 0
        while not self._board.is_game_over():
            moves = self._board.weak_legal_moves()
            rd.shuffle(moves)
            i = 0
            while not self._board.push(moves[i]):
                self._board.pop()
                i+=1
            count+=1

        ans = int(self._board.final_go_score()[0].lower() == self._mycolor)

        for i in range(count):
            self._board.pop()

        return ans

    def _buildTree(self):
        count = 0
        node = self._node
        while not node.isLeaf:
            node = node.randChild()
            self._board.push(node.getMove())
            count += 1

        moves = self._board.weak_legal_moves()
        for move in moves:
            if self._board.push(move):
                self._node.addStat("n", 1)
                node = self._node.addChild(move)
                node.addStat("n", 1)
                node.addStat("w", self._rollout())
            self._board.pop()

        for i in range(count):
            node.getParent().addStat("n", node.getStat("n"))
            node.getParent().addStat("w", node.getStat("w"))
            node = node.getParent()
            self._board.pop()

    def _getNode(self, simulation = 100):
        for i in range(simulation):
            self._buildTree()

        return self._node.getBetterChild(tau=1)

    def getPlayerMove(self):
        if self._board.is_game_over():
            print("Referee told me to play but the game is over!")
            return "PASS"

        self._node = self._getNode()
        move = self._node.getMove()
        self._board.push(move)
        self._node.delParent()

        # New here: allows to consider internal representations of moves
        print("I am playing ", self._board.move_to_str(move))
        print("My current board :")
        self._board.prettyPrint()

        # move is an internal representation. To communicate with the interface I need to change if to a string
        return Goban.Board.flat_to_name(move)

    def playOpponentMove(self, move):
        #print("Opponent played ", move, "i.e. ", move) # New here
        # the board needs an internal represetation to push the move.  Not a string
        self._board.push(Goban.Board.name_to_flat(move))

    def newGame(self, color):
        self._mycolor = color
        self._opponent = Goban.Board.flip(color)

    def endGame(self, winner):
        if self._mycolor == winner:
            print("I won!!!")
        else:
            print("I lost :(!!")
