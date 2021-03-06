# -*- coding: utf-8 -*-
'''
'''

import Goban
from playerInterface import *

import matplotlib.pyplot as plt
import random as rd
import numpy as np

from mcts_tree import Node

C_PUCT = 1

class myPlayer(PlayerInterface):

    def __init__(self):
        self._board = Goban.Board()
        self._mycolor = None
        self._node = Node()

    def getPlayerName(self):
        return "MCTS Go Player"

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
        """ Build MCTS tree """

        # Come down to a leaf
        count = 0
        node = self._node
        while not node.isLeaf:
            node = node.randChild()
            self._board.push(node.getMove())
            count += 1

        # roll out once every child of this leaf
        moves = self._board.weak_legal_moves()
        for move in moves:
            if self._board.push(move):
                self._node.addStat("n", 1)
                node = self._node.addChild(move)
                node.addStat("n", 1)
                node.addStat("w", self._rollout())
            self._board.pop()

        # update every node while coming up to the current node
        for i in range(count):
            node.getParent().addStat("n", node.getStat("n"))
            node.getParent().addStat("w", node.getStat("w"))
            node = node.getParent()
            self._board.pop()

    def _getNode(self, simulation = 10):
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
        #??the board needs an internal represetation to push the move.  Not a string
        self._board.push(Goban.Board.name_to_flat(move))

    def newGame(self, color):
        self._mycolor = color
        self._opponent = Goban.Board.flip(color)

    def endGame(self, winner):
        if self._mycolor == winner:
            print("I won!!!")
        else:
            print("I lost :(!!")
