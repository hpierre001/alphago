# -*- coding: utf-8 -*-
'''
'''

import Goban
from playerInterface import *

import matplotlib.pyplot as plt
import random as rd
import numpy as np
import torch

from mcts_tree import Node
import AlphaZeroNet as azn

C_PUCT = 1

torch.no_grad()

class myPlayer(PlayerInterface):

    def __init__(self):
        self._board = Goban.Board()
        self._mycolor = None
        self._opponent = None
        self._history = None
        self._save_history = None
        self._node = Node()
        self._net = azn.Net()
        self._net.load_state_dict(torch.load("model/model_60.pt"))
        self._net.eval()
        self.softmax = torch.nn.Softmax(dim=0)

    def getPlayerName(self):
        return "NN guided MCTS Go Player"

    #def _rollout(self):
    #    count = 0
    #    while not self._board.is_game_over():
    #        moves = self._board.weak_legal_moves()
    #        rd.shuffle(moves)
    #        i = 0
    #        while not self._board.push(moves[i]):
    #            self._board.pop()
    #            i+=1
    #        count+=1
    #
    #    ans = int(self._board.final_go_score()[0].lower() == self._mycolor)
    #
    #    for i in range(count):
    #        self._board.pop()
    #
    #    return ans

    def _buildTree(self):
        """ Build MCTS tree"""
        self._save_history = self._history.clone()

        # Come down to a leaf
        count = 0
        node = self._node
        while not node.isLeaf:
            node = node.randChild()
            self._board.push(node.getMove())
            self._update_history()
            count += 1

        # use net to predict policy and value
        p, v = self._net(self._history)
        p = p[0]
        v = v[0]

        moves = self._board.weak_legal_moves()

        # change logit to 0 for illegal move
        not_legal = [True]*82
        for move in moves:
            not_legal[move] = False
        p[not_legal] = 0

        p = self.softmax(p)

        for move in moves:
            if self._board.push(move):
                self._node.addStat("n", 1)
                node = self._node.addChild(move, p=p[move].item())
                node.addStat("n", 1)
            self._board.pop()

        node.addStat("w", v.item())

        # update every node while coming up to the current node
        for i in range(count):
            node.getParent().addStat("n", node.getStat("n"))
            node.getParent().addStat("w", node.getStat("w"))
            node = node.getParent()
            self._board.pop()

        self._history = self._save_history

    def _getNode(self, simulation = 20):
        for i in range(simulation):
            self._buildTree()

        return self._node.getBetterChild(tau=1)

    def _update_history(self):
        player_stones, other_stones = self._unflatten_board()
        self._history[2:14] = self._history[:12]
        self._history[0] = player_stones
        self._history[1] = other_stones

    def _unflatten_board(self):
        player_stones = torch.zeros((9,9), dtype=self.dtype)
        other_stones = torch.zeros((9,9), dtype=self.dtype)
        for i, stone in enumerate(self._board._board):
            if stone == self._mycolor:
                player_stones[i//9, i%9] = 1
            elif stone == self._opponent:
                other_stones[i//9, i%9] = 1
        return player_stones, other_stones

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
        self._history = torch.zeros(1, 15, 9, 9)
        if color == 2:
            self._history[:,-1,:,:]=torch.ones(9, 9)

    def endGame(self, winner):
        if self._mycolor == winner:
            print("I won!!!")
        else:
            print("I lost :(!!")
