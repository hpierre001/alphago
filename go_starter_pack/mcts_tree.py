import random as rd
import numpy as np

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
