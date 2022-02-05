import os
import gzip
import json
import urllib.request
import copy
import numpy as np
import torch
import Goban
from tqdm.auto import tqdm


class GoDataset(torch.utils.data.Dataset):

    url_data = "https://www.labri.fr/perso/lsimon/ia-inge2/samples-9x9.json.gz"
    raw_samples_file = "samples-9x9.json.gz"
    DEBUG=False

    def __init__(self, data=None, dtype=np.single):
        super(GoDataset, self).__init__()

        self.dtype = dtype

        if data:
            self.data, self.targets = data
        else:
            if not os.path.isfile(self.raw_samples_file):
                print("File", self.raw_samples_file, "not found, I am downloading it...", end="")
                urllib.request.urlretrieve(self.url_data, self.raw_samples_file)
                print(" Done")

            if not self.DEBUG:
                self.data, self.targets = self._load_data()

    def split(self, test_rate=0.05):
        """Split data into two new datasets"""
        test_index = int(self.__len__() * (1 - test_rate))
        return GoDataset(data=(self.data[:test_index], self.targets[:test_index])), \
               GoDataset(data=(self.data[test_index:], self.targets[test_index:]))

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)

    def _load_data(self, lim=5):
        data = []
        targets = []
        with gzip.open(self.raw_samples_file) as fz:
            raw_data = json.loads(fz.read().decode("utf-8"))

            if self.DEBUG:
                raw_data=raw_data[:lim]

            board = Goban.Board()
            for raw in tqdm(raw_data, unit=" raw", desc='Processing dataset'):
                ans =  self._raw_to_data(raw)
                if ans:
                    input, target = ans
                    data.append(input)
                    targets.append(target)

        if self.DEBUG:
            self.debug_data = data
            self.debug_targets = targets

        data = torch.from_numpy(np.array(data))
        targets = torch.from_numpy(np.array(targets))
        return data, targets

    def _raw_to_data(self, raw):
        board = Goban.Board()
        v = (raw["black_wins"] - raw["white_wins"]) / raw["rollouts"]

        # play the game until th last 8 moves (7 for the history and 1 for the policy)
        for name_move in raw["list_of_moves"][:-8]:
            move = board.name_to_flat(name_move)
            # play a move and handle errors
            try:
                board.push(move)
            except:
                return

        p = raw["depth"]%2
        player = p + 1
        other_player = p if p else 2

        # update last feature if player is white
        if p:
            input = [np.ones((9,9), dtype=self.dtype)]
        else:
            input = [np.zeros((9,9), dtype=self.dtype)]

        if raw["depth"] < 8:
            for i in range(8 - raw["depth"]):
                input.insert(0, np.zeros((9,9), dtype=self.dtype))
                input.insert(0, np.zeros((9,9), dtype=self.dtype))

        # construct history
        for name_move in raw["list_of_moves"][-8:-1]:
            move = board.name_to_flat(name_move)
            # play a move and handle errors
            try:
                board.push(move)
            except:
                return
            player_board, other_board = self._unflatten_board(board._board, player, other_player)
            input.insert(0, other_board)
            input.insert(0, player_board)

        move = board.name_to_flat(raw["list_of_moves"][-1])
        v = -v if p else v
        return np.array(input), self._policy_value_from_move(move, v)

    def _unflatten_board(self, board, player, other_player):
        player_stones = np.zeros((9,9), dtype=self.dtype)
        other_stones = np.zeros((9,9), dtype=self.dtype)
        for i, stone in enumerate(board):
            if stone == player:
                player_stones[i//9, i%9] = 1
            elif stone == other_player:
                other_stones[i//9, i%9] = 1
        return player_stones, other_stones

    def _policy_value_from_move(self, move, value):
        policy_value = np.zeros(83, dtype=self.dtype)
        policy_value[-1] = value
        policy_value[move]=1
        return policy_value
