from poker.action import Action
from poker.game_state import GameState

import numpy as np
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.network = nn.Sequential 

def myAI(state: GameState) -> Action:
    return defaultAI(state)


def defaultAI(state: GameState) -> Action: # This ai runs if myAI raises an exception, returns an invalid action or takes too long
    return Action.check() if not state.my_player.chips_in_stack or state.my_player.chips_bet == max(player.chips_bet for player in state.players) else Action.fold()