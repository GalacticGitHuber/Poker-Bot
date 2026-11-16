from poker.action import Action
from poker.game_state import GameState
from poker.card import Card, Rank, Suit
from poker.internals.hand import evaluate, RANK_VALUES
from poker.internals.rules import Rules
from poker.internals.engine import Match

from examples.call import myAI as call_ai
from examples.default import myAI as default_ai
from examples.call_preflop import myAI as call_preflop_ai
from examples.call_when_big_blind import myAI as call_when_big_blind_ai
from examples.call_when_blind import myAI as call_when_blind_ai
from examples.raise_by_big_blind_unless_raised import myAI as raise_by_big_blind_ai
from examples.raise_by_half_of_stack import myAI as raise_by_half_stack_ai
from examples.raise_by_pot import myAI as raise_by_pot_ai
from examples.raise_by_pot_when_biggest_stack import myAI as raise_by_pot_biggest_stack_ai
from examples.go_all_in import myAI as go_all_in_ai
from examples.go_all_in_on_pair import myAI as go_all_in_on_pair_ai
from examples.go_all_in_on_ace import myAI as go_all_in_on_ace_ai

OPPONENT_POOL = [
    ("call", call_ai),
    ("default", default_ai),
    ("call_preflop", call_preflop_ai),
    ("call_when_big_blind", call_when_big_blind_ai),
    ("call_when_blind", call_when_blind_ai),
    ("raise_by_big_blind", raise_by_big_blind_ai),
    ("raise_by_half_stack", raise_by_half_stack_ai),
    ("raise_by_pot", raise_by_pot_ai),
    ("raise_by_pot_biggest_stack", raise_by_pot_biggest_stack_ai),
    ("go_all_in", go_all_in_ai),
    ("go_all_in_on_pair", go_all_in_on_pair_ai),
    ("go_all_in_on_ace", go_all_in_on_ace_ai),
]

from __future__ import annotations

import argparse
import os
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

MODEL_PATH = "poker_dqn.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STATE_SIZE = 191
ACTION_SIZE = 6

FOLD = 0
CALL_CHECK = 1
MIN_RAISE = 2
DOUBLE_RAISE = 3
POT_RAISE = 4
ALL_IN = 5

GAMMA = 0.99
LEARNING_RATE = 0.0003
BATCH_SIZE = 128
REPLAY_CAPACITY = 50_000
TARGET_UPDATE_EVERY = 500

EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995

EQUITY_SAMPLES = 12


RANKS = list(Rank)
SUITS = list(Suit)

def get_rank_value(card: Card) -> int:
    return RANK_VALUES[card.rank]

def get_all_cards() -> list[Card]:
    return [Card(rank,suit) for rank in RANKS for suit in SUITS]

def get_me(state):
    return state.my_player

def get_active_opponents(state):
    me = get_me(state)
    return [
        player 
        for player in state.players
        if player is not me and not player.has_folded
    ]

def get_to_call(state) -> int:
    me = get_me(state)

    current_bet = max((player.chips_bet for player in state.player), default = 0)
    return max(0, current_bet - me.chips_bet)

def get_raise_capacity(state) -> int:
    me = get_me(state)
    to_call = get_to_call(state)
    return max(0, me.chips_in_stack - to_call)

def estimate_equity(state, samples = EQUITY_SAMPLES) -> float:
    me = get_me(state)
    opponents = get_active_opponents(state)

    if not opponents:
        return 1.0

    known = list(me.hole_cards) + list(state.community_cards)
    remaining = [card for card in get_all_cards() if card not in known]

    missing_board = 5 - len(state.community_cards)
    cards_needed = missing_board + 2 * len(opponents)

    if cards_needed > len(remaining):
        return 0.5

    wins = 0.0

    for _ in range(samples):
        dealt = random.sample(remaining, cards_needed)

        simulated_board = list(state.community_cards)
        simulated_board.extend(dealt[:missing_board])

        offset = missing_board

        my_score = evaluate(list(me.hole_cards) + simulated_board)

        opponent_scores = []

        for _opponent in opponents:
            opponent_cards = dealt[offset:offset + 2]
            offset += 2

            score = evaluate(opponent_cards + simulated_board)
            opponent_scores.append(score)

        best_score = max([my_score] + opponent_scores)

        if my_score == best_score:
            winners = sum(score == best_score for score in [my_score] + opponent_scores)
            wins += 1.0 / winners

    return wins/samples

def get_draw_features(state) -> list[float]:
    cards = list(state.my_player.hole_cards) + list(state.community_cards)

    ranks = sorted({get_rank_value(card) for card in cards})
    suits = [card.suit for card in cards]

    suit_counts = [suits.count(suit) for suit in SUITS]
    max_suit_count = max(suit_counts, default = 0)

    flush_draw = 1.0 if max_suit_count == 4 else 0.0
    made_flush = 1.0 if max_suit_count >= 5 else 0.0

    straight_draw = 0.0
    max_straight = 0.0

    straight_ranks = set(ranks)

    if 14 in straight_ranks:
        straight_ranks.add(1)

    for start in range(1,11):
        window = set(range(start, start + 5))
        count = len(window.intersection(straight_ranks))

        if count == 5:
            made_straight = 1.0
        elif count == 4:
            straight_draw = 1.0

    high_card = max(ranks, default = 2) / 14.0
    paired = 1.0 if len(ranks) < len(cards) else 0.0

    distinct_ranks = len(ranks) / 7.0

    return [flush_draw, made_flush, straight_draw, made_straight, high_card, paired, distinct_ranks]

def encode_card(card: Card | None) -> list[float]:
    result = [0.0] * 17

    if card is None: 
        return result

    rank_index = RANKS.index(card.rank)
    suit_index = SUITS.index(card.suit)

    result[rank_index] = 1.0
    result[13 + suit_index] = 1.0

    return result

def legal_action_mask(state) -> np.ndarray:
    mask = np.ones(ACTION_SIZE, dtype = np.float32)

    me = get_me(state)
    to_call = get_to_call(state)
    capacity = get_raise_capacity(state)

    mask[CALL_CHECK] = 1.0
    mask[FOLD] = 1.0

    can_be_called = any( opponent.chips_in_stack > 0  for opponent in get_active_opponents(state))

    if capacity <= 0 or not can_be_called:
        mask[MIN_RAISE:] = 0.0

    return mask

def encode_state(state, equity_samples = EQUITY_SAMPLES) -> np.ndarray:
    features = []
    me = get_me(state)

    hole_cards = sorted(list(me.hole_cards), key = lambda card: (get_rank_value(card), SUITS.index(card.suit)))
    board_cards = sorted(list(state.community_cards), key = lambda card: (get_rank_value(card), SUITS.index(card.suit)))

    cards = hole_cards[:2] + board_cards[:5]
    cards += [None] * (7 - len(cards))

    for card in cards[:7]:
        features.extend(encode_card(card))

    players = list(state.players)

    players.sort(key = lambda player: 0 if player is me else 1)
    for index in range(8):
        if index < len(players):
            player = players[index]
            features.extend([min(player.chips_in_stack /200.0, 2.0), min(player.chips_bet / max(state.big_blind, 1), 10.0), float(player.has_folded), float(player.blind is not None and player.blind.name == "SMALL"), float(player.blind is not None and player.blind.name == "BIG"), float(player is me)])
        else:
            features.extend([0.0]*6)

    to_call = get_to_call(state)
    equity = estimate_equity(state, samples = equity_samples)

    features.extend([min(state.chips_in_pot/ 1000.0, 2.0), min(me.chips_in_stack / 200.0, 2.0), min(to_call / max(state.big_blind, 1), 10.0), min(state.min_raise / max(state.big_blind, 1), 10.0), min(state.big_blind / 100.0, 2.0), min(state.ante / 100.0, 2.0), len(state.community_cards) / 5.0, equity, ])

    if len(me.hole_cards) + len(state.community_cards) >=5:
        score = evaluate(list(me.hole_cards) + list(state.community_cards))
        category = score[0]
    else:
        category = 0

    for index in range(9):
        features.append(1.0 if index == category else 0.0)

    features.extend(get_draw_features(state))
    result = np.asarray(features, dtype = np.float32)

    if len(result) != STATE_SIZE:
        raise ValueError( f"Incorrect state size: expected {STATE_SIZE}, got {len(result)}")
    
    return result

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(STATE_SIZE, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, ACTION_SIZE)
        )

    def forward(self, state):
        return self.network(state)

@dataclass
class Transition:
    state:np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    next_mask: np.ndarray
    done: bool

class ReplayBuffer:
    def __init__(self, capacity = REPLAY_CAPACITY):
        self.memory = deque(maxlen = capacity)

    def add(self, transition: Transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)

        states = np.asarray([item.state for item in batch], dtupe = np.float32)
        actions = np.asarray([item.action for item in batch], dtype = np.int64)
        rewards = np.asarray([item.action for item in batch], dtype = np.float32)
        next_states = np.asarray([item.next_state for item in batch], dtype = np.float32)
        next_masks = np.asarray([item.next_mask for item in batch], dtype = np.float32)
        dones = np.asarray([item.done for item in batch], dtype = np.float32)

        return( torch.tensor(states, device = DEVICE), torch.tensor(actions, device = DEVICE), torch.tensor(rewards, device = DEVICE), torch.tensor(next_states, device = DEVICE), torch.tensor(next_masks, device = DEVICE), torch.tensor(dones, device = DEVICE), )

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self):
        self.online = DQN().to(DEVICE)
        self.target = DQN().to(DEVICE)

        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=LEARNING_RATE)

        self.memory = ReplayBuffer()
        self.epsilon = EPSILON_START
        self.steps = 0

    def choose_action(self, state, mask, training = True):
        legal_actions = np.flatnonzero(mask)
        if len(legal_actions) == 0:
            return CALL_CHECK

        if training and random.random() < self.epsilon:
            return int(random.choice(legal_actions.tolist()))

        state_tensor = torch.tensor(state, dtype = torch.float32, device = DEVICE).unsqueeze(0)

        with torch.no_grad():
            q_values = self.online(state_tensor).squeeze(0)

        mask_tensor = torch.tensor(mask, device = DEVICE)
        q_values = q_values.masked_fill(mask_tensor == 0, -1e9)

        return int(torch.argmax(q_values).item())

    def remember(self, transition):
        self.memory.add(transition)

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return None

        (states, actions, rewards, next_states, next_masks, dones,) = self.memory.sample(BATCH_SIZE)

        current_q = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            online_next = self.online(next_states)
            online_next = online_next.masked_fill(next_masks == 0, -1e9)

            next_actions = online_next.argmax(dim=1)
            target_next = self.target(next_states)
            next_q = target_next.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            targets = rewards + GAMMA * next_q * (1.0 - dones)

        loss = nn.functional.smooth_11_loss(current_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 5.0)

        self.optimizer.step()
        self.steps += 1
        if self.steps % TARGET_UPDATE_EVERY == 0:
            self.target.load_state_dict(self.online.state_dict())

        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
        return float(loss.item())

    def save(self, path = MODEL_PATH):
        torch.save({"model": self.online.state_dict(), "target": self.target.state_dict(), "optimizer": self.optimizer.state_dict(), "epsilon": self.epsilon, "steps": self.steps}, path)

    def load(self, path = MODEL_PATH):
        checkpoint = torch.load(path, map_location = DEVICE, weights_only = False)

        self.online.load_state_dict(checkpoint["model"])
        self.target.load_state_dict(checkpoint.get("target", checkpoint["model"]))
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.epsilon = checkpoint.get("epsilon", EPSILON_MIN)
        self.steps = checkpoint.get("steps", 0)

def action_to_poker_action(action_index, state) -> Action:
    me = get_me(state)
    to_call = get_to_call(state)
    capacity = get_raise_capacity(state)

    if action_index == FOLD:
        return Action.fold()

    if action_index == CALL_CHECK:
        if to_call == 0:
            return Action.check()

        return Action.call()

    if capacity <= 9 or not get_active_opponents(state):
        return Action.call() if to_call > 0 else Action.check()

    minimum = max(1, state.min_raise)
    pot = max(0, state.chips_in_pot)

    if action_index == MIN_RAISE:
        amount = minimum
    elif action_index == DOUBLE_RAISE:
        amount = minimum * 2
    elif action_index == POT_RAISE:
        amount = max(minimum, pot + to_call)
    elif action_index == ALL_IN:
        amount = capacity
    else :
        return Action.call() if to_call > 0 else Action.check()

    amount = min(amount, capacity)

    if capacity < minimum:
        amount = capacity
    else:
        amount = max(amount, minimum)

    return Action.raise_by(int(amount))

class TrainingPolicy:
    def __init__(self):
        self.agent = self.agent
        self.pending = None
        self.starting_stack = None
        self.losses = []

    def start_hand(self, stack):
        self.pending = None
        self.starting_stack = stack
        self.losses = []

    def __call__(self, state) -> Action:
        encoded = encode_state(state)
        mask = legal_action_mask(state)

        if self.pending is not None:
            previous_state, previous_action = self.pending

            transition = Transition(state = previous_state, action = previous_action, reward = 0.0, next_state = encoded, next_mask = mask, done = False)
            self.agent.remember(transition)
        action_index = self.agent.choose_action(encoded, mask, training = True)
        self.pending = (encoded, action_index)
        return action_to_poker_action(action_index, state)

    def finish_hand(self, final_stack):
        if self.pending is None:
            return

        reward = (final_stack - self.starting_stack / 200.0)

        terminal_state = np.zeros(STATE_SIZE, dtype = np.float32)
        terminal_mask = np.zeros(ACTION_SIZE, dtype = np.float32)

        state, action = self.pending

        self.agent.remember(Transition(state = state, action = action, reward = float(reward), next_state = terminal_state, next_mask = terminal_mask, done = True ))
        self.pending = None


def train(episodes=500, hands_per_episode=100, resume=False, save_every=10,):
    agent = DQNAgent()

    if resume and os.path.exists(MODEL_PATH):
        print(f"Loading existing model: {MODEL_PATH}")
        agent.load(MODEL_PATH)

    policy = TrainingPolicy(agent)

    print(f"Training on: {DEVICE}")
    print(f"Episodes: {episodes}")
    print(f"Hands per episode: {hands_per_episode}")
    print(f"Opponent pool: {len(OPPONENT_POOL)} AIs")
    print()

    for episode in range(1, episodes + 1):

        rules = Rules()

        rules.timeout = 0

        rules.hands = 0

        opponents = random.sample(OPPONENT_POOL, 4)

        ais = [("DQN", policy), *opponents, ]

        print(f"Episode {episode:4d}/{episodes} | " f"Opponents: " f"{', '.join(name for name, _ in opponents)}")

        match = Match(ais=ais, rules=rules, seed=random.randint(0, 2**31 - 1),)

        episode_losses = []
        hands_played = 0

        for hand_number in range(1, hands_per_episode + 1):

            if match.is_over():
                break

            # DQN is currently the first player in the match.
            starting_stack = match.seats[0].stack

            policy.start_hand(starting_stack)

            match.play_hand(hand_number)

            final_stack = match.seats[0].stack

            policy.finish_hand(final_stack)

            hands_played += 1

            # Perform learning updates after the hand.
            for _ in range(2):

                loss = agent.learn()

                if loss is not None:
                    episode_losses.append(loss)

        final_stack = match.seats[0].stack

        average_loss = (float(np.mean(episode_losses)) if episode_losses else 0.0)

        print(f"  Finished | " f"Hands: {hands_played:3d} | " f"Stack: {final_stack:4d} | " f"Average loss: {average_loss:.5f} | " f"Epsilon: {agent.epsilon:.3f} | " f"Replay: {len(agent.memory)}")

        if episode % save_every == 0:
            agent.save()
            print(f"  Saved model to {MODEL_PATH}")

        print()
    agent.save()

    print(f"Training complete. Model saved to {MODEL_PATH}")

_competition_agent = None

def load_competition_agent():
    global _competition_agent
    if _competition_agent is not None:
        return _competition_agent

    agent = DQNAgent()
    if os.path.exists(MODEL_PATH):
        try:
            agent.load(MODEL_PATH)
            print("Loaded trained poker DQN Model")
        except Exception as error:
            print(f"Could not load model: {error}")
    else:
        print("No trained model found. using an untrained DQN.")

    _competition_agent = agent
    return _competition_agent

def myAI(state) -> Action:
    try:
        agent = load_competition_agent()
        encoded = encode_state(state)
        mask = legal_action_mask(state)

        action_index = agent.choose_action(encoded, mask, training = False)

        return action_to_poker_action(action_index, state)
    except Exception as error:
        print(f"myAI error: {type(error).__name__}: {error}")
        if get_to_call(state) == 0:
            return Action.check()
        return Action.fold()
    