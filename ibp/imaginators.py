import torch
import torch.nn
import torch.nn.functional
from typing import List, Tuple

class Imaginator_DState(torch.nn.Module):
    def __init__(
            self,
            state_dim: int,
            num_states: int,
            action_dim: int,
            hidden_dim: int
    ) -> None:
        super(Imaginator_DState, self).__init__()

        self.next_state_predictor = torch.nn.Sequential(
            torch.nn.Linear(state_dim + action_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_states)
        )
        self.reward_predictor = torch.nn.Sequential(
            torch.nn.Linear(state_dim + action_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )
    
    def forward(
            self,
            states: torch.Tensor,
            actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        states_actions = torch.cat([states, actions], dim=1)
        next_states_logits = self.next_state_predictor(states_actions)
        next_states = torch.argmax(next_states_logits, dim=-1)
        rewards = self.reward_predictor(states_actions)
        return next_states, next_states_logits, rewards
    
    def step(
            self,
            states: torch.Tensor,
            actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self(states, actions)

class Imaginator_CState(torch.nn.Module):
    def __init__(
            self,
            state_dim: int,
            state_min: List[float],
            state_max: List[float],
            action_dim: int,
            hidden_dim: int
    ) -> None:
        super(Imaginator_DState, self).__init__()
        
        self.state_min = torch.tensor(state_min, dtype=torch.float32)
        self.state_max = torch.tensor(state_max, dtype=torch.float32)
        self.next_state_predictor = torch.nn.Sequential(
            torch.nn.Linear(state_dim + action_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, state_dim)
        )
        self.reward_predictor = torch.nn.Sequential(
            torch.nn.Linear(state_dim + action_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )
    
    def forward(
            self,
            states: torch.Tensor,
            actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        states_actions = torch.cat([states, actions], dim=1)
        next_states_unbound = self.next_state_predictor(states_actions)
        next_states = torch.clamp(
            next_states_unbound, min=self.state_min, max=self.state_max
        )
        rewards = self.reward_predictor(states_actions)
        return next_states, rewards
    
    def step(
            self,
            states: torch.Tensor,
            actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self(states, actions)
