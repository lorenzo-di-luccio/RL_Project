import collections
import torch
import torch.distributions
import torch.nn
import torch.nn.functional

class Controller_DAction(torch.nn.Module):
    def __init__(
            self,
            state_dim: int,
            history_dim: int,
            num_actions: int,
            hidden_dim: int
    ) -> None:
        super(Controller_DAction, self).__init__()

        self.actor = torch.nn.Sequential(
            torch.nn.Linear(state_dim + history_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_actions)
        )
        self.value_fn_predictor = torch.nn.Sequential(
            torch.nn.Linear(state_dim + history_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(
            self,
            states: torch.Tensor,
            histories: torch.Tensor
    ) -> torch.distributions.Categorical:
        states_histories = torch.cat([states, histories], dim=1)
        actions_logits = self.actor(states_histories)
        actions_probs = torch.softmax(actions_logits, dim=-1)
        actions_prob_dist = torch.distributions.Categorical(probs=actions_probs)
        return actions_prob_dist

    def act(
            self,
            states: torch.Tensor,
            histories: torch.Tensor
    ) -> torch.distributions.Categorical:
        return self(states, histories)

    def get_values(
            self,
            states: torch.Tensor,
            histories: torch.Tensor
    ) -> torch.Tensor:
        states_histories = torch.cat([states, histories], dim=1)
        values = self.value_fn_predictor(states_histories)
        return values

    def loss_fn(
            self,
            gamma: float,
            log_probs: torch.Tensor,
            states: torch.Tensor,
            histories: torch.Tensor,
            rewards: torch.Tensor
    ) -> torch.Tensor:
        Gs = collections.deque(maxlen=len(rewards))
        G = torch.tensor(0., dtype=torch.float32)
        for t in reversed(range(len(rewards))):
            G = G + gamma * rewards[t]
            Gs.appendleft(G)
        returns = torch.tensor(Gs, dtype=torch.float32)
        values = self.get_values(states, histories).view(-1)
        with torch.no_grad():
            advantages = returns - values
        reinforce_loss = (-log_probs * advantages).mean()
        value_loss = torch.nn.functional.smooth_l1_loss(values, returns)
        return reinforce_loss + value_loss
