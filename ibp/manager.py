import torch
import torch.distributions
import torch.nn
import torch.nn.functional

class Manager(torch.nn.Module):
    def __init__(
            self,
            state_dim: int,
            history_dim: int,
            hidden_dim: int,
            num_routes: int
    ) -> None:
        super(Manager, self).__init__()

        self.actor = torch.nn.Sequential(
            torch.nn.Linear(state_dim + history_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_routes),
            torch.nn.Softmax(dim=-1)
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
        routes_probs = self.actor(states_histories)
        routes_prob_dist = torch.distributions.Categorical(probs=routes_probs)
        return routes_prob_dist

    def decide(
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
