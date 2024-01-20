import torch
import torch.nn
import torch.nn.functional

class Memory(torch.nn.Module):
    def __init__(
            self,
            route_dim: int,
            state_dim: int,
            action_dim: int,
            history_dim: int
    ) -> None:
        super(Memory, self).__init__()

        data_dim = route_dim + state_dim + state_dim + action_dim + state_dim + 1
        self.rec = torch.nn.LSTMCell(data_dim, history_dim)
        self.history_embeddings = torch.zeros((1, history_dim))
        self.cell_state = torch.zeros((1, history_dim))

    def forward(
            self,
            routes: torch.Tensor,
            real_states: torch.Tensor,
            imagined_states: torch.Tensor,
            actions: torch.Tensor,
            new_states: torch.Tensor,
            rewards: torch.Tensor
    ) -> torch.Tensor:
        data = torch.cat(
            [routes, real_states, imagined_states, actions, new_states, rewards],
            dim=1
        )
        self.history_embeddings, self.cell_state = self.rec(
            data, (self.history_embeddings, self.cell_state)
        )
        return self.history_embeddings

    def reset(self) -> torch.Tensor:
        self.history_embeddings = torch.zeros_like(self.history_embeddings)
        self.cell_state = torch.zeros_like(self.cell_state)
        return self.history_embeddings

    def update(
            self,
            routes: torch.Tensor,
            real_states: torch.Tensor,
            imagined_states: torch.Tensor,
            actions: torch.Tensor,
            new_states: torch.Tensor,
            rewards: torch.Tensor
    ) -> torch.Tensor:
        return self(routes, real_states, imagined_states, actions, new_states, rewards)
