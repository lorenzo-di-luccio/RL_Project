import gymnasium
import torch
import torch.nn
import torch.optim
from typing import Any, Dict

class IBPAgent():
    def __init__(
            self,
            train_env: gymnasium.Env,
            eval_env: gymnasium.Env,
            manager_cls: torch.nn.Module,
            manager_args: Dict[str, Any],
            controller_cls: torch.nn.Module,
            controller_args: Dict[str, Any],
            imaginator_cls: torch.nn.Module,
            imaginator_args: Dict[str, Any],
            memory_cls: torch.nn.Module,
            memory_args: Dict[str, Any]
    ) -> None:
        self.train_env = train_env
        self.eval_env = eval_env
        self.manager = manager_cls(**manager_args)
        self.controller = controller_cls(**controller_args)
        self.imaginator = imaginator_cls(**imaginator_args)
        self.memory = memory_cls(**memory_args)
