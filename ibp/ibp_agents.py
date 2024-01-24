import gymnasium
import itertools
import numpy
import torch
import torch.nn
import torch.optim
from typing import Any, Dict, List, Tuple
from .manager import Manager
from .controllers import Controller_DAction
from .imaginators import Imaginator_DState, Imaginator_CState
from .memory import Memory

class IBPAgent():
    def __init__(
            self,
            train_env: gymnasium.Env,
            eval_env: gymnasium.Env,
            manager: Manager,
            controller: Controller_DAction,
            imaginator: Imaginator_DState | Imaginator_CState,
            memory: Memory
    ) -> None:
        self.train_env = train_env
        self.eval_env = eval_env

        self.manager = manager
        self.controller = controller
        self.imaginator = imaginator
        self.memory = memory

        self.manager_optimizer = torch.optim.Adam(
            self.manager.parameters(), lr=1.e-3
        )
        self.imaginator_optimizer = torch.optim.Adam(
            self.imaginator.parameters(), lr=1.e-3
        )
        self.controller_memory_optimizer = torch.optim.Adam(
            itertools.chain(self.controller.parameters(), self.memory.parameters()),
            lr=1.e-3
        )
    
    def save(
            self,
            filename: str
    ) -> None:
        ckpt = dict(
            manager=self.manager.state_dict(),
            controller=self.controller.state_dict(),
            imaginator=self.imaginator.state_dict(),
            memory=self.memory.state_dict(),
            manager_optimizer=self.manager_optimizer.state_dict(),
            imaginator_optimizer=self.imaginator_optimizer.state_dict(),
            controller_memory_optimizer=self.controller_memory_optimizer.state_dict()
        )
        torch.save(ckpt, filename)
    
    def load(
            self,
            filename: str
    ) -> None:
        ckpt = torch.load(filename)
        self.manager.load_state_dict(ckpt["manager"])
        self.controller.load_state_dict(ckpt["controller"])
        self.imaginator.load_state_dict(ckpt["imaginator"])
        self.memory.load_state_dict(ckpt["memory"])
        self.manager_optimizer.load_state_dict(ckpt["manager_optimizer"])
        self.imaginator_optimizer.load_state_dict(ckpt["imaginator_optimizer"])
        self.controller_memory_optimizer.load_state_dict(ckpt["controller_memory_optimizer"])

    def load_imaginator(
            self,
            filename: str
    ) -> None:
        ckpt = torch.load(filename)
        self.imaginator.load_state_dict(ckpt["imaginator"])
        self.imaginator_optimizer.load_state_dict(ckpt["imaginator_optimizer"])
    
    def load_controller_memory(
            self,
            filename: str
    ) -> None:
        ckpt = torch.load(filename)
        self.controller.load_state_dict(ckpt["controller"])
        self.memory.load_state_dict(ckpt["memory"])
        self.controller_memory_optimizer.load_state_dict(ckpt["controller_memory_optimizer"])
    
    def train_mode(self) -> None:
        self.manager.train()
        self.controller.train()
        self.imaginator.train()
        self.memory.train()

    def eval_mode(self) -> None:
        self.manager.eval()
        self.controller.eval()
        self.imaginator.eval()
        self.memory.eval()

    def freeze(self) -> None:
        self.manager.requires_grad_(requires_grad=False)
        self.controller.requires_grad_(requires_grad=False)
        self.imaginator.requires_grad_(requires_grad=False)
        self.memory.requires_grad_(requires_grad=False)

    def unfreeze(self) -> None:
        self.manager.requires_grad_(requires_grad=True)
        self.controller.requires_grad_(requires_grad=True)
        self.imaginator.requires_grad_(requires_grad=True)
        self.memory.requires_grad_(requires_grad=True)
    
    def set_lr(
            self,
            optimizer: torch.optim.Optimizer,
            lr: float
    ) -> None:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    
    def manager_step(
            self,
            t_last_real_state: torch.Tensor,
            t_history: torch.Tensor
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        routes_prob_dist = self.manager.decide(t_last_real_state, t_history)
        t_route = routes_prob_dist.sample()
        t_route_log_prob = routes_prob_dist.log_prob(t_route)
        move = t_route.item()
        return move, torch.atleast_2d(t_route), t_route_log_prob
    
    def train_controller_step(
            self,
            t_state: torch.Tensor,
            t_history: torch.Tensor
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        actions_prob_dist = self.controller.act(t_state, t_history)
        t_action = actions_prob_dist.sample()
        t_action_log_prob = actions_prob_dist.log_prob(t_action)
        action = t_action.item()
        return action, torch.atleast_2d(t_action), t_action_log_prob
    
    def eval_controller_step(
            self,
            t_state: torch.Tensor,
            t_history: torch.Tensor
    ) -> Tuple[int, torch.Tensor]:
        actions_prob_dist = self.controller.act(t_state, t_history)
        t_action = actions_prob_dist.mode
        action = t_action.item()
        return action, torch.atleast_2d(t_action)
    
    def train_act_step(
            self,
            action: int,
            t_last_real_state: torch.Tensor,
            t_action: torch.Tensor,
            score: float
    ) -> Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
            numpy.ndarray, bool, float
        ]:
        next_real_state, reward, terminated, truncated, _ = self.train_env.step(action)
        done = terminated or truncated
        score += float(reward)
        next_real_state: numpy.ndarray = numpy.atleast_1d(next_real_state)
        reward = numpy.atleast_1d(reward)
        t_next_imagined_state,\
        t_next_imagined_state_logits,\
        t_imagined_reward = self.imaginator.step(t_last_real_state, t_action)
        t_next_real_state = torch.from_numpy(next_real_state).\
            to(dtype=torch.float32).unsqueeze(0)
        t_reward = torch.from_numpy(reward).\
            to(dtype=torch.float32).unsqueeze(0)

        return t_next_real_state,\
            t_next_imagined_state, t_next_imagined_state_logits,\
            t_reward, t_imagined_reward,\
            next_real_state, done, score
    
    def eval_act_step(
            self,
            action: int,
            score: float
    ) -> Tuple[torch.Tensor, torch.Tensor, numpy.ndarray, bool, float]:
        next_real_state, reward, terminated, truncated, _ = self.eval_env.step(action)
        done = terminated or truncated
        score += float(reward)
        next_real_state: numpy.ndarray = numpy.atleast_1d(next_real_state)
        reward = numpy.atleast_1d(reward)
        
        t_next_real_state = torch.from_numpy(next_real_state).\
            to(dtype=torch.float32).unsqueeze(0)
        t_reward = torch.from_numpy(reward).\
            to(dtype=torch.float32).unsqueeze(0)

        return t_next_real_state, t_reward,\
            next_real_state, done, score
    
    def imagination_step(
            self,
            t_action: torch.Tensor,
            t_last_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t_next_imagined_state,\
        _,\
        t_imagined_reward = self.imaginator.step(t_last_state, t_action)

        return t_next_imagined_state, t_imagined_reward
    
    def init_train_data(
            self,
            state_continuous: bool
    ) -> Dict[str, List[torch.Tensor]]:
        suffix = "" if state_continuous else "_logits"
        train_data = {
            "manager_log_probs": list(),
            "manager_states": list(),
            "manager_histories": list(),
            "manager_rewards": list(),
            "imaginator_real_states": list(),
            f"imaginator_imagined_states{suffix}": list(),
            "imaginator_real_rewards": list(),
            "imaginator_imagined_rewards": list(),
            "controller_memory_log_probs": list(),
            "controller_memory_states": list(),
            "controller_memory_histories": list(),
            "controller_memory_rewards": list()
        }
        return train_data
    
    def clear_train_data(
            self,
            train_data: Dict[str, List[torch.Tensor]],
            state_continuous: bool
    ) -> None:
        suffix = "" if state_continuous else "_logits"
        train_data["manager_log_probs"].clear()
        train_data["manager_states"].clear()
        train_data["manager_histories"].clear()
        train_data["manager_rewards"].clear()
        train_data["imaginator_real_states"].clear()
        train_data[f"imaginator_imagined_states{suffix}"].clear()
        train_data["imaginator_real_rewards"].clear()
        train_data["imaginator_imagined_rewards"].clear()
        train_data["controller_memory_log_probs"].clear()
        train_data["controller_memory_states"].clear()
        train_data["controller_memory_histories"].clear()
        train_data["controller_memory_rewards"].clear()
    
    def update_train_data_on_act(
            self,
            train_data: Dict[str, List[torch.Tensor]],
            state_continuous: bool,
            new_train_data: Dict[str, torch.Tensor]
    ) -> None:
        suffix = "" if state_continuous else "_logits"
        train_data["manager_log_probs"].append(
            new_train_data["t_route_log_prob"]
        )
        train_data["manager_states"].append(
            new_train_data["t_last_real_state"]
        )
        train_data["manager_histories"].append(
            new_train_data["t_history_detached"]
        )
        train_data["manager_rewards"].append(
            new_train_data["t_reward"]
        )
        train_data["imaginator_real_states"].append(
            new_train_data["t_next_real_state"]
        )
        train_data[f"imaginator_imagined_states{suffix}"].append(
            new_train_data[f"t_next_imagined_state{suffix}"]
        )
        train_data["imaginator_real_rewards"].append(
            new_train_data["t_reward"]
        )
        train_data["imaginator_imagined_rewards"].append(
            new_train_data["t_imagined_reward"]
        )
        train_data["controller_memory_log_probs"].append(
            new_train_data["t_action_log_prob"]
        )
        train_data["controller_memory_states"].append(
            new_train_data["t_last_real_state"]
        )
        train_data["controller_memory_histories"].append(
            new_train_data["t_history"]
        )
        train_data["controller_memory_rewards"].append(
            new_train_data["t_reward"]
        )
    
    def update_train_data_on_imagine1(
            self,
            train_data: Dict[str, List[torch.Tensor]],
            new_train_data: Dict[str, torch.Tensor]
    ) -> None:
        train_data["manager_log_probs"].append(
            new_train_data["t_route_log_prob"]
        )
        train_data["manager_states"].append(
            new_train_data["t_last_real_state"]
        )
        train_data["manager_histories"].append(
            new_train_data["t_history_detached"]
        )
        train_data["manager_rewards"].append(
            new_train_data["t_imagined_reward"]
        )
        train_data["controller_memory_log_probs"].append(
            new_train_data["t_action_log_prob"]
        )
        train_data["controller_memory_states"].append(
            new_train_data["t_last_real_state"]
        )
        train_data["controller_memory_histories"].append(
            new_train_data["t_history"]
        )
        train_data["controller_memory_rewards"].append(
            new_train_data["t_imagined_reward"]
        )
    
    def update_train_data_on_imagine2(
            self,
            train_data: Dict[str, List[torch.Tensor]],
            new_train_data: Dict[str, torch.Tensor]
    ) -> None:
        train_data["manager_log_probs"].append(
            new_train_data["t_route_log_prob"]
        )
        train_data["manager_states"].append(
            new_train_data["t_last_imagined_state"]
        )
        train_data["manager_histories"].append(
            new_train_data["t_history_detached"]
        )
        train_data["manager_rewards"].append(
            new_train_data["t_imagined_reward"]
        )
        train_data["controller_memory_log_probs"].append(
            new_train_data["t_action_log_prob"]
        )
        train_data["controller_memory_states"].append(
            new_train_data["t_last_imagined_state"]
        )
        train_data["controller_memory_histories"].append(
            new_train_data["t_history"]
        )
        train_data["controller_memory_rewards"].append(
            new_train_data["t_imagined_reward"]
        )
    
    def batch_train_data(
            self,
            train_data: Dict[str, List[torch.Tensor]],
            state_continuous: bool
    ) -> Dict[str, torch.Tensor]:
        suffix = "" if state_continuous else "_logits"
        batched_train_data = {
            "t_manager_log_probs": torch.stack(
                train_data["manager_log_probs"], dim=0
            ),
            "t_manager_states": torch.stack(
                train_data["manager_states"], dim=0
            ),
            "t_manager_histories": torch.stack(
                train_data["manager_histories"], dim=0
            ),
            "t_manager_rewards": torch.stack(
                train_data["manager_rewards"], dim=0
            ),
            "t_imaginator_real_states": torch.stack(
                train_data["imaginator_real_states"], dim=0
            ),
            f"t_imaginator_imagined_states{suffix}": torch.stack(
                train_data[f"imaginator_imagined_states{suffix}"], dim=0
            ),
            "t_imaginator_real_rewards": torch.stack(
                train_data["imaginator_real_rewards"], dim=0
            ),
            "t_imaginator_imagined_rewards": torch.stack(
            train_data["imaginator_imagined_rewards"], dim=0
            ),
            "t_controller_memory_log_probs": torch.stack(
                train_data["controller_memory_log_probs"], dim=0
            ),
            "t_controller_memory_states": torch.stack(
                train_data["controller_memory_states"], dim=0
            ),
            "t_controller_memory_histories": torch.stack(
                train_data["controller_memory_histories"], dim=0
            ),
            "t_controller_memory_rewards": torch.stack(
                train_data["controller_memory_rewards"], dim=0
            )
        }
        if not state_continuous:
            batched_train_data["t_imaginator_real_states"] = \
                batched_train_data["t_imaginator_real_states"].view(-1)\
                .to(dtype=torch.int64)
        return batched_train_data
    
    def train(self, args: Dict[str, Any]) -> None:
        self.unfreeze()
        self.train_mode()

        num_episodes = args["max_num_episodes"]
        state_continuous = args["state_continuous"]
        log_file = open(args["log_file"], mode="w", encoding="utf-8")
        log_file.write("Episode,Score,AvgScore\n")
        suffix = "" if state_continuous else "_logits"

        train_data = self.init_train_data(state_continuous)

        cum_score = 0.

        for episode in range(1, num_episodes + 1):
            state, _ = self.train_env.reset()
            state = numpy.atleast_1d(state)
            done = False
            score = 0.
            t_history = self.memory.reset()
            num_steps = 0
            num_real_steps = 0
            num_imagined_steps = 0
            imagination_budget = args["imagination_budget"]
            last_real_state = state
            last_imagined_state = state
            gamma = args["gamma"]

            self.clear_train_data(train_data, state_continuous)

            while not done:
                t_last_real_state = torch.from_numpy(last_real_state)\
                    .to(dtype=torch.float32).unsqueeze(0)
                t_last_imagined_state = torch.from_numpy(last_imagined_state)\
                    .to(dtype=torch.float32).unsqueeze(0)
                
                move, t_route, t_route_log_prob = self.manager_step(
                    t_last_real_state, t_history.detach()
                )

                if move == 0 or num_imagined_steps >= imagination_budget:
                    action, t_action, t_action_log_prob = self.train_controller_step(
                        t_last_real_state, t_history
                    )

                    t_next_real_state,\
                    t_next_imagined_state,t_next_imagined_state_logits,\
                    t_reward, t_imagined_reward,\
                    next_real_state, done, score = self.train_act_step(
                        action, t_last_real_state, t_action, score
                    )

                    num_real_steps += 1
                    num_imagined_steps = 0
                    memory_data = [
                        t_route,
                        t_last_real_state, t_last_imagined_state,
                        t_action,
                        t_next_real_state,
                        t_reward
                    ]
                    last_real_state = next_real_state
                    last_imagined_state = next_real_state

                    new_train_data = {
                        "t_route_log_prob": t_route_log_prob,
                        "t_last_real_state": t_last_real_state.squeeze(0),
                        "t_history_detached": t_history.detach().squeeze(0),
                        "t_reward": t_reward.squeeze(0),
                        "t_next_real_state": t_next_real_state.squeeze(0),
                        f"t_next_imagined_state{suffix}":\
                            t_next_imagined_state.squeeze(0) if state_continuous else\
                            t_next_imagined_state_logits.squeeze(0),
                        "t_imagined_reward": t_imagined_reward.squeeze(0),
                        "t_action_log_prob": t_action_log_prob,
                        "t_history": t_history.squeeze(0)
                    }
                    self.update_train_data_on_act(
                        train_data, state_continuous, new_train_data
                    )
                elif move == 1:
                    action, t_action, t_action_log_prob = self.train_controller_step(
                        t_last_real_state, t_history
                    )

                    with torch.no_grad():
                        t_next_imagined_state, t_imagined_reward = self.imagination_step(
                            t_action, t_last_real_state
                        )
                    
                    num_imagined_steps += 1
                    memory_data = [
                        t_route,
                        t_last_real_state, t_last_imagined_state,
                        t_action,
                        t_next_imagined_state,
                        t_imagined_reward
                    ]
                    last_imagined_state = t_next_imagined_state.squeeze(0).numpy()

                    new_train_data = {
                        "t_route_log_prob": t_route_log_prob,
                        "t_last_real_state": t_last_real_state.squeeze(0),
                        "t_history_detached": t_history.detach().squeeze(0),
                        "t_imagined_reward": t_imagined_reward.squeeze(0),
                        "t_action_log_prob": t_action_log_prob,
                        "t_history": t_history.squeeze(0)
                    }
                    self.update_train_data_on_imagine1(train_data, new_train_data)
                elif move == 2:
                    action, t_action, t_action_log_prob = self.train_controller_step(
                        t_last_imagined_state, t_history
                    )

                    with torch.no_grad():
                        t_next_imagined_state, t_imagined_reward = self.imagination_step(
                            t_action, t_last_imagined_state
                        )
                    
                    num_imagined_steps += 1
                    memory_data = [
                        t_route,
                        t_last_real_state, t_last_imagined_state,
                        t_action,
                        t_next_imagined_state,
                        t_imagined_reward
                    ]
                    last_imagined_state = t_next_imagined_state.squeeze(0).numpy()

                    new_train_data = {
                        "t_route_log_prob": t_route_log_prob,
                        "t_last_imagined_state": t_last_imagined_state.squeeze(0),
                        "t_history_detached": t_history.detach().squeeze(0),
                        "t_imagined_reward": t_imagined_reward.squeeze(0),
                        "t_action_log_prob": t_action_log_prob,
                        "t_history": t_history.squeeze(0)
                    }
                    self.update_train_data_on_imagine2(train_data, new_train_data)
                t_history = self.memory.update(*memory_data)
                num_steps += 1
            cum_score += score

            batched_train_data = self.batch_train_data(train_data, state_continuous)

            self.manager_optimizer.zero_grad()
            manager_loss = self.manager.loss_fn(
                gamma,
                batched_train_data["t_manager_log_probs"],
                batched_train_data["t_manager_states"],
                batched_train_data["t_manager_histories"],
                batched_train_data["t_manager_rewards"]
            ) * (0. if imagination_budget == 0 else 1.)
            manager_loss.backward()
            self.manager_optimizer.step()

            self.imaginator_optimizer.zero_grad()
            imaginator_loss = self.imaginator.loss_fn(
                batched_train_data[f"t_imaginator_imagined_states{suffix}"],
                batched_train_data["t_imaginator_real_states"],
                batched_train_data["t_imaginator_imagined_rewards"],
                batched_train_data["t_imaginator_real_rewards"]
            )
            imaginator_loss.backward()
            self.imaginator_optimizer.step()

            self.controller_memory_optimizer.zero_grad()
            controller_memory_loss = self.controller.loss_fn(
                gamma,
                batched_train_data["t_controller_memory_log_probs"],
                batched_train_data["t_controller_memory_states"],
                batched_train_data["t_controller_memory_histories"],
                batched_train_data["t_controller_memory_rewards"]
            )
            controller_memory_loss.backward()
            self.controller_memory_optimizer.step()

            print(
                f"Episode {episode:4d}/{num_episodes:4d} Steps={num_steps:4d} Real|Imagined Steps={num_real_steps:4d}|{num_steps - num_real_steps:4d} "
                f"M-Loss={manager_loss.item():4.4f} I-Loss={imaginator_loss.item():4.4f} CM-Loss={controller_memory_loss.item():4.4f} "
                f"Mean Score={cum_score / episode:4.4f} Score={score:4.4f} {'<==' if score > 0. else '   '}")
            log_file.write(f"{episode},{score:4.4f},{cum_score / episode:4.4f}\n")
        self.train_env.close()
        log_file.close()
    
    def evaluate(self, args: Dict[str, Any]) -> None:
        self.freeze()
        self.eval_mode()

        num_episodes = args["max_num_episodes"]
        log_file = open(args["log_file"], mode="w", encoding="utf-8")
        log_file.write("Episode,Score,AvgScore\n")

        cum_score = 0.

        for episode in range(1, num_episodes + 1):
            state, _ = self.eval_env.reset()
            state = numpy.atleast_1d(state)
            done = False
            score = 0.
            t_history = self.memory.reset()
            num_steps = 0
            last_real_state = state
            last_imagined_state = state

            while not done:
                t_last_real_state = torch.from_numpy(last_real_state)\
                    .to(dtype=torch.float32).unsqueeze(0)
                t_last_imagined_state = torch.from_numpy(last_imagined_state)\
                    .to(dtype=torch.float32).unsqueeze(0)

                action, t_action = self.eval_controller_step(
                    t_last_real_state, t_history
                )

                t_next_real_state, t_reward,\
                next_real_state, done, score = self.eval_act_step(
                    action, score
                )

                memory_data = [
                    torch.tensor([[0]], dtype=torch.int64),
                    t_last_real_state, t_last_imagined_state,
                    t_action,
                    t_next_real_state,
                    t_reward
                ]
                last_real_state = next_real_state
                last_imagined_state = next_real_state
                t_history = self.memory.update(*memory_data)
                num_steps += 1
            cum_score += score

            print(
                f"Episode {episode:4d}/{num_episodes:4d} Steps={num_steps:4d} "
                f"Mean Score={cum_score / episode:4.4f} Score={score:4.4f} {'<==' if score > 0. else '   '}")
            log_file.write(f"{episode},{score:4.4f},{cum_score / episode:4.4f}\n")
        self.eval_env.close()
        log_file.close()

class IBPAgent_CartPole(IBPAgent):
    def __init__(self) -> None:
        train_env = gymnasium.make("CartPole-v1", render_mode="rgb_array")
        eval_env = gymnasium.make("CartPole-v1", render_mode="human")
        state_dim = 4
        num_states = 1
        action_dim = 1
        num_actions = 2
        history_dim = 12
        hidden_dim = 64
        route_dim = 1
        num_routes = 3
        manager = Manager(
            state_dim, history_dim, hidden_dim, num_routes
        )
        controller = Controller_DAction(
            state_dim, history_dim, num_actions, hidden_dim
        )
        imaginator = Imaginator_CState(
            state_dim,
            [-4.8000002e+00, -3.4028235e+38, -4.1887903e-01, -3.4028235e+38],
            [4.8000002e+00, 3.4028235e+38, 4.1887903e-01, 3.4028235e+38],
            action_dim, hidden_dim
        )
        memory = Memory(
            route_dim, state_dim, action_dim, history_dim
        )

        super(IBPAgent_CartPole, self).__init__(
            train_env, eval_env,
            manager, controller, imaginator, memory
        )
