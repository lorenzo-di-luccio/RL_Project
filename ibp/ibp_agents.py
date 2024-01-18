import gymnasium
import itertools
import numpy
import torch
import torch.nn
import torch.optim
from typing import Any, Dict, Tuple
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
    
    def save(self) -> None:
        ckpt = dict(
            manager=self.manager.state_dict(),
            controller=self.controller.state_dict(),
            imaginator=self.imaginator.state_dict(),
            memory=self.memory.state_dict(),
            manager_optimizer=self.manager_optimizer.state_dict(),
            imaginator_optimizer=self.imaginator_optimizer.state_dict(),
            controller_memory_optimizer=self.controller_memory_optimizer.state_dict()
        )
        torch.save(ckpt, "IBP.pt")
    
    def load(self) -> None:
        ckpt = torch.load("IBP.pt")
        self.manager.load_state_dict(ckpt["manager"])
        self.controller.load_state_dict(ckpt["controller"])
        self.imaginator.load_state_dict(ckpt["imaginator"])
        self.memory.load_state_dict(ckpt["memory"])
        self.manager_optimizer.load_state_dict(ckpt["manager_optimizer"])
        self.imaginator_optimizer.load_state_dict(ckpt["imaginator_optimizer"])
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
    
    def controller_step(
            self,
            t_state: torch.Tensor,
            t_history: torch.Tensor
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        actions_prob_dist = self.controller.act(t_state, t_history)
        t_action = actions_prob_dist.sample()
        t_action_log_prob = actions_prob_dist.log_prob(t_action)
        action = t_action.item()
        return action, torch.atleast_2d(t_action), t_action_log_prob
    
    def act_step(
            self,
            action: int,
            t_last_real_state: torch.Tensor,
            t_action: torch.Tensor,
            score: float
            ) -> Tuple[
                torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                numpy.ndarray, bool, float
            ]:
        next_real_state, reward, terminated, truncated, _ = self.train_env.step(action)
        done = terminated or truncated
        score += reward
        next_real_state = numpy.atleast_1d(next_real_state)
        reward = numpy.atleast_1d(reward)
        t_next_imagined_state, t_imagined_reward = self.imaginator.step(
            t_last_real_state, t_action
        )
        t_next_real_state = torch.from_numpy(next_real_state).\
            to(dtype=torch.float32).unsqueeze(0)
        t_reward = torch.from_numpy(reward).\
            to(dtype=torch.float32).unsqueeze(0)
        score += reward
        done = terminated or truncated

        return t_next_real_state, t_next_imagined_state,\
            t_reward, t_imagined_reward,\
            next_real_state, done, score
    
    def train(self, args: Dict[str, Any]) -> None:
        self.unfreeze()
        self.train_mode()

        num_episodes = args["max_num_episodes"]

        manager_log_probs = list()
        manager_states = list()
        manager_histories = list()
        manager_rewards = list()
        imaginator_real_states = list()
        imaginator_imagined_states = list()
        imaginator_real_rewards = list()
        imaginator_imagined_rewards = list()
        controller_memory_log_probs = list()
        controller_memory_states = list()
        controller_memory_histories = list()
        controller_memory_rewards = list()

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

            manager_log_probs.clear()
            manager_states.clear()
            manager_histories.clear()
            manager_rewards.clear()
            imaginator_real_states.clear()
            imaginator_imagined_states.clear()
            imaginator_real_rewards.clear()
            imaginator_imagined_rewards.clear()
            controller_memory_log_probs.clear()
            controller_memory_states.clear()
            controller_memory_histories.clear()
            controller_memory_rewards.clear()

            while not done:
                t_last_real_state = torch.from_numpy(last_real_state)\
                    .to(dtype=torch.float32).unsqueeze(0)
                t_last_imagined_state = torch.from_numpy(last_imagined_state)\
                    .to(dtype=torch.float32).unsqueeze(0)
                
                move, t_route, t_route_log_prob = self.manager_step(
                    t_last_real_state, t_history.detach()
                )

                if move == 0 or num_imagined_steps >= imagination_budget:
                    action, t_action, t_action_log_prob = self.controller_step(
                        t_last_real_state, t_history
                    )
                    t_next_real_state, t_next_imagined_state,\
                    t_reward, t_imagined_reward,\
                    next_real_state, done, score = self.act_step(
                        action, t_last_real_state, t_action, t_route, score
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

                    manager_log_probs.append(t_route_log_prob)
                    manager_states.append(t_last_real_state.squeeze(0))
                    manager_histories.append(t_history.detach().squeeze(0))
                    manager_rewards.append(t_reward.squeeze(0))
                    imaginator_real_states.append(t_next_real_state.squeeze(0))
                    imaginator_imagined_states.append(t_next_imagined_state.squeeze(0))
                    imaginator_real_rewards.append(t_reward.squeeze(0))
                    imaginator_imagined_rewards.append(t_imagined_reward.squeeze(0))
                    controller_memory_log_probs.append(t_action_log_prob)
                    controller_memory_states.append(t_last_real_state.squeeze(0))
                    controller_memory_histories.append(t_history.squeeze(0))
                    controller_memory_rewards.append(t_reward.squeeze(0))
                elif move == 1:
                    action, t_action, t_action_log_prob = self.controller_step(
                        t_last_real_state, t_history
                    )

                    t_next_imagined_state, t_imagined_reward = self.imaginator.step(t_last_real_state, t_action)
                    num_imagined_steps += 1
                    memory_data = [
                        t_route,
                        t_last_real_state, t_last_imagined_state,
                        t_action,
                        t_next_imagined_state,
                        t_imagined_reward
                    ]
                    last_imagined_state = t_next_imagined_state.detach().squeeze(0).numpy()

                    manager_log_probs.append(t_route_log_prob)
                    manager_states.append(t_last_real_state.squeeze(0))
                    manager_histories.append(t_history.detach().squeeze(0))
                    manager_rewards.append(t_imagined_reward.detach().squeeze(0))
                    controller_memory_log_probs.append(t_action_log_prob)
                    controller_memory_states.append(t_last_real_state.squeeze(0))
                    controller_memory_histories.append(t_history.squeeze(0))
                    controller_memory_rewards.append(t_imagined_reward.detach().squeeze(0))
                elif move == 2:
                    action, t_action, t_action_log_prob = self.controller_step(
                        t_last_imagined_state, t_history
                    )

                    t_next_imagined_state, t_imagined_reward = self.imaginator.step(t_last_imagined_state, t_action)
                    num_imagined_steps += 1
                    memory_data = [
                        t_route,
                        t_last_real_state, t_last_imagined_state,
                        t_action,
                        t_next_imagined_state,
                        t_imagined_reward
                    ]
                    last_imagined_state = t_next_imagined_state.detach().squeeze(0).numpy()

                    manager_log_probs.append(t_route_log_prob)
                    manager_states.append(t_last_imagined_state.detach().squeeze(0))
                    manager_histories.append(t_history.detach().squeeze(0))
                    manager_rewards.append(t_imagined_reward.detach().squeeze(0))
                    controller_memory_log_probs.append(t_action_log_prob)
                    controller_memory_states.append(t_last_imagined_state.detach().squeeze(0))
                    controller_memory_histories.append(t_history.squeeze(0))
                    controller_memory_rewards.append(t_imagined_reward.detach().squeeze(0))
                t_history = self.memory.update(*memory_data)
                num_steps += 1
            cum_score += score

            t_manager_log_probs = torch.stack(manager_log_probs, dim=0)
            t_manager_states = torch.stack(manager_states, dim=0)
            t_manager_histories = torch.stack(manager_histories, dim=0)
            t_manager_rewards = torch.stack(manager_rewards, dim=0)
            t_imaginator_real_states = torch.stack(imaginator_real_states, dim=0)
            t_imaginator_imagined_states = torch.stack(imaginator_imagined_states, dim=0)
            t_imaginator_real_rewards = torch.stack(imaginator_real_rewards, dim=0)
            t_imaginator_imagined_rewards = torch.stack(imaginator_imagined_rewards, dim=0)
            t_controller_memory_log_probs = torch.stack(controller_memory_log_probs, dim=0)
            t_controller_memory_states = torch.stack(controller_memory_states, dim=0)
            t_controller_memory_histories = torch.stack(controller_memory_histories, dim=0)
            t_controller_memory_rewards = torch.stack(controller_memory_rewards, dim=0)

            self.manager_optimizer.zero_grad()
            manager_loss = self.manager.loss_fn(
                gamma,
                t_manager_log_probs,
                t_manager_states, t_manager_histories,
                t_manager_rewards
            )
            manager_loss.backward()
            self.manager_optimizer.zero_grad()

            self.imaginator_optimizer.zero_grad()
            imaginator_loss = self.imaginator.loss_fn(
                t_imaginator_imagined_states, t_imaginator_real_states,
                t_imaginator_imagined_rewards, t_imaginator_real_rewards
            )
            imaginator_loss.backward()

            self.imaginator_optimizer.step()
            self.controller_memory_optimizer.zero_grad()
            controller_memory_loss = self.controller.loss_fn(
                gamma,
                t_controller_memory_log_probs,
                t_controller_memory_states, t_controller_memory_histories,
                t_controller_memory_rewards
            )

            print(
                f"Episode {episode:4d}/{num_episodes:4d} Steps={num_steps:4d} Real Steps={num_real_steps:4d} Imagined Steps={num_steps - num_real_steps:4d} "
                f"M-Loss={manager_loss.item():4.4f} I-Loss={imaginator_loss.item():4.4f} CM-Loss={controller_memory_loss.item():4.4f} "
                f"Mean Score={cum_score / episode:4.4f} Score={score:4.4f} {'<==' if score > 0. else '   '}")
