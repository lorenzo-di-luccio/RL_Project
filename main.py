import collections
import gymnasium
import numpy
import numpy.random
import random
import torch

import ibp

class IBPAgent__(ibp.IBPAgent):
    def __init__(self) -> None:
        train_env = gymnasium.make("LunarLander-v2", render_mode="rgb_array")
        eval_env = gymnasium.make("LunarLander-v2", render_mode="human")
        state_dim = 8
        action_dim = 1
        num_actions = 4
        history_dim = 12
        hidden_dim = 120
        route_dim = 1
        num_routes = 3
        manager = ibp.Manager(
            state_dim, history_dim, hidden_dim, num_routes
        )
        controller = ibp.Controller_DAction(
            state_dim, history_dim, num_actions, hidden_dim
        )
        imaginator = ibp.Imaginator_CState(
            state_dim,
            [-1.5, -1.5, -5., -5., -3.1415927, -5., -0., -0.],
            [1.5, 1.5, 5., 5., 3.1415927, 5., 1., 1.],
            action_dim, hidden_dim
        )
        memory = ibp.Memory(
            route_dim, state_dim, action_dim, history_dim
        )

        super(IBPAgent__, self).__init__(
            train_env, eval_env,
            manager, controller, imaginator, memory
        )

if __name__ == "__main__":
    RNG_SEED = 0xdeadbeef
    random.seed(RNG_SEED)
    numpy.random.seed(RNG_SEED)
    torch.manual_seed(RNG_SEED)
    agent = IBPAgent__()
    train_args = dict(
        max_num_episodes=1_000,
        imagination_budget=0,
        gamma=0.99,
        state_continuous=True,
        log_file="log_train.csv"
    )
    eval_args = dict(
        max_num_episodes=32,
        log_file="log_eval.csv"
    )
    #agent.load()
    agent.set_lr(agent.manager_optimizer, 1.e-3)
    agent.set_lr(agent.imaginator_optimizer, 1.e-3)
    agent.set_lr(agent.controller_memory_optimizer, 1.e-3)
    #agent.train(train_args)
    agent.evaluate(eval_args)
    #agent.save()