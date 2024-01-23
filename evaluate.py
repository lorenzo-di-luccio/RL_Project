import gymnasium
import numpy
import numpy.random
import random
import torch

import ibp

class IBPAgent__(ibp.IBPAgent):
    def __init__(self) -> None:
        train_env = gymnasium.make("CartPole-v1", render_mode="rgb_array")
        eval_env = gymnasium.make("CartPole-v1", render_mode="human")
        state_dim = 4
        num_states = 1
        action_dim = 1
        num_actions = 2
        history_dim = 32
        hidden_dim = 48
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
            [-4.8000002e+00, -3.4028235e+38, -4.1887903e-01, -3.4028235e+38],
            [4.8000002e+00, 3.4028235e+38, 4.1887903e-01, 3.4028235e+38],
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
    eval_args = dict(
        max_num_episodes=16,
        log_file="tmp/CartPole_imag0_log_eval.csv"
    )
    #agent.load("tmp/CartPole_imag0_IBP.pt")
    agent.evaluate(eval_args)
