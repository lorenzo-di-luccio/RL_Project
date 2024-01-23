import gymnasium
import numpy
import numpy.random
import random
import torch

import ibp

if __name__ == "__main__":
    RNG_SEED = 0xdeadbeef
    random.seed(RNG_SEED)
    numpy.random.seed(RNG_SEED)
    torch.manual_seed(RNG_SEED)
    agent = ibp.IBPAgent_CartPole()
    train_args = dict(
        max_num_episodes=1_000,
        imagination_budget=0,
        gamma=0.99,
        state_continuous=True,
        log_file="tmp/CartPole_imag0_ep00000_01000_log_train.csv"
    )
    #agent.load("tmp/CartPole_imag4_IBP.pt")
    agent.set_lr(agent.manager_optimizer, 1.e-3)
    agent.set_lr(agent.imaginator_optimizer, 1.e-3)
    agent.set_lr(agent.controller_memory_optimizer, 1.e-3)
    agent.train(train_args)
    agent.save("tmp/CartPole_imag4_IBP.pt")
