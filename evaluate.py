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
    agent = ibp.IBPAgent_LunarLander()
    eval_args = dict(
        max_num_episodes=16,
        log_file="tmp/LunarLander_imag0_log_evalx.csv"
    )
    agent.load("models/LunarLander/LunarLander_imag0_IBP.pt")
    agent.evaluate(eval_args)
