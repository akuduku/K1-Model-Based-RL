r"""Model-based RL with the pH plant as the model: plan a short time ahead at every step, learn a policy and a value.

The thesis' model-based arm (docs/EXPERIMENTS.md).  Known-model planning with a learned value function (Lowrey et
al. 2019, "Plan online, learn offline"), plus a policy network that learns the planner's actions, so the robot can
run the policy without a planner.  Everything starts from scratch.

    nets.py      the policy and value networks (the firmware's 512-256-128 ELU MLP, rsl-rl's checkpoint layout)
    planner.py   MPPI around the policy, rolled out in a second copy of the task env (the model)
    learner.py   the loop: plan, act in the world, learn from what was executed; logging, checkpoints, clean stop
    train.py     python -m mbrl.mpc.train stand ...
"""
