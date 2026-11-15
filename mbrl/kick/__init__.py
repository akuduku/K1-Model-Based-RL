r"""A model-based walk-and-kick: plan the kick with the pH model x-dot, then learn a policy that tracks the plan.

REFERENCE ONLY, not the thesis' model-based arm (docs/EXPERIMENTS.md): the plan acts through Eric Plass's walk policy
(policies/walk_teacher.pt), its searches can start from his kick motion (--init-demo), and the tracker ends with PPO.

    controller.py  the plan: a feedback controller around the deployed walk policy (walk to a stand-off pose behind
                   the ball, then a kick motion added to the walk policy's action) and its parameters
    plan.py        the planner: CEM over the plan's parameters, rolled out in the batched pH plant (Lecture 11, 1b)
    track_env.py   the tracking task: the robot's kick_walk contract (165 + clock -> 22), reward = the kick + track
    train.py       DAgger warm start from the plan (Lecture 11, option 1), then PPO (option 2) with the harness
"""
