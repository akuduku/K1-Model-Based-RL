r"""A model-based get-up: plan the motion with the pH model x-dot, then learn a feedback policy that tracks the plan.

    keyframes.py   the plan's parameters (mirrored keyframes + durations) and their interpolation
    plan.py        the planner: CEM over keyframe motions, rolled out in the batched pH plant (Lecture 11, 1b)
    track_env.py   the tracking task: the robot's StandupAgent contract (72 -> 22), reward = track the plan + stand
    train.py       DAgger warm start from the plan (Lecture 11, option 1), then PPO (option 2) with the harness
"""
