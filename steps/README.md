# Steps: every work package of the thesis method, repeatable by name

One script per step of the roadmap in `docs/EXPERIMENTS.md`. "Repeat W0" means: run `steps/W0.sh`.

    steps/list.sh                          every step: its question, where it runs, what it needs, the result we got
    steps/W0.sh                            the derivative when a foot lands
    steps/W1.sh score grad-stand-v1c       score a standing run again
    steps/W1.sh launch 2                   show the Kaggle launch for training seed 2 (add --yes to start it)
    steps/W2.sh                            the checks of the walk task
    steps/W3.sh check | laptop | launch    the learned term 2: tests, a 40 min laptop run, the Kaggle comparison run
    steps/W5.sh check | laptop | launch    walking: tests, a 30 min smoke run, the Kaggle run; then fetch, score
    steps/W5b.sh check | gait | lift       air_time's smooth stand-in: tests, on a real gait, the learner's derivative
                                           (WEIGHTS="1 2" ITERATIONS=100)
    steps/W5b.sh laptop | launch           ... a 30 min smoke run (W=2, QUIET=1), the Kaggle run (W=1.0 by default)
    steps/W5c.sh check | lift | laptop     the gait clock: tests, the stepping check (WEIGHTS="2 4" ITERATIONS=100), a smoke run

| Step | Question | Runs on | Result |
|---|---|---|---|
| W0 | Is the derivative through the pH model right when a foot lands? | laptop, 5 min | yes: 6.6e-8 and 2.8e-8 against finite differences |
| W1 | Does the professor's formula learn to stand from scratch? | Kaggle 6 h per seed, scoring on the laptop 8 min | seed 1: 1.0 % falls, 2.3 W at 1.64 M steps; seeds 2 and 3 open |
| W2 | Is the walk task the same on both plants, and is the derivative right while walking? | laptop, 25 min | yes: every reward term within 1e-5 on both plants; derivative within 6e-7 over 1 s of walking (8 foot events), no blow-up |
| W3 | Can a learned term 2 replace the fixed one, and is standing as good with it? | laptop 50 min; Kaggle 6 h | yes, with learned weights (a free value network failed): Kaggle 0.5 % falls / 2.4 W at 0.82 M steps, against the fixed term 2's 1.0 % / 2.3 W at 1.64 M; stands straighter; the weights keep growing over a run (3.5 W at the end) |
| W4 | Speed for walking | moved after W5: a walking window costs about as much as a standing one | open |
| W5 | Does the professor's formula learn to walk from scratch? | laptop 30 min (smoke run); Kaggle 6 h; scoring 15 min | no: no steps, 93 % falls over the six commands with pushes (78 % when asked to stand); quiet and in MuJoCo it stands but does not move.  Cause: `air_time` has no derivative (W5b) |
| W5b | Does a smooth stand-in for `air_time` make the method step? | laptop: gait 1 min, lift 20-55 min, smoke run 30 min | not yet: it gives the slope; paid only for unloading kept 0.06 s (Eric's minimum), the optimiser shifts the weight from foot to foot instead of rocking (86 % off a foot at weight 2), but no foot comes up, and 30 min learner runs do not change; no Kaggle run |
| W5c | Does a gait clock (two inputs, a reward for stepping in time) make the method step? | laptop: lift 25-55 min, smoke run 30 min | in the stepping check yes: at weight 4 a foot comes up (0.12 s in 2 of 4 robots), the weight shifts in time with the clock; the learner did not step in 30 min (feet in the air 1.5 %); no Kaggle run yet (the 77 inputs in evaluation, export and scoring are open) |

## What a step script does

1. Prints its header: the question, where it runs and how long, what it needs, and the result we got ("Then").
2. Runs the commands, one job on one CPU core (`THREADS=2 steps/W0.sh` for more).
3. Prints the lines that carry the result ("Now") above the result we got then.
4. Keeps the whole output in `outputs/steps/<step>/<date>_<commit>.log` (not in git).

A step never starts a Kaggle run by itself: `launch` shows what it would start, `launch <seed> --yes` starts it.

## The same code as then

A step script runs on today's code. Every finished step has a git tag with the same name in both repositories
(Model Based RL and Full Body):

    git checkout W0 && git -C "../Full Body" checkout W0        # the code of that day; steps/ exists from tag W2 on
    git checkout main && git -C "../Full Body" checkout native-cuda

| Tag | Model Based RL | Full Body | What |
|---|---|---|---|
| `W0` | `0c3071b` | `e135c25` | the foot-landing derivative check |
| `W1-seed1` | `fa27c77` | `e135c25` | the code Kaggle cloned for `grad-stand-v1c` |
| `W2` | the commit tagged `W2` | `8522edc` | the walk task on both plants, the steps kit |
| `W3` | the commit tagged `W3` | `8522edc` | the learned term 2 (`--term2 weights`) and its Kaggle comparison; Kaggle cloned `98ab675` |
| `W5` | the commit tagged `W5` | `3406afd` | walking from scratch (`grad-walk-v1`, failed: no steps); Kaggle cloned `7b0e185` |

## Adding a step

Copy `W0.sh`. The comment block at the top is the step's record: `Wn: the question`, `Runs:`, `Needs:`, `Then:` (date,
tag, the numbers). `begin`, `run <command>` and `finish "<pattern of the result lines>"` come from `common.sh`.
When the step is finished: its numbers into `Then:` and into `docs/EXPERIMENTS.md`, commit, tag both repositories
(`git tag -a Wn -m "..."`), push with `--tags`.
