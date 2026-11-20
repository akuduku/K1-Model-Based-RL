# Thesis experiments: the professor's formula on the pH plant, against the team's PPO in MuJoCo

This file is the working plan of the thesis experiments: the rules, what exists, and the next work packages. Each
package has a goal, the files it touches, what to reuse, and a check that says it is done. Update the status tables
when a package is finished.

## Start here (state of 20 Nov 2026)

- **The thesis method** (decided by the user on 16 Nov): the professor's formula as written
  (`MA/ph_formula_rl_mbrl.pdf`): minimise L = running cost (term 1) + cost at the end (term 2) over the
  controller's weights, with the derivative taken **through the pH model**.  Code: `mbrl/grad/`.  Details, numbers
  and traps: section "Gradients through the pH model".
- **It works on standing:** Kaggle `grad-stand-v1c` (T4, 1024 robots, 6 h, from a random network, seed 1): the
  best in-run checkpoint (update 80, 1.6 M environment steps, 2.9 h) scores 1.0 % falls / 2.3 W with pushes and
  noise (3 scoring seeds x 128 robots), 0 % / 1.1 W quiet, and stands 20 s in MuJoCo.  PPO on pH: 0 % / 2.1 W
  after 295 M steps.  So: close to PPO's quality on standing with about 180 times fewer environment steps.  Open: one
  training seed so far, and PPO's row is the end of its 7 h run (when PPO first reached this quality has not been
  looked up, so no claim about wall time yet).  Numbers: table "Reference rows for stage 1" and W1.
- **The walk task exists on both plants (W2, 17 Nov):** Eric's rewards and commands written once
  (`Full Body/rl/walk/task.py`), on the C++ plant (`rl/walk/env.py`) and on the differentiable plant
  (`mbrl/grad/walk_env.py`).  Both pay the same for the same motion, and the derivative is right while the robot
  walks (within 6e-7 over 1 s with 8 foot landings and lift-offs) and does not blow up.  Nothing is trained yet.
- **Term 2 learned (W3, done 19 Nov):** a free value network failed twice on the laptop (its slope has nothing to
  do with the physics); term 2 with learned weights works: the costs the end state causes, each weighted by how long
  it will last, learned from the returns, starting at the fixed term 2.  Kaggle `grad-stand-weights` (the settings
  of `grad-stand-v1c`, only term 2 changed): best in-run checkpoint (update 40, 0.82 M steps, 1.4 h) 0.5 % falls /
  2.4 W, tilt 0.30°, stands 20 s in MuJoCo.  So the fixed term 2's quality in half the training, standing
  straighter.  Open: the weights keep growing over a run and the power with them (3.5 W at the end).  Numbers: W3.
- **Every step can be repeated by name:** `steps/W0.sh` ... `steps/W3.sh` (`steps/list.sh`, `steps/README.md`);
  each finished step has a git tag of the same name in both repositories.
- **Walking from scratch failed (W5, 20 Nov):** Kaggle `grad-walk-v1` (6 h, 3.15 M steps) took no steps: the
  robots stood or slid in place, the curriculum never left its first stage, and with the walk world's pushes, delays
  and gain errors 93 % fell over the six scoring commands (78 % even when asked to stand).  Quiet, and in MuJoCo, it
  stands but does not move.  Cause, examined: Eric's `air_time`, the only term that pays for lifting a foot, has no
  derivative and reached the learner in no way.  Numbers: W5.
- **W5b (20 Nov, laptop):** a smooth stand-in for `air_time` (`--air-time-smooth`, the weight a foot has taken off
  the ground) gives the learner the missing slope: following it, weight comes off a foot.  The first version was
  collected by rocking between the feet.  Paid only for unloading kept 0.06 s (Eric's minimum), it makes the
  optimiser shift the weight from foot to foot at a walking rhythm, but no foot comes up, and 30 min learner runs do
  not change.  No Kaggle run.  Numbers: W5b.
- **W5c (20 Nov, laptop): a gait clock**, the way Booster trains its own T1: two clock inputs and a reward for
  stepping in time with them, the inputs at 0 while standing.  In the stepping check a foot comes up (0.12 s, 2 of 4
  robots, weight 4); the learner did not step in a 30 min laptop run.  The firmware needs a walk agent with 77 inputs
  (drafted for Eric, K1_Deploy branch `walk-clock`).  Numbers: W5c.
- **Next:** the 77 inputs in the evaluation, export and scoring, then a real run; longer windows (W4).  W1's standing
  seeds 2 and 3 wait until the end (only Kaggle time, no code).
- The planner (`mbrl/mpc`, packages 3-5 below) and PPO on pH are **comparison rows** now, not the thesis method.
- Commands: tests `scripts/test_all.sh`; Kaggle runs are started by the user (`kaggle/run.sh`, commit and push
  first); Python: `~/Dokumente/Robotics/venv/bin/python`; on the laptop one gradient job at a time (7 GB of memory)
  and `--threads 1`.

## The question and the rules

**This thesis** = the port-Hamiltonian plant as the model + model-based RL on it.  **The work it is compared with**
= how the team trains today: model-free PPO in MuJoCo (Eric Plass's k1_kick, k1_sprint, RL_Core).  The two are kept
apart everywhere: in the names, the tables and the code.

| Row | Physics | Learning | Whose | Role |
|---|---|---|---|---|
| **Thesis approach** | pH plant (Full Body, fitted to robot logs with `mbrl/fit.py`) | the professor's formula: the cost is minimised over the controller's weights with derivatives through the pH model (`mbrl/grad`) | this thesis | the result |
| **Baseline** | MuJoCo | model-free PPO (rsl-rl) | the team (Eric's policies and training) | what the thesis is measured against; never a part of the thesis approach |
| Ablation: PPO on pH | pH plant | model-free PPO (Full Body `rl/<skill>/train.py`) | this thesis | is a difference due to the physics (pH instead of MuJoCo)? |
| Ablation: planner on pH | pH plant | at every control step a planner tries variations in the pH model; policy and value learn with it (`mbrl/mpc`); the model is a black box | this thesis | is a difference due to using the model's derivatives? |

In the older sections of this file (packages 3-5) "model-based arm" means the planner, which was the thesis
approach until 16 Nov; "PPO arm" = PPO on pH unless MuJoCo is named.

The question: does model-based RL on the pH model reach the quality of the team's PPO-in-MuJoCo policies, on the pH
plant, in MuJoCo and on the robot, and at what cost (environment steps, model steps, hours)?  "It does not" is a
valid result.  (Checked against Plaat, *Deep Reinforcement Learning* (2022), ch. 5 on 16 Nov 2026: the protocol
rules 4-6 below come from there.)

Rules (set on 15 Nov 2026):

1. The model-based arm learns **from scratch**. No policy, motion or demonstration from Eric Plass's k1_kick or walk
   goes into it, and no PPO stage. From his work only the **task and the rewards** are used.
2. A skill that needs an earlier skill (the kick needs standing and walking) starts from that arm's **own** earlier
   stage. Nothing crosses between the arms.
3. Both arms: the same plant, task, rewards, randomisation (noise, pushes, delays, gain errors), the same final
   network (512-256-128 ELU, the firmware's `.bin`), the same evaluation script.
4. Cost is reported as: **environment steps** (executed; what a robot did), **model steps** (planning included),
   and **hours** on the named GPU.  The book's warning (5.3.3): fewer environment steps do not mean less time.
5. Quality is scored on an environment that is **not** the planning model: MuJoCo first (`rl/stand/eval_mujoco.py`),
   the real robot where possible; the pH numbers stand next to it.
6. A row of the thesis table needs **3 seeds** per arm (results depend on the seed, book 5.3.1).  The checkpoint
   that counts is the one with the best evaluation inside the run, by the same rule for both arms.
7. Correct before fast. A result that only holds with a trick is reported as such or not at all.

Eric's policies are used for one thing only: to check an environment (his MuJoCo-trained kick runs in `rl/kick`
unchanged, so the env, the ball and the observations are right). `mbrl/kick/` (a kick plan that acts through his
walk policy and can start from his kick motion) is a **reference**, not the model-based arm.

## Stages

| Stage | Skill | Agent contract | PPO arm | Model-based arm |
|---|---|---|---|---|
| 1 | stand under pushes | firmware `stand`, 72 -> 22 (`rl/stand`) | done: `stand-ph-v1` (3000 iterations, ~295 M steps, 7 h on a T4, no falls) | planner: `mb-stand-v1` + fix (6.8 W, 5.5 % falls); **thesis method: `grad-stand-v1c` (1.6 M steps, 2.9 h on a T4: 1.0 % falls, 2.3 W); with term 2's weights learned (W3): `grad-stand-weights` (0.82 M steps, 1.4 h: 0.5 % falls, 2.4 W); seeds 2 and 3 open (W1)** |
| 2 | walk on velocity commands, standing included; run = higher commands | sprint family, 75 -> 22 | task done (W2); training open (W5) | task done and checked (W2); first training `grad-walk-v1` (W5, 3.15 M steps, 6 h on a T4) failed: no steps, 93 % falls over the six commands; cause examined (W5b) |
| 3 | kick from standing | kick_walk, 165 (166 with the request clock) -> 22 | open (W6) | open (W6) |
| 4 | walk, stay and kick | kick_walk | open (W6) | open (W6) |

(The last column is the thesis approach; until 16 Nov it was the planner.)

Stage 1 tests the method on its own. Stages 2-4 share one contract family: the walk's 75 inputs are the first 75 of
kick_walk, so a network continues from the previous stage with its new inputs at zero weight.

## What exists (15 Nov 2026)

Full Body (`native-cuda` = `main`):
- the standard robot model `assets/K1_standard/` (the older hand variant only via `ph_k1.model.HAND_XML`);
- the ball in the plant, Python reference and C++ (`ph_k1/ball.py`, `ph_k1_cuda/include/ph_k1/step.hpp`);
  `validation/ball_vs_mujoco.py`. **Its CUDA build has not been compiled yet (package 2).**
- `rl/stand` (env, PPO training, export, MuJoCo check), `rl/getup`, `rl/walk` (evaluation tools only, no env),
  `rl/kick` (env, evaluation, Hamiltonian figure, PPO training);
- the harness `rl/common/harness.py` (W&B, time budget, clean stop, checkpoints with `.bin`, videos, evaluation).

Model Based RL (`main`): the fit to robot logs (`mbrl/fit.py`, made on the older hand model), the get-up planner and
tracker (`mbrl/getup`), the reference kick plan (`mbrl/kick`, `plans/kick-3p5-v0.pt`, `plans/kick-v1.pt`), the Kaggle
kit (`kaggle/`).

Results so far that belong in the thesis:
- Eric's MuJoCo-trained kick on the pH plant: kick-016 100 % kicked, 9.6°, full power 4.57 m/s, 0.2 % falls (his own
  MuJoCo env: 99.3 %, 9.4°, 4.59 m/s, 2.1 %). `python rl/kick/evaluate.py --policy ../k1_kick/checkpoints/kick_016.pt -B 512`
- The ball against MuJoCo's: roll identical, ball speed of a leg swing within 13 % (`validation/ball_vs_mujoco.py`).
- PPO from scratch did not learn the get-up (getup-mb-v1d/e); the get-up plan stood up but its tracker did not.

## Working on the code

    scripts/test_all.sh                 # in each repository: all tests on the CPU, about a minute
    ~/Dokumente/Robotics/venv/bin/python   # the local Python (torch, mujoco, rsl-rl-lib 5.5.2, tensordict)
    cd "Full Body/ph_k1_cuda" && python setup.py build_ext --inplace     # after any change under ph_k1_cuda/

- The laptop has no GPU: it runs tests, small experiments (a few hundred robots, ~5000 robot-steps per second) and
  viewers. Trainings run on Kaggle.
- Kaggle clones GitHub: **commit and push both repositories first**, and bump the submodule
  (`git -C external/full-body fetch && git -C external/full-body checkout <commit> && git add external/full-body`).
  Every kernel builds the CUDA plant and runs the tests on the GPU before it trains.
- Ask the owner before any Kaggle launch (quota: about 30 GPU-hours per week).
- Gitea needs the HTWK VPN (`snxctl connect`). Slides, PDFs and notes stay out of the repositories.

## Status of the packages (update this list)

| Package | Status |
|---|---|
| 1 secure, commit, handover file | done 15 Nov |
| 2 ball on the GPU | done 16 Nov inside the `mb-stand` kernel: nvcc build, 70 Full Body tests and 40 tests of this repo passed on the T4 (`outputs/kaggle/mb-stand-v1/logs/setup.log`) |
| 3 snapshot / restore | done 15 Nov for the plant and `StandEnv` (test `rl/stand/tests`, exact) |
| 4 learner on standing | **done 16 Nov, laptop gate passed** (see "Package 4: findings"): the policy alone stands |
| 5 stage 1 result | **`mb-stand-v1` done 16 Nov: not level with PPO, drifted after its best point. Cause found and fixed the same day (planner centred, small noise for refining): 14.0 W -> 6.8 W on the laptop. Next: `mb-stand-v2` on Kaggle (see "Package 5: findings")** |
| 6-7 | open; do not start before the learner holds its best point |

Reference rows for stage 1 (`python rl/stand/evaluate.py --policy <file> -B 128`, 10 s episodes, pushes and noise):

| Policy | Fell | Stood | Power | Tilt |
|---|---|---|---|---|
| doing nothing (`--policy zero`) | 100 % | 3.1 s | 8.6 W | 6.2° |
| PPO `stand-ph-v1` (`deploy_2999.bin`, 295 M executed steps) | 0 % | 10.0 s | 2.1 W | 0.33° |
| model-based, laptop (`policies/mb-stand-laptop3_346.bin`, 0.53 M executed + 21 M model steps, 85 min on 6 CPU cores) | 11.7 % | 9.65 s | 14.4 W | 2.5° |
| **professor's formula, Kaggle `grad-stand-v1c`, best in-run checkpoint** (`policies/grad-stand-v1c_80.bin`: 1024 robots, 1.64 M environment steps, 2.9 h on a T4; training seed 1; scored with seeds 7, 8, 9 = 384 episodes) | **1.0 %** | 9.98 s | **2.3 W** | 0.46° |
| the same run at its end (`policies/grad-stand-v1c_155.bin`, 3.17 M steps, 5.6 h; same 384 episodes) | 0.5 % | 10.0 s | 2.4 W | 0.54° |
| **professor's formula with term 2's weights learned (W3), Kaggle `grad-stand-weights`, best in-run checkpoint** (`policies/grad-stand-weights_40.bin`: 1024 robots, 0.82 M environment steps, 1.4 h on a T4; training seed 1; same 384 episodes) | **0.5 %** | 9.99 s | **2.4 W** | 0.30° |
| the same run at its end (`policies/grad-stand-weights_164.bin`, 3.36 M steps, 5.6 h; same 384 episodes) | 0.5 % | 9.99 s | 3.5 W | 0.32° |
| **professor's formula, from scratch on the laptop** (`policies/grad-stand-laptop_74.bin`: gradients through the pH model, 16 robots, 23 680 environment steps, 41 min on one CPU core; one seed) | **2.3 %** | 9.94 s | **2.8 W** | 0.68° |
| the same run at update 41 (`policies/grad-stand-laptop_41.bin`, 13 120 steps, 20 min) | 6.2 % | 9.86 s | 2.5 W | 0.73° |
| planner with the fix of 16 Nov (`policies/mb-stand-s05_392.bin`, from `mb-stand-v1`'s best + 50 min laptop) | 5.5 % | 9.87 s | 6.8 W | 0.99° |
| model-based, Kaggle `mb-stand-v1`, best checkpoint (`policies/mb-stand-v1_260.bin`, 1.6 M executed + 64 M model steps, 2.2 h on a T4) | 2.3 % | 9.94 s | 14.0 W | 1.3° |
| model-based, Kaggle `mb-stand-v1`, end of the run (`policies/mb-stand-v1_708.bin`, 4.35 M executed + 174 M model steps, 6 h) | 10.2 % | 9.73 s | 24.6 W | 2.1° |

Without pushes and noise (`--quiet`): PPO 0 % falls, 1.0 W; professor's formula on Kaggle 0 % falls, 1.1 W, tilt
0.15° (update 80) and 1.2 W, 0.59° (end); with learned term-2 weights 0 % falls, 1.0 W, 0.22° (update 40) and 1.4 W,
0.06° (end); on the laptop 0 % falls, 0.9 W (update 74) and 0.8 W (update 41); planner with the fix 0 %, 5.3 W; model-based 0 % falls at 13.3 W (best) and 20.9 W (end).
In MuJoCo (`rl/stand/eval_mujoco.py --seconds 20`, quiet and `--push 0.3`): all of them stand the 20 s; largest tilt
PPO 0.4° / 0.7°, professor's formula (`grad-stand-v1c`) 0.7° / 1.6° (update 80) and 0.9° / 1.4° (end), with learned
term-2 weights (`grad-stand-weights`) 0.1° / 0.7° (update 40) and 0.4° / 0.7° (end),
model-based best 3.9° / 3.9°, model-based end 1.5° / 2.1°.

**Stage 1 as it stands (16 Nov): the model-based arm learns to stand from scratch with about 1 M executed steps
(PPO: about 14 M), but its policy is not as good as PPO's: 7 times the power, and it still falls under pushes.
It is not a result for the thesis table yet.**  (This paragraph is about the planner, `mbrl/mpc`.  The thesis
method's standing result of 17 Nov is in the two `grad-stand-v1c` rows above and under W1.)

## Work packages

Do them in order. Each ends with `scripts/test_all.sh` green, a commit, a push, and the tables above updated.

### Package 2: the ball on the GPU (small; 15 min of Kaggle)
- Goal: the CUDA build of the ball code compiles and all tests pass on a T4.
- Do: `cd "Full Body" && kaggle/run.sh MJX=0 BATCHES=1024 NATIVE_BATCHES=1024,4096` (the benchmark kernel builds with
  nvcc and runs `ph_k1_cuda/tests` on the GPU), then `kaggle/fetch.sh`.
- Done when: the kernel log shows the native tests passing on cuda, `test_ball.py` included. If nvcc rejects
  something, it is in `step.hpp` (the `ball_*` functions) or `model_data.hpp`.

### Package 3: copy a robot's state (medium, laptop only)
- Goal: the planner must start model rollouts from a world robot's exact state.
- Do: `K1PHPlantNative.snapshot(rows=None) -> dict` and `.restore(snap, rows=None)` in `Full Body/ph_k1/native.py`
  (tensors `q, p, xi, lam, ball, _led, last_target, ring, ring_idx, lag, ring_kp, ring_kd`); the same pair on
  `StandEnv` (`rl/stand/env.py`: the plant's snapshot plus `actions, prev_actions, qd, prev_qd, gyro, v_body, contact,
  kp, kd, push_timer, episode_length_buf, episode_sums`).
- Done when: a test builds two envs, runs one for 50 steps, restores the other from its snapshot, steps both with
  the same actions for 30 more steps (pushes and noise off, `delay_hold_prob=None`) and gets identical states and
  rewards (float64).

### Package 4: the model-based learner, on standing (large; the hard one)
- Goal: `Model Based RL/mbrl/mpc/`: known-model planning with a learned policy and value (the recipe of POLO,
  Lowrey et al. 2019, "Plan online, learn offline", plus a distilled policy).
- World = the task env with its noise, pushes and delays, R robots. Model = a second instance of the same env class,
  quiet (no noise, no pushes), R x N robots.
- Every control step: restore each world robot's snapshot into its N model copies; roll the model H steps with
  actions = policy(model observation) + time-correlated Gaussian noise (candidate 0 without noise); score =
  sum of discounted model rewards + discounted value at the end (no value after a termination); the executed action
  is the MPPI-weighted mean of the candidates' first actions.
- Learning after every T executed steps: the policy regresses on the executed actions (a replay of the last
  rollouts); the value network fits lambda-returns of the executed rollouts (as `dagger_warm_start` in
  `mbrl/kick/train.py` computes them). Both networks start random.
- Networks and checkpoints in rsl-rl's layout (`actor_state_dict` with `mlp.N.weight/bias`,
  `obs_normalizer._mean/_std`), so `rl/common/export.py` writes the `.bin` and `rl/kick/policy.py` loads them.
- Its own loop with what the harness gives PPO: W&B, `--hours`, clean stop, timed checkpoints, resume, video
  (`rl/common/harness.VideoMaker`), evaluation. Log `Cost/executed_steps`, `Cost/model_steps`, `Plan/return_planner`,
  `Plan/return_policy`, `Plan/weight_entropy`, `Learn/policy_loss`, `Learn/value_loss`.
- Start sizes: laptop R 16, N 16, H 10; T4 R 256, N 32, H 12, T 24, gamma 0.99, lambda 0.95.
- Also: `rl/stand/evaluate.py` in Full Body (one evaluation for both arms: falls, time to fall, tilt, watts; with and
  without pushes and noise).
- Done when (laptop, 30-60 min, 16 robots): robots under the planner stand clearly longer than the 3.8 s of doing
  nothing, and the policy alone (no planner) improves over the run. If not: horizon 10 -> 20, candidates 16 -> 64,
  the noise level, value targets from the planner's own estimate.

#### Package 4: findings (16 Nov)
- First laptop run (`--robots 16 --candidates 16 --horizon 10`, 24 iterations, 9216 executed steps): the robots
  under the planner fell after 1.3-1.8 s, sooner than with no action at all (3.1 s); the bare policy fell after 1.5 s.
  The planner did score above the bare policy in its own rollouts, so the mechanics work; what it optimised was wrong.
- Suspected cause: the untrained value network. Rewards inside a 0.2 s horizon sum to about 0.3, the value estimate
  at its end was up to 3 and noisy (value loss jumped to 2), so the planner followed the value's noise. The value was
  fitted on 384 samples per iteration, far too few.
- Next steps, in this order: (a) the planner alone, value = 0, policy = start pose: does it stand longer than 3.1 s,
  and with which horizon (10, 20, 35 steps)? (b) fit the value on the replay of the last rollouts, lower its learning
  rate, and let its weight in the planner's score rise from 0 over the first iterations; (c) more world robots.
- (a) done: the planner alone (value 0, policy = start pose) does not stand: 3.1-3.4 s with 10, 20 and 35 steps of
  horizon, as with no action.  A fall takes 3 s; inside the horizon "do nothing" scores best (moving costs
  action-rate, the fall's penalty is out of sight).  The value network has to carry that knowledge.
- (b) + (c) done (value on the replay, `value_lr` 3e-4, `value_warmup` 10-40 iterations, 64 robots x 8 x 5):
  laptop run 2 (40 min, 165 iterations, 253 k executed and 10.1 M model steps): **the robots under the planner stand
  9.5-9.8 s of 10 s** from iteration ~140 on (PPO `stand-ph-v1` stood whole episodes after about 14 M steps).
  The bare policy still fell after ~2 s: it was only in control for single steps.
- So `policy_share` is now a fixed half of the robots that run the bare policy for whole episodes (DAgger); the log
  shows the episode lengths of both groups.  Laptop run 3 continues run 2 with it
  (`--resume logs/mb-stand-laptop2/model_165.pt`).  **The gate needs: the policy group's episodes near 10 s and
  `rl/stand/evaluate.py` on the `deploy_<it>.bin` with few falls.**
- Laptop run 3 (45 min more, to iteration 346, 531 k executed and 21.3 M model steps in total): from iteration ~190
  on both groups stand 9.5-10 s.  The policy alone (`rl/stand/evaluate.py`, 128 robots): 11.7 % falls with pushes and
  noise, 0 % without.  **Gate passed.**  Settings that worked: `--robots 64 --candidates 8 --horizon 5 --sigma 0.2`
  (a short horizon is enough once the value is learned; many world robots matter more than many candidates).
- Open questions for the long run: does it reach PPO's 0 % falls and 2 W?  The energy term of the reward is the same
  for both arms, so the gap should close with training; if it does not, look at `sigma` (the planner's noise is also
  what the policy imitates) and at lowering it over the run.
- Trap: do not stop a run with `pkill -f mbrl.mpc.train` from the same shell line that mentions it; it kills the
  shell too. Use the process id.

### Package 5: stage 1 result (small; ~6 h of Kaggle)
- Do: `TASK=mb-stand kaggle/run.sh HOURS=6 RUN_NAME=mb-stand-v1` (the task exists in `kaggle/train_mb.py`: 256 world
  robots x 8 x 5; W&B project "Akos pH MB"), then `KERNEL_SLUG=mbrl-k1-mb-stand TASK=mb-stand kaggle/fetch.sh mb-stand-v1`; `mbrl/compare.py` prints the table for two policy files (quality, falls, watts, executed and
  simulated steps); also `Full Body/rl/stand/eval_mujoco.py` for both.
- Done when: the comparison table of stage 1 is in this file. If the model-based policy is clearly worse, fix the
  learner before any later stage: it is the same learner everywhere.

#### Package 5: findings (16 Nov, `mb-stand-v1`, 256 x 8 x 5, sigma 0.2, 6 h)
- The evaluation inside the run (128 robots, every 20 min): 100 % falls at 0.28 M executed steps, 11 % at 0.8 M,
  0 % at 1.05 M (13.6 W).  Best stretch: 1.05-1.6 M steps (0-2 % falls, 13-14 W).
- After that the run got slowly **worse**, not better: reward per step 0.025 -> 0.021, power 14 -> 18-25 W, falls
  under pushes 0-2 % -> 10-18 %, the action-rate penalty 5 times larger.  Quiet evaluation stayed at 0 % falls.
- What the log says about the cause (a reading of the log, not yet proven): once the robot stands, the best of the
  8 candidates scores only 0.3 % above the bare policy (`Plan/return_best` 2.44 against `Plan/return_policy` 2.43).
  With 5 steps of horizon the model's own rewards are about 0.1 of a score of 2.4; the rest is the value network at
  the horizon.  So the ranking of the candidates is decided by small differences of the value network, the policy
  imitates that, the value then fits the changed policy: nothing ties the loop to the real reward, and it wanders.
- New in the log since 16 Nov: `Plan/reward_step_planner` against `Plan/reward_step_policy` (reward in the world of
  the robots the planner controls against the robots the bare policy controls) and `Plan/correction` (mean size of
  planner action minus policy action).  If the planner's robots do not earn more than the policy's, imitating the
  planner cannot improve the policy.
- Laptop check 1 (16 Nov; `model_260.pt`, networks frozen with `--lr 0 --value-lr 0`, 64 x 8, sigma 0.2, the old
  weighted mean of the candidates): reward per step of the planner's robots against the policy's robots:
  horizon 5: 0.0220 against 0.0231 (-4.6 %, standard error 0.5 %, 35 iterations); horizon 25: 0.0214 against
  0.0231 (-7.1 % +- 1.3 %, 18 iterations).  **The planner makes a standing robot worse, and a longer horizon does
  not repair it.**  So the policy was imitating something worse than itself.
- Found in the planner: it executed the weighted mean of 8 noisy candidates; with equal scores that is still the
  mean of their noise, new at every control step (jitter).  Since 16 Nov the planner is centred (`center`, default
  on): action = policy action + sum (w_i - 1/N) (candidate_i - policy action); `--no-center` gives the old one.
  `--temperature-abs` weighs score differences in reward units instead of against the candidates' own spread.
- Laptop check 2 (16 Nov; the centred planner, same set-up, horizon 5): sigma 0.2: -4.6 % +- 0.6 (32 iterations,
  no better than before); **sigma 0.05: +12.0 % +- 0.5 (36 iterations): the planner's robots now earn more than
  the policy's.**  Horizon 15 with sigma 0.05: +5.6 % +- 0.6 (13 iterations, still running when written).  So near
  a good stand the planner helps only with small noise; sigma 0.2 was right for learning to stand at all and wrong
  for refining it.
- **Learning run with the fix** (`mb-stand-s05`: from `model_260.pt`, 64 x 8 x 5, `--sigma 0.05 --lr 1e-4
  --value-lr 1e-4`, 50 min on the laptop, 0.2 M more executed steps): `policies/mb-stand-s05_392.bin` on the C++
  plant, 128 robots: **6.8 W** (from 14.0), tilt 1.0 deg (from 1.3), falls under pushes 5.5 % (from 2.3 %); quiet:
  5.3 W, no falls.  The policy improves again instead of drifting.  Open: the falls under pushes went up (small
  noise explores less against pushes), and PPO-on-pH is at 2.1 W / 0 %.
- Next for the Kaggle run `mb-stand-v2`: sigma 0.2 until the robots stand (the in-run evaluation shows it), then
  0.05; `--lr 1e-4` after a resume (the replay is empty then).  A schedule for sigma is not written yet.
- Fixes to try, in this order, each on the laptop first from `model_260.pt` (`--resume`): (1) a longer horizon
  (25 steps = 0.5 s) so that model rewards carry the score; (2) lower `--sigma` (0.2 -> 0.05) once it stands, the
  policy imitates that noise; (3) keep the checkpoint with the best evaluation and say so in the thesis (the same
  rule for PPO).

### Gradients through the pH model (test of 16 Nov; `mbrl/grad/`)
Question: can the policy learn from the gradient of the reward through the pH model (SVG / Dreamer / SHAC in the
literature) instead of from random tries?  The Python plant (`Full Body/ph_k1/plant.py`, `firmware.py`) is PyTorch,
so autograd runs through it **without any change to the plant**.
- `python -m mbrl.grad.check --steps 1 5 20 [--push 0.3]`: d cost / d joint targets of a standing robot (foot
  contact, PD, motor driver; cost = copper energy + tilt) against central finite differences:
  1 control step (20 plant steps): worst relative error 2.5e-7; 5 steps: 5.7e-8; **20 steps (400 plant steps):
  6.2e-7**, gradient norm 3.3 / 6.8 / 17 (grows about linearly, no blow-up); with a 0.3 m/s sideways push at the
  start (contact changes): 4.9e-8.  **The gradients are right and stay usable over 0.4 s of standing.**
- Speed (laptop, one thread, 8 robots): 16 ms forward and 49 ms backward per plant step, i.e. about 13 s for one
  gradient through 10 control steps.  With PyTorch's default threads next to other jobs it was 30 times slower:
  always pass `--threads 1`.  A real training needs derivative code in the C++/CUDA plant or the Python plant on
  a GPU; neither exists.
- `python -m mbrl.grad.refine --policy <model.pt> --threads 1 ...`: rl/stand's task rebuilt on the Python plant
  (`GradStand`: contract, firmware chain with slew and 20 ms delay, the smooth reward terms), truncated
  backpropagation through K control steps, Adam on the policy.
- Results of `refine` from `model_260.pt` (13.3 W on the Python plant, the same as on the C++ plant), 10 control
  steps per rollout, **without** a value network: lr 3e-4, 16 robots: better for two steps (11.6 W), then broken
  (50 W at step 5, falls from step 8; `deploy_10.bin` on the C++ plant: 100 % falls).  lr 2e-5 (all layers) and
  lr 1e-4 (last layer only), 8 robots: no falls, but the power rises steadily: 13.4 -> 36 W and 13.4 -> 80 W in
  12 iterations (runs cut off there).  **Correct gradients, but following them over a 0.2 s window makes the stand
  worse**: the window does not see what the change costs later (the same lesson as the planner without its value
  network).
- `--value` adds the checkpoint's value network at the end of the window (the SHAC recipe).  **Not tested yet**:
  the runs were stopped after one iteration because the laptop (7 GB) ran out of memory with three jobs.  Run
  one job at a time: `python -m mbrl.grad.refine --policy outputs/kaggle/mb-stand-v1/runs/mb-stand-v1/model_260.pt
  --threads 2 --robots 8 --steps 10 --lr 2e-5 --value --minutes 25 --save-every 20 --run-name grad-stand-f`, then
  score `logs/grad-stand-f/deploy_<it>.bin` with `Full Body/rl/stand/evaluate.py`.
- **Decision of 16 Nov (evening): this is the thesis method.**  It is the professor's formula as written
  (`MA/ph_formula_rl_mbrl.pdf`): minimise L = term 1 (running cost) + term 2 (cost at the end) over the controller's
  weights, through the pH model.  The planner (`mbrl/mpc`) and PPO on pH are comparison rows.
- What was wrong with the first learner: no term 2 (it only saw the next 0.2 s), an uncontrolled step, and later
  one sign error of mine in the controlled step (fixed; `refine.py` comments it).
- Now in the code (`mbrl/grad/env.py`, `refine.py`, `train.py`): **term 2** = `GradStand.terminal_cost`: kinetic
  energy 1/2 p^T M^-1 p at the end of the window plus pose, tilt and height error; **controlled step** = action
  targets y = a + delta * g / RMS(g), the network is fitted to them, each robot's g at most 3 x the batch median;
  **memory** = one `torch.utils.checkpoint` block per control step (`tests/test_grad.py`: same gradient);
  any device and precision.
- Refinement test with term 2 and the controlled step (`grad-stand-prof`: from `model_260.pt`, 8 robots, window 20,
  delta 0.03): inside its own windows the cost falls (reward per step -0.016 -> -0.010 in 8 updates), but the
  checkpoint after 10 updates **falls from the start pose after 1.5 s, on the C++ plant and on the Python plant
  alike** (the original stands).  So it is not a difference between the plants: 8 robots that only ever continue a
  steady stand are too narrow a set of states; the network forgets the start.  A refinement on 8 robots cannot
  answer more.  The learner for the real run therefore restarts robots at staggered times, and has noise and pushes.
- `python -m mbrl.grad.train` = standing **from scratch** with this method (the world of the other rows, scoring on
  the C++ plant).  Kaggle tasks: `TASK=grad-bench kaggle/run.sh HOURS=1` (speed and memory, GPU against CPU; run
  it first, its numbers set `GRAD_ROBOTS`), `TASK=grad-stand kaggle/run.sh HOURS=6 RUN_NAME=grad-stand-v1`.
- **First result from scratch (16 Nov, `grad-scratch-laptop`):** `python -m mbrl.grad.train --robots 16 --steps 20
  --threads 1 --minutes 40 --eval-every-min 10 --eval-robots 64 --no-wandb` on the laptop: a random network, the
  world with noise, pushes and delays, 74 updates, 23 680 environment steps, 40.8 min.  In-run evaluation
  (C++ plant, 64 robots): update 21: 26.6 % falls with pushes, quiet 0 % / 1.25 W; update 41: 3.1 %, 2.5 W, quiet
  0.86 W; update 61: 26.6 %, 6.1 W (worse again: 16 robots make noisy updates); update 74: 4.7 %, 2.8 W.
  Scored like the other rows (128 robots): see the table at the top of this file.  In MuJoCo
  (`eval_mujoco.py --seconds 20`): stands quiet (largest tilt 1.0 deg) and with `--push 0.3` (1.6 deg).
  **One seed, one run; not steady from update to update.  Next: more robots per update and a shrinking step
  (`--delta-final`), 3 seeds.**
- Speed: laptop, one thread: about 10 robot-control-steps per second with gradient (16 robots, 30 s per update).
  Kaggle CPU (2 threads, float32, `grad-bench` of 16 Nov): 64 robots 24.5, 256 robots 38.5 robot-steps/s, 3.5 GiB.
  Kaggle T4: failed at first: PyTorch 2.11+cu130 has no GPU code for batched `torch.linalg.solve` on a T4
  ("no kernel image").  Since Full Body e135c25 the plant's solves go through `ph_k1/solve.py`, which solves on the
  CPU when the GPU cannot (`probe`).  With it (`grad-bench` run 3, T4, float32, robot-control-steps per second
  with gradient / GPU memory): 64 robots 19.9 / 0.9 GiB; 256 robots 66.9 / 3.6 GiB; 1024 robots 154.7 / 14.2 GiB
  (too close to the T4's 15 GiB).  Memory is about 14 MB per robot, so 512 robots need about 7 GiB (not measured).
  For comparison: the C++ plant runs about 11 600 steps per second forwards on the same GPU.
- **Memory: two failed launches, then the fix (16 Nov, night).**  `grad-stand-v1` (512 robots) and `grad-stand-v1b`
  (256 robots, window 20) both ran out of the T4's memory.  Cause: the window was one graph with a
  `torch.utils.checkpoint` block per control step, and with that the memory still grew with the window (laptop,
  32 robots, float32: 1.23 GiB at a window of 2, 1.88 GiB at 12; on Kaggle's PyTorch 2.11 more); the benchmark had
  used a window of 2 and hid it.  The fallback to half the robots did not free the GPU memory either.
  **Fix:** `mbrl/grad/train.py: window` now runs forward without a graph, remembers the state before every control
  step and every random draw, and goes backward one control step at a time, carrying the co-state (the derivative of
  everything later with respect to the state).  Memory is one control step: 1.25 GiB at a window of 2, 1.29 GiB at
  12.  `tests/test_grad.py::test_window_step_by_step_equals_one_graph`: the same action gradients as one graph, to
  1e-8, with noise, pushes and changing delays.  `grad-bench` now times this same `window`.
- `--robots 0` (the Kaggle default `GRAD_ROBOTS = 0`): the run measures one short window with 64 robots on the GPU and
  takes as many robots as fit in 60 % of its memory.  No more sizes by hand.
- Relaunched 16 Nov, about 23:50: `TASK=grad-stand kaggle/run.sh HOURS=6 RUN_NAME=grad-stand-v1c` (window 20,
  delta 0.05 -> 0.01, W&B "Akos pH MB").  Fetch:
  `KERNEL_SLUG=mbrl-k1-grad-stand TASK=grad-stand kaggle/fetch.sh grad-stand-v1c`.
- Ways to more speed, in this order: the T4 with many robots; sole contacts only and 2 ms steps in the gradient
  plant; derivative code in the C++/CUDA plant (the large work package).

## Roadmap after standing (written 16 Nov; for the thesis method `mbrl/grad`)

Order of the skills: stand -> walk -> run (the walk task with higher commanded speeds) -> kick from standing -> walk
and kick.  Only standing exists on the differentiable plant.  Each package ends with the tests green, a commit, a
push and its result in this file.

### W0: is the derivative right when a foot lands?  **Done 16 Nov: yes.**
`python -m mbrl.grad.check --steps 5 20 --drop 0.02`: the robot starts 2 cm above the floor and lands after about
64 ms, inside the window.  Autograd against finite differences: worst relative error 6.6e-8 (5 control steps) and
2.8e-8 (20 steps); gradient norm 9.3 and 14.2, the same size as in a quiet stand (6.8 and 17).  So one landing
neither breaks nor blows up the derivative.  Not covered by this: repeated steps, sliding feet, the ball.

### W1: standing, solid  (seed 1 done 17 Nov; seeds 2 and 3 at the end of the work, decided 19 Nov)
- **`grad-stand-v1c`, fetched 17 Nov** (`outputs/kaggle/grad-stand-v1c/`).  The run sized itself to 1024 robots
  (8.1 MiB per robot measured), 156 updates of 20 480 environment steps in 5.6 h: 129 s per update (43 s forward,
  87 s backward), about 160 robot-steps per second.  No out-of-memory fallback.
- In-run evaluations (128 robots, every 10 updates), falls with pushes: 5.5, 2.3, 0.8, 0.8, 3.1, 2.3, 0, 0, 1.6, 0,
  3.1, 0.8, 0, 0, 0 %; power 2.1 to 2.6 W throughout; quiet: 0 % falls at every evaluation.  The swing is 0 to 4
  robots of 128 (the laptop run with 16 robots swung between 3 % and 27 %).
- The checkpoint that counts (rule 6: best in-run evaluation; 0 % falls and the lowest power among those) is update
  80.  Scored again with other seeds: `Full Body/rl/stand/evaluate.py --policy <deploy.bin> -B 128 --seed 7|8|9`
  gives 1.6, 0, 1.6 % falls (4 of 384 episodes), 2.2 to 2.3 W; the end of the run (update 155) 0, 0.8, 0.8 % (2 of
  384), 2.4 to 2.5 W.  The difference between the two is inside the noise of 384 episodes.  PPO's row was scored
  with one seed of 128 robots; score it with the same three seeds before the thesis table is final.
- To watch: in the quiet evaluation the tilt grows over the run (0.08° at update 40, 0.62° at update 150) and the
  power from 0.9 to 1.2 W: the late policy stands with a small lean.  It does not cost falls; if it shows in the
  other seeds too, look at term 2's pose weight and at the shrinking step (`--delta-final`).
- Two more seeds: the same Kaggle command with `TRAIN_ARGS="--seed 2"` and `"--seed 3"`, other run names
  (`grad-stand-s2`, `grad-stand-s3`).
- Done when: three seeds in the table.  Target: falls with pushes near 0 %, power near 2 W, no swing back between
  evaluations (the laptop run swung between 3 % and 27 %).

### W2: the walk task, once, for both plants  **Done 17 Nov.**  Repeat: `steps/W2.sh`
- **Where the task is written:** `Full Body/rl/walk/task.py`, and nowhere else.  Eric's walk task (the "base"
  variant `walk_v9` was trained on): 32 of his 33 reward terms with his weights and formulas, his command drawing
  and the 2 m/s^2 reference ramp, plus the pH plant's electrical power (`energy`, -0.002 per watt).  Left out:
  `self_collisions` (the pH plant has no contact between the robot's own bodies).  From his work: rewards and
  commands only.  Contract: the firmware's sprint agent, 75 inputs -> 22 targets (`rl/walk/contract.py`).
- **C++ plant:** `Full Body/rl/walk/env.py` (`WalkEnv`, an rsl-rl environment: start pose, gain errors 0.6-1.3,
  encoder bias, delays 10-70 ms, sensor noise, pushes; an episode ends at the firmware's tilt abort, a body other
  than a foot on the ground, or the trunk below 0.25 m).  Scoring: `rl/walk/evaluate.py` (per command: falls, speed
  error, power, joule per metre; `WalkEval` for training loops), `rl/walk/eval_mujoco.py` (the same file in MuJoCo).
- **Differentiable plant:** `mbrl/grad/walk_env.py` (`GradWalk`).  `mbrl/grad/env.py` now has a base class
  `GradEnv` (plant, firmware chain, delays, the state of a control step by name); `GradStand` is unchanged: a
  fixed-seed window gives bit-identical gradients before and after.  The learner's `window` runs on `GradWalk`
  as it is (`env.between_steps()` redraws delays and advances the commands, outside the derivative).
- **Which terms a gradient learner gets:** all but the two that only count events (`air_time`, `terminated`;
  `rl/walk/task.py: COUNTS`).  `GradWalk.step` returns the reward of the terms with a derivative; the full reward
  is in its log.  The two counting terms reach the learner only through the learned value of W3.
- **Results** (`steps/W2.sh`, 25 min on one core):
  - Signals against MuJoCo on a random moving robot: centre of mass, angular momentum, foot position and velocity
    agree to 1e-8.  Single reward terms agree with Eric's formulas on hand-made inputs.
  - Eric's `walk_v9` in `WalkEnv` (16 robots, 10 s per command, quiet), a check of the environment: no falls on
    stand, 0.5 and 1.0 m/s forward, 0.5 m/s back, 0.5 m/s sideways, a turn of 1 rad/s; speed error 0.05-0.10 m/s;
    105 W at 0.5 m/s, 182 W at 1.0 m/s (standing: 7 W).  In MuJoCo (`eval_mujoco.py`): the same six commands, no
    falls, speed error 0.00-0.07 m/s.
  - Both plants pay the same: `walk_v9`'s actions on the C++ plant and on the Python plant for 0.8 s (first steps
    included): every reward term of every step within 1e-5, the plant states within 1e-14.
  - **The derivative while walking** (`python -m mbrl.grad.check --task walk --steps 5 20 50`: `walk_v9` walks for
    2 s at 0.5 m/s, its next K actions are replayed open loop, cost = minus the differentiable rewards, autograd
    against finite differences):

    | window | foot landings and lift-offs inside | worst relative error | gradient norm | first / last control step |
    |---|---|---|---|---|
    | 5 control steps (0.1 s) | 1 | 3.8e-8 | 0.12 | 0.081 / 0.003 |
    | 20 (0.4 s) | 3 | 3.1e-7 | 0.29 | 0.119 / 0.002 |
    | 50 (1.0 s) | 8 | 5.6e-7 | 0.73 | 0.207 / 0.007 |

    The derivative is right through repeated steps, and it does not blow up: its norm grows about as the window
    does (the cost grows the same way: 0.82, 3.5, 8.8).  So the window length for walking can be chosen for
    speed and memory, not out of fear of exploding gradients, at least up to 1 s around a working gait.  Not
    covered: a falling or stumbling robot (far from a gait), and the ball.
  - The learner's window on the walk task with noise, pushes, changing delays, gain errors and a request that
    changes inside the window: step by step = one graph to 1e-8 (`tests/test_grad_walk.py`).
- **Found on the way: which joint velocity an environment reads.**  The two plants agree in state to 1e-15, but the
  C++ plant reports the velocity of the last plant step (nu with p = M(q_n) nu), and `GradStand` reads
  M(q_{n+1})^-1 p after it.  They differ by up to 2e-2 rad/s while walking.  The Python plant now reports the
  step's velocity too (`StepInfo.extra["nu"]`), and both walk environments use it.  `GradStand` was left as it is,
  so the standing result (W1) stays reproducible; its policy was scored on the C++ plant anyway.
- **Not done here, by plan:** training (W5), Eric's top-speed curriculum gate (`WalkTask.top_speed` and
  `command_scale` are the knobs the training will turn), a PPO training script for the comparison row (W5),
  randomised masses / ground stiffness / friction (the pH plant's own parameters are used).
- `GradWalk.terminal_cost` is a place holder (the end state's tracking and posture terms, charged for 1 s more);
  W3 replaces it.

### W3: a learned value in place of the fixed term 2  **Done 19 Nov: yes, with learned weights.**  Repeat: `steps/W3.sh check | laptop | launch | score`
- **Why:** term 2 of the professor's formula charges the state at the end of a window against a goal state x_T.
  Standing has one (upright, home pose, at rest); walking has none.  So term 2 becomes **-V(x(T))**: minus the reward
  the robot can still expect from x(T) on, learned from what the robots really get.  The learner maximises
  sum_t gamma^t r_t + gamma^K V(x(T)), with the derivative through the pH model into V as before into the fixed
  term 2.  (The same split as SHAC, Xu et al. 2022: model gradients inside the window, a learned value beyond it.)
- **What V learns from** (`--value-reward`): `smooth` (default) = the objective the learner differentiates (the
  reward terms with a derivative and the fall charge).  `full` = the full reward, including the terms without a
  derivative (standing: alive, feet_grounded, undesired_contacts, termination; walking: air_time, terminated;
  `GradStand.step` and `GradWalk.step` report it as `info["reward_all"]`).  `full` was the first plan and failed
  (attempt 1 below).
- **How:** `mbrl/grad/value.py`.  TD(lambda) returns of every window (lambda 0.95, gamma 0.99), bootstrapped with a
  slowly following copy of V (Polyak, tau 0.2); the copy is also what the learner differentiates at the window's
  end.  V (128-128, as SHAC's critics) is fitted on the last 4 windows (5 epochs, 4 minibatches, Adam 5e-4).
  Input: the critic's observation (standing 78: the clean observation, trunk velocity, height error, feet down;
  walking 90: Eric's critic), computed from the state so the derivative reaches q and p.  V learns from the first
  update on, but the window's end keeps the fixed term 2 for 20 updates (`--value-start`) and then goes over to -V
  within 10 (`--value-ramp`).  Falls inside the window are charged the fixed term 2 as before, so the only change
  against W1 is the window's end.  `python -m mbrl.grad.train ... --term2 value`.
- **Checks** (`tests/test_grad_value.py`): the TD(lambda) targets by hand; V learns a function of its input; the
  full reward adds exactly the counting terms; the window's derivative through V equals one graph over the window
  (standing with the full V and during the hand-over, walking with noise, pushes, delays and changing requests);
  the fixed term 2 is unchanged (a fixed-seed standing window gives bit-identical gradients).
- **Done when:** standing learned with -V is as good as with the fixed term 2.
- **Attempt 1 (17 Nov, laptop, failed): V from the full reward, 512-256-128, hand-over from the first update.**
  The settings of 16 Nov's fixed-term-2 laptop run (16 robots, window 20, seed 1, 40 min, one core;
  `logs/grad-stand-value-full-laptop`).  In-run evaluations, falls with pushes (64 robots):

  | | 10 min | 20 min | 30 min | 40 min |
  |---|---|---|---|---|
  | fixed term 2 (16 Nov) | 26.6 % | 3.1 % | 26.6 % | 4.7 % |
  | learned V, attempt 1 | 57.8 % | 100 % | 100 % | 100 % |

  It went wrong during the hand-over, before any robot fell: at updates 4-6 (V at 30-50 % weight) the power rose
  from 2 W to 7 W (fixed run: 2 W), while V's explained variance was still below zero: the learner followed the
  slope of a V that had not learned anything.  The deeper reason: while the robots stand, nobody falls, so the
  counting terms (alive, feet, termination) are the same for every state; a V of the full reward is then a
  constant plus fitting noise, and its slope carries nothing.  The action gradient grew 40-fold (median 0.08
  against 0.002) and, normalised per update, followed that slope.  Changes for attempt 2: V learns the
  differentiated objective (whose terms differ from state to state), a smaller V (128-128), and V only used after
  20 updates of learning.
- **Attempt 2 (17 Nov, laptop, failed): V from the differentiated objective, 128-128, used after 20 updates**
  (`logs/grad-stand-value-net-laptop`): 100 % falls at 10, 20, 30 and 40 min.  While V only learned (updates 1-20)
  the robots stood (0 falls); as soon as V was used the power rose (12 W at half weight, 54 W at 0.8) and they
  fell.  Two measurements on windows from update 24
  (`python -m mbrl.grad.term2_diag logs/grad-stand-value-net-laptop/model_24.pt`):
  - The critic's input contains the last action, so V's raw slope in that input goes straight into the last
    actions: the gradient of the last control step was 300 times the fixed run's (2.6e-2 against 8e-5).
  - At the end states, V's slope in the pH state (q, p) is unrelated to the fixed term 2's (cosine about 0,
    -0.15 to +0.08 over 48 robots) and 10-20 times larger.  The resulting action gradient was anti-aligned with the
    fixed one (cosine -0.5 to -0.7).
  A network fitted to returns gets the values right on average, but nothing makes its slope right; with 16 robots
  near one pose there is far too little variation to learn a slope, and the method lives on exactly that slope.
- **Attempt 3: term 2 with learned weights** (`--term2 weights`, `value.WeightLearner`):
  term 2 = sum_k s_k c_k(x(T)), V(x) = b - sum_k s_k c_k(x).  c_k are the costs the end state causes
  (`env.state_costs()`: the running cost's terms that read only the state, per second; for standing also the
  kinetic energy), s_k >= 0 "how many seconds each will last", b the level, learned from the same TD(lambda) returns
  (Adam 1e-2, the weights kept >= 0, the same target copy).  Standing's fixed term 2 is this sum with
  s = (1, 1, 1, 1 per J, 0, 0, 0), so the learner starts exactly there (tested: the same action gradients to
  1e-10) and needs no hand-over; walking has no goal state but has these costs (its STATE_TERMS).  The slope is
  always a positive mix of physical slopes, and 7 numbers (standing) or 14 (walking) are learned instead of 27 000.
- **Attempt 3 on the laptop (17 Nov): it works.**  The same settings as the fixed run and attempts 1 and 2 (16
  robots, window 20, seed 1, 40 min, one core; `logs/grad-stand-weights-laptop`; repeat: `steps/W3.sh laptop`).
  In-run evaluations (64 robots):

  | | 10 min | 20 min | 30 min | 40 min |
  |---|---|---|---|---|
  | fixed term 2 (16 Nov), falls with pushes | 26.6 % | 3.1 % | 26.6 % | 4.7 % |
  | fixed term 2, power with pushes | 3.5 W | 2.5 W | 6.1 W | 2.8 W |
  | learned weights, falls with pushes | 1.6 % | 0 % | 3.1 % | 1.6 % |
  | learned weights, power with pushes | 2.8 W | 3.1 W | 3.4 W | 3.4 W |

  Quiet: 0 % falls at every evaluation (fixed run: 3.1 % once, at 30 min).  The swing between evaluations is gone.
  Scored as W1 (`python -m mbrl.grad.score <run folder>`: the best in-run checkpoint, 3 seeds x 128 robots with
  pushes and noise, once quiet, 20 s in MuJoCo):

  | term 2 | checkpoint | falls with pushes | power | tilt | quiet | MuJoCo 20 s, quiet / push 0.3 m/s |
  |---|---|---|---|---|---|---|
  | fixed (16 Nov) | update 41 | 5.2 % (20 of 384) | 2.6 W | 0.74° | 0 %, 0.8 W | stands, 0.7° / 1.2° |
  | learned weights | update 48 | 1.3 % (5 of 384) | 3.1 W | 0.64° | 0 %, 0.9 W | stands, 0.5° / 1.3° |

  - **What the returns taught** (weights at update 48, seconds): orientation 3.8, base height 1.8, pose
    (similar_to_default) 2.1, kinetic energy 1.2 per J, trunk speed 0.8, trunk turning 1.2, joint speed 0.15.  A
    tilt or a wrong pose at the window's end lasts 2-4 s, not the 1 s the fixed term 2 assumed, and the trunk's
    speed counts (weight 0 in the fixed term 2).  By update 88 they had settled (3.9, 1.7, 2.5, 1.2, 0.8, 1.3, 0.06).
  - **The price: 0.5 W more.**  The energy is a cost of the actions, not of the state, so it is not in term 2; when
    term 2's weights grow, posture weighs more against energy and the robot stands a little stiffer.  Training was
    less calm than in the fixed run: training power 1.6-5.8 W (fixed 1.8-3.9 W), training falls in 7 of 88 updates
    (fixed 3 of 74), mean action gradient 0.0034 (fixed 0.0020).
  - **The diagnosis that found attempt 2's fault, on this run**
    (`python -m mbrl.grad.term2_diag logs/grad-stand-weights-laptop/model_48.pt`, 2 min):

    | | attempt 2 (network) | attempt 3 (learned weights) |
    |---|---|---|
    | direction of the action gradient against the fixed term 2's (cosine) | -0.5 to -0.7 | +0.97 |
    | slope in (q, p) at the end states against the fixed term 2's (cosine per robot, median) | about 0 | +0.95 to +0.97 (worst robot +0.83) |
    | size of that slope against the fixed term 2's | 10-20 times | 2 times (the weights grew) |
    | gradient of the last control step against the fixed term 2's | 300 times | the same |

  - Not shown by the laptop: one training seed of 16 robots each, and laptop runs swing (the fixed run between 3 %
    and 27 %).  So the Kaggle run below decides.
- **The Kaggle comparison (18 Nov): `grad-stand-weights`**, the settings of `grad-stand-v1c` (1024 robots, window 20,
  6 h, seed 1) with only `--term2 weights` added (repeat: `steps/W3.sh launch`, `fetch`, `score`).  164 updates in
  5.6 h (123 s each; v1c 129 s), 3.36 M environment steps.  In-run evaluations (128 robots, every 10 updates):

  | update | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 | 110 | 120 | 130 | 140 | 150 | 160 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | fixed term 2 (v1c), falls % | 5.5 | 2.3 | 0.8 | 0.8 | 3.1 | 2.3 | 0 | 0 | 1.6 | 0 | 3.1 | 0.8 | 0 | 0 | 0 | |
  | learned weights, falls % | 5.5 | 0.8 | 3.1 | 0 | 0.8 | 0.8 | 1.6 | 0 | 3.1 | 0 | 0 | 0.8 | 0.8 | 0.8 | 0 | 0 |
  | fixed term 2 (v1c), W | 2.65 | 2.19 | 2.06 | 2.16 | 2.42 | 2.36 | 2.32 | 2.17 | 2.38 | 2.26 | 2.52 | 2.44 | 2.46 | 2.51 | 2.36 | |
  | learned weights, W | 2.66 | 2.25 | 2.44 | 2.33 | 2.56 | 2.67 | 2.87 | 2.90 | 3.19 | 3.10 | 3.19 | 3.42 | 3.48 | 3.42 | 3.39 | 3.48 |

  Quiet: 0 % falls at every evaluation in both runs; the quiet tilt stays at 0.04-0.21° with learned weights
  (v1c: it grows from 0.08° to 0.62°, the lean noted under W1).  Scored as W1 (3 seeds x 128 robots with pushes
  and noise, once quiet, 20 s in MuJoCo):

  | term 2 | checkpoint | steps, time | falls with pushes | power | tilt | quiet | MuJoCo 20 s, quiet / push 0.3 m/s |
  |---|---|---|---|---|---|---|---|
  | fixed (`grad-stand-v1c`) | update 80 (best) | 1.64 M, 2.9 h | 1.0 % (4 of 384) | 2.3 W | 0.46° | 0 %, 1.1 W, 0.15° | stands, 0.7° / 1.6° |
  | fixed | update 155 (end) | 3.17 M, 5.6 h | 0.5 % (2 of 384) | 2.4 W | 0.54° | 0 %, 1.2 W, 0.59° | stands, 0.9° / 1.4° |
  | learned weights (`grad-stand-weights`) | update 40 (best) | 0.82 M, 1.4 h | 0.5 % (2 of 384) | 2.4 W | 0.30° | 0 %, 1.0 W, 0.22° | stands, 0.1° / 0.7° |
  | learned weights | update 164 (end) | 3.36 M, 5.6 h | 0.5 % (2 of 384) | 3.5 W | 0.32° | 0 %, 1.4 W, 0.06° | stands, 0.4° / 0.7° |

  - **Answer to W3: yes.**  The learned term 2 reaches the fixed term 2's quality (falls and power the same within
    the noise of 384 episodes: 2 against 4 falls, 2.4 against 2.3 W) with half the environment steps and half the
    time, and the robot stands straighter on both plants (tilt 0.30° against 0.46° on the pH plant, a third to a
    half of v1c's tilt in MuJoCo).  One training seed each.
  - **The weights do not settle** (seconds, at update 40 / 80 / 164): orientation 6.3 / 7.7 / 8.7, base height
    0.01 / 0 / 0, pose 2.9 / 3.8 / 5.9, kinetic energy 1.2 / 1.5 / 1.3 per J, trunk speed 0.8 / 1.2 / 1.6, trunk
    turning 0.7 / 0.05 / 0.01, joint speed 0 / 0.3 / 0.5.  The posture weights keep growing, and the power grows
    with them (2.3 W at update 40, 3.5 W at the end), most likely because the energy is a cost of the actions and
    not in term 2, so posture weighs more and more against it.  Falls do not change.  The checkpoint rule (best
    in-run evaluation) picks an early checkpoint, so standing's result does not suffer; the long walking runs (W5)
    need a limit on the weights or the energy in term 2.  Decide there, on walking's own numbers.
- **Open, the user's decision, at the end (19 Nov: walking first):** which term 2 W1's seeds 2 and 3 use.
  Recommended: the learned weights (`--term2 weights`), the term 2 walking needs anyway, so that standing and walking
  are one method; `grad-stand-weights` is then their seed 1, and `grad-stand-v1c` stays as the comparison row "term
  2 written by hand".

### W4: speed for walking  **Moved after W5 (19 Nov).**
- Why moved: a walking window costs about as much as a standing one (19 Nov, laptop, 16 robots x 20 control steps,
  one core: walking 28.9 s, standing 26.2 s; forward 7.0 / 6.8 s, backward 22.0 / 19.4 s).  So a 6 h Kaggle run gives
  about 3 M walking steps, as many as standing had.  W4 comes when walking shows it needs more than that.
- In the gradient plant only: the 8 sole contact points instead of 128 (most of the 14 MB per robot is contact
  data) and 2 ms plant steps.  Measure with `TASK=grad-bench kaggle/run.sh HOURS=1`.
- Derivative code in the C++/CUDA plant (`Full Body/ph_k1_cuda/include/ph_k1/step.hpp`, about 800 lines to
  differentiate by hand and to test against the Python plant) only if walking is too slow after this.

### W5: walk, then run  **Done 20 Nov: no, it does not step (cause examined in W5b).**  Repeat: `steps/W5.sh check | laptop | launch | fetch | score`
- **What:** walking from scratch with the thesis method: `python -m mbrl.grad.train --task walk` (Kaggle:
  `TASK=grad-walk`).  The task of W2 (`Full Body/rl/walk/task.py`, Eric's rewards and requests) on the
  differentiable plant (`mbrl/grad/walk_env.py`), in the world of the scoring environment (`rl/walk/env.py
  WalkConfig`: sensor noise, pushes 0.5 m/s, delays 10-70 ms, gain errors 0.6-1.3, encoder bias, start-pose noise,
  random heading, 20 s episodes).  The policy is the firmware's sprint agent (75 -> 22; the robot loads the same
  `.bin` reader as for standing: `K1_Deploy/firmware/src/agents/sprint.cpp`, `policy_mlp.h`), from a random network.
  Term 2: learned weights (W3), starting from the 14 state terms at 1 s each.
- **Curriculum** (`train.py: next_command_scale`): every request x `command_scale`, 0.25 at the start (forward up
  to 0.5 m/s, back 0.375, sideways 0.25 m/s, turns 0.5 rad/s); one step of 0.05 up when, over the last 20 updates, at
  most 0.2 % of the robot-windows ended in a fall (about 1 robot in 10 per 20 s) and the speed error was at most 30 %
  of the requested speed; up to 1 (Eric's ranges, sprints up to 2 m/s).  It never goes down; it is kept in the
  checkpoint.  Running above 2 m/s (`top_speed`) is a later step.
- **What the W&B run shows** (project "Akos pH MB"; also in the run folder: `metrics.jsonl`, `figures/`, `videos/`).
  To plot against episodes instead of updates, choose `Train/episodes` as a panel's x-axis; against samples:
  `Cost/executed_steps`.
  - Every 20 min, with the evaluation:
    - `Video/walk`: one quiet 10 s episode on the pH plant (stand 2 s, forward 0.5 m/s 4 s, turn 0.5 rad/s 2 s,
      sideways 0.3 m/s 2 s), the request as a green arrow (`Full Body/rl/walk/video.py`).
    - `Energy/hamiltonian`, the figure of the same episode (`rl/walk/hamiltonian.py`): (a) the tanks of H and the
      closed-loop H_d = H + sum 1/2 kp (theta - target)^2 (the motor law's PD is a spring to the policy's target);
      (b) the ledger, supplied = heat + change of H + not booked; (c, d) H_d over (angle, momentum) of the left knee
      and hip with the episode's path (a gait shows as a closed orbit, standing as a point; after Sprangers et al.,
      Fig. 3b).
    - `Gradient/through_window` (`mbrl/grad/figures.py`): (a) how strongly each control step's action is pushed by
      the derivative; (b) the co-state, the cost's derivative carried back through the pH model, for positions,
      momenta and coil fluxes, from the window's end (term 2) to its start.
    - `Eval/*`: falls, time standing, speed error, yaw error, power, averaged over the six fixed commands;
      `EvalCommand/<command>/*` per command.  This is the progress measure.
  - Every update:
    - `Gradient/*`: the action gradient at the window's first, middle and last step; the co-state's size for q, p
      and the coil fluxes at the window's end and start.
    - `Grad/*`: its median, max/median, the share of robots clipped, term 2, the objective, falls per window, seconds.
    - `Energy/*` over the training robots: power from the battery (`supplied_W`) and the heat in coils, gearboxes,
      contacts and end stops; the tanks of H at the window's end (`H_kinetic_J`, `H_gravity_J`, ..., `H_total_J`),
      the PD's spring energy and H_d; `unbooked_W` and `unbooked_share` (see the next bullet).
    - `Reward/*`: each of Eric's terms per robot and control step as it enters the reward; `(differentiated)` is
      what the learner differentiates, `(full reward)` adds air time and the fall penalty.
    - `Walk/*`: speed error and requested speed, the share of feet in the air (steps or shuffles), how fast the feet
      on the ground slide.  `Curriculum/command_scale`.
    - `Value/*`: the learned term-2 weights `s_<term>` (14), the level, the fit, the explained variance.
    - `Learn/*`: the network's fit, the step size, how far its actions moved (`action_change`) and which share of
      the asked step it made (`step_taken`).
    - `Train/*`: episodes and falls so far, the length of the episodes that ended, the share of them that fell.
- **Found 19 Nov: the energy ledger does not book the integrator's damping.**  Over an episode the plant's ledger
  should close, supplied = heat + change of H.  It does when the joint targets move smoothly (0.014 J of 12.6 J);
  with targets that jump every 20 ms it does not (small random targets: 2.9 J of 16 J missing; large: 24 J of 68 J).
  The gap is the damping of the plant's implicit Euler step (`Full Body/ph_k1/plant.py`): 1/2 L (i1 - i0)^2 in the
  coils and 1/2 dnu^T M dnu in the mechanics per 1 ms step, summed over the same episode 2.867 and 23.856 J against
  gaps of 2.869 and 23.862 J.  A real motor turns this into heat in a current transient faster than the step; the
  discrete plant only ever removes energy (it stays passive).  Logged as `Energy/unbooked_W` (robots not pushed in
  the window; pushes add energy outside the ledger): it should fall as the policy's commands get smoother.  Booking
  it in the ledger (both plants) is a small plant change for later.
- **In-run evaluation** (`Full Body/rl/walk/evaluate.py WalkEval`, every 20 min, 64 robots on Kaggle): the six fixed
  commands of W2 (stand, forward 0.5 and 1.0 m/s, back 0.5, sideways 0.5, turn 1 rad/s) with pushes and noise,
  the same at every evaluation, whatever the curriculum asks.  Every evaluated update is saved.
- **Scoring** (`python -m mbrl.grad.score <run>`, the task from the run's `config.json`): the checkpoint with the
  fewest falls over the six commands, then the lowest speed error; per command 3 seeds x 128 robots on the pH plant
  with pushes and noise, once quiet, and 20 s per command in MuJoCo, quiet and pushed.  Row for the table: falls over
  all six commands; speed error, power and energy per metre at 0.5 m/s.
- **Checks** (`steps/W5.sh check`): `tests/test_grad_train.py`: 2 updates of walking write the file the robot
  loads (its forward pass = the network); the curriculum rises, stops at 1, never goes down, survives a resume;
  standing gives the numbers of the trainer before W5 (bit-identical, recorded with commit `e91df64`).  Full Body
  `rl/walk/tests`: the sprint file round trip; `WalkEval` reports every command.  `GradWalk` now clips actions at
  +-20 as `WalkEnv` does.
- **Done when (first run):** it walks 0.5 m/s on the pH plant with pushes (few falls, speed error at most
  0.15 m/s), stands when asked, and walks 20 s in MuJoCo.  Comparison row: Eric's `walk_v9` (the team's PPO, MuJoCo;
  W2): no falls on the six commands, speed error 0.05-0.10 m/s, 105 W at 0.5 m/s.  A failure (shuffling instead of
  stepping: air time has no derivative; or falling) is reported as found and examined before anything is changed.
- **Smoke run on the laptop (19 Nov, `steps/W5.sh laptop`: 16 robots, 30 min, one core, seed 1; not a result):**
  everything runs end to end (training, curriculum, evaluation every 10 min, export, scoring).  61 updates, 19 520
  steps.  In the six-command evaluation the robots stay up longer: 1.5 s on average at update 21, 2.5 s at 42,
  3.5 s at 61 (nearly all still fall under pushes).  Scored once (update 42, 1 seed x 16 robots): it stands 20 s in
  MuJoCo and on the quiet pH plant when asked to stand, but does not walk yet (speed 0 on "forward 0.5"); feet in
  the air 1-3 % of the time (no steps yet), training falls 12-20 % of the robots per window, 40-55 W.  Harder than
  standing at the start: the sprint's home pose tips over by itself within 1.6 s (standing's home pose stood), so the
  network must learn to balance first.  The curriculum stayed at x 0.25.
- **Result of `grad-walk-v1` (Kaggle T4, 19 Nov 19:54 -> 20 Nov 02:14, 1024 robots, seed 1; 154 updates, 3.15 M
  environment steps, 365.6 min; `outputs/kaggle/grad-walk-v1/`, W&B run `nudz8axd`; tag `W5`): it does not walk,
  and with the walk world's disturbances it does not stand either.**
  - The curriculum never left x 0.25.  It needs at most 0.2 % of the robot-windows to end in a fall over 20 updates;
    the run had 18 % (updates 1-25), 10-11 % (51-100) and 12-14 % at the end.  The speed error stayed at the
    requested speed (0.25 against 0.23 m/s): the robots did not move toward the request.
  - No steps: feet in the air 1.6 % of the time at the start, 3.4 % at the end; the feet on the floor slid more
    (0.04 -> 0.06 m/s).  Power from the battery 48 -> 60 W (Eric's walk_v9: 7 W standing, 105 W at 0.5 m/s).
  - In-run evaluations (15, 64 robots per command, 10 s, pushes and noise): 92-98 % fell over the six commands at
    every one.  Asked to stand: 73-91 % fell (5.1-6.6 s up on average); forward 0.5 m/s: 89-100 % fell, speed error
    0.44-0.59 m/s.  Best around updates 59-79, then worse (time up 3.3 -> 2.2 s over the commands, power when
    standing 41 -> 57 W).  The video (`videos/walk_00059.mp4`): a crouched stand in place, no step, tipping over.
  - Term 2's learned weights kept drifting: at the end upright 18.3 s, angular momentum 9.5, joint limits 8.6, feet
    parallel 4.2; the speed tracking's weight fell from 1 to 0.32.
  - Scored (`steps/W5.sh score`, update 59 by the rule, 1.21 M steps, 1.9 h; log
    `outputs/steps/W5-score/20261120-0216_7b0e185.log`).  pH plant with pushes and noise, 3 seeds x 128 robots, 10 s
    per command: stand 298 of 384 fell (47 W), forward 0.5 359 (speed error 0.44 m/s, 56 W), forward 1.0 378, back
    384, sideways 384, turn 344.  Quiet: stand, forward 0.5 and turn without a fall but without moving; forward
    1.0, back and sideways all fell.  MuJoCo, 20 s per command, quiet and pushed (0.3 m/s): stands still on stand,
    forward 0.5 and 1.0 and turn (speed 0), falls on back (3.0 s) and sideways (1.2 s).

    | Run | Falls (6 commands, pushes) | Speed error at 0.5 m/s | Power at 0.5 m/s | Energy per metre |
    |---|---|---|---|---|
    | grad-walk-v1, update 59 (1.21 M environment steps, 1.9 h; seeds 7, 8, 9 x 128 robots x 6 commands) | 93.2 % | 0.44 m/s | 56 W | 214 J/m |
    | Eric's walk_v9 (the team's PPO, MuJoCo; W2, quiet, 16 robots) | 0 % | 0.05-0.10 m/s | 105 W | |

  - Examined (W5b): (1) No step.  `air_time`, Eric's only term that pays for lifting a foot, counts yes/no events,
    so its derivative is zero, and it reached the learner in no way: it is left out of the window, not in V's
    targets (`--value-reward smooth`), and the learned term 2 (14 state costs, no foot signals) cannot express it.
    Following the learner's own derivative from standing, the robot leans forward instead of stepping (W5b, `lift`
    without the stand-in).  (2) It does not stand either under the walk world's disturbances, while W1's stand task
    reached about 1 % falls: the sprint's home pose tips over by itself, the walk world is harsher (gain errors
    0.6-1.3, delays 10-70 ms, encoder bias), and the walk reward pays mostly for tracking (the tracking terms are the
    largest part of the reward at the end; `terminated` costs 0.013 per robot and step).
- Other rows to compare with later: PPO on pH, the planner.

### W5b: a smooth stand-in for `air_time`  **Examined 20 Nov on the laptop: it gives the slope and a weight shift, not yet steps.**  Repeat: `steps/W5b.sh check | gait | lift | laptop | launch | fetch | score`
- **Why:** W5 took no steps.  In Eric's reward, `air_time` (+0.1) is the only term that pays for lifting a foot.  It
  counts yes/no events (a foot's air clock between 0.05 and 0.5 s), so its derivative is zero, and in W5 it reached
  the learner in no way: left out of the window (`COUNTS`), not in V's targets (`--value-reward smooth`), and not
  expressible by the learned term 2 (14 state costs, no foot signals).  Every other foot term is a penalty, so in
  the learner's eyes lifting a foot only costs.
- **What, first version** (the user agreed on 19 Nov to change the reward for the learner as long as it keeps Eric's
  idea; Full Body `1304a8b`): `Full Body rl/walk/task.py`, optional term `air_time_smooth` (`STAND_INS`, off by default).  Per foot
  `unload = clamp(1 - ground force / half the weight, 0, 1)`; the ground force comes from the plant's contact spring,
  so it has a derivative.  Paid while the foot has been nearly without weight (under a quarter of the robot) for less
  than 0.5 s, and only for requests above 0.2: Eric's limit and gate.  The limit runs on a clock of its own because
  Eric's air clock stops while a foot touches, and a foot resting on the floor without weight would pay for ever
  (standing on one leg).  `train.py --air-time-smooth W` adds it with weight W to what is differentiated; the logged
  full reward and the scoring stay Eric's.  With W = 0 the trainer is bit-identical to W5 (test with the numbers of
  `7b0e185`).  Not the foot height: it is held at 0 while the foot is down, so no slope there.  Not a softened clock:
  it would pay only after 0.05 s in the air, so a standing robot would see no slope either.
- **On a real gait** (`steps/W5b.sh gait`; Eric's walk_v9 as a check only, 0.5 m/s on the pH plant, 4 robots, 5 s;
  it walks 0.49 m/s, feet in the air 36 %): the stand-in is 1.00 wherever Eric's count pays and 0.10 on average on
  the ground; correlation 0.75; of its total, 63 % comes where Eric's count pays, 21 % in the first 0.05 s in the
  air, 16 % while the weight comes off before a lift.  Figure `outputs/steps/W5b/air_time_on_walk_v9.png`.
- **The learner's own derivative from standing** (`steps/W5b.sh lift`: 4 robots asked for 0.3 m/s; their next
  0.8 s of actions optimised with Adam through `train.window`; figures and numbers in `outputs/steps/W5b/`):

  | Stand-in weight | Adam steps | Weight off a foot | Feet in the air | Trunk moved | Tilt | Falls |
  |---|---|---|---|---|---|---|
  | 0 (Eric's terms, as W5) | 40 | 19 % | 0 % | 0.14 m | 12° | 0 |
  | 0.1 (Eric's weight) | 40 | 19 % | 0 % | 0.15 m | 12° | 0 |
  | 0.5 | 40 | 47 % | 0 % | 0.14 m | 12° | 0 |
  | 1.0 | 40 | 79 % | 0 % | 0.15 m | 13° | 0 |
  | 1.0 | 100 | 96 % | 1 % (up to 0.10 s) | 0.18 m | 6° | 0 |

  Without the stand-in the optimiser leans forward for speed and both feet stay loaded: W5 in miniature.  The
  stand-in takes weight off a foot in proportion to its weight, with no hop, fall or crouch.  But at 0.5 and 1.0 it
  first collects it by rocking the weight from foot to foot about 12 times a second (each foot half unloaded for
  20-40 ms), exactly what Eric's 0.05 s lower bound refuses to pay for.  With 100 steps a slower rhythm appears in
  the second half of the window (a weight shift every 0.2-0.25 s, less lean) and a foot leaves the ground briefly.
- **The learner, 30 min on the laptop with weight 1.0** (`steps/W5b.sh laptop`, 16 robots, as W5's smoke run): no
  difference.  Feet in the air 0-4 % (W5: 1-3 %), speed error about the requested speed, 0-31 % of robot-windows
  fell, final evaluation 92 % falls and 3.6 s up.  The stand-in's reward stayed flat at 0.003 per robot and step:
  no more weight came off the feet.
- **Then the choice** (20 Nov night): (a) run it anyway at weight 1.0; (b) first stop paying for brief unloading (a
  smooth version of Eric's 0.05 s minimum) and check with `lift` again; (c) a gait clock.  The user chose (b), then
  (a) if the check shows real steps.  The bar, set before the check: in at least 2 of the 4 robots a foot off the
  ground for 0.1 s or more, with no fall, hop or crouch.
- **(b), the unload kept for 0.06 s** (20 Nov morning; Full Body `a875a01`, Model Based RL `eaf6f8b`):
  - **What:** the stand-in pays the smallest unload of the last 3 control steps.  That is Eric's 0.05 s minimum (his
    count pays from the 3rd step in the air), so rocking pays nothing.  A standing robot keeps its slope: the lighter
    foot of a quiet stance keeps a little weight off (0.7 % without noise), so the smallest of 3 steps still moves
    with the load.  The 0.5 s limit now counts seconds of unload (unload x dt) since the foot last carried 90 % of its
    half: a swing gets 0.5 s as before; a lasting lean on one foot (unload just under 0.5, which the first clock never
    counted) is paid for about 1 s; a quiet stance never runs out.  `WalkParams(air_time_range=(0, 0.5))` pays the
    unload of the moment again.  Tests: the rule on hand-set forces (rocking, a long lean, a quiet stance), autograd
    against finite differences through the memory of the last unloads, the step-by-step window equal to one graph.
  - **On the real gait** (as above; `outputs/steps/W5b-held/`): 1.00 where Eric's count pays, 0.02 on the ground
    (first version 0.10), correlation 0.95 (0.75); of its total, 89 % comes where Eric's count pays (63 %), 7 % in the
    first 0.05 s in the air (21 %), 4 % before a lift (16 %).
  - **The slope in W5's world** (32 robots standing in the walk world with W5's action noise, asked for 0.3 m/s, three
    windows of `train.window` one after the other): as many robots get a slope as with the first version (32, 32 and
    24 of 32; the others had fallen), median size 3.3e-3, 2.2e-3, 5.7e-4 against 2.6e-3, 1.6e-3, 4.7e-4.  What it
    pays drops from 0.040 to 0.023: the brief unloads.
  - **lift** (as above; `WEIGHTS="1 2" ITERATIONS=40|100 steps/W5b.sh lift`; `outputs/steps/W5b-held/lift-*`):

    | Stand-in weight | Adam steps | Weight off a foot | Longest off the ground | Robots with a foot up 0.1 s | Trunk moved | Tilt | Falls |
    |---|---|---|---|---|---|---|---|
    | 1 | 40 | 43 % | 0 | 0 of 4 | 0.15 m | 13° | 0 |
    | 1 | 100 | 59 % | 0 | 0 of 4 | 0.17 m | 6° | 0 |
    | 2 | 40 | 70 % | 0 | 0 of 4 | 0.14 m | 12° | 0 |
    | 2 | 100 | 86 % | 0.02 s | 0 of 4 | 0.17 m | 5° | 0 |

    The rocking is gone: the weight goes from one foot to the other every 0.2-0.25 s and stays there (robot 0 at
    weight 2 after 100 steps: for 0.1 s the left foot carries 0-25 % of its half, the right foot nearly the whole
    robot), and the robots lean less (12° → 5°).  No hop, no crouch (the trunk drops 2.4 cm at most), no fall.  But
    no foot is off the ground for more than one control step: the bar is not met.
  - **The learner, 30 min on the laptop with weight 2** (`W=2 steps/W5b.sh laptop`): as before.  The stand-in's
    reward stays flat (0.003-0.005 per robot and step after the first iterations), feet in the air 0-5 % (W5: 1-3 %),
    6-31 % of robot-windows fell, final evaluation 85 % falls and 4.0 s up.  So no Kaggle run.
  - **The same in a quiet world** (`QUIET=1 W=0|2 steps/W5b.sh laptop`: no noise, pushes or gain errors, 20 ms
    delay), to tell the two problems apart: no stepping either.  Feet in the air 0-2 % without the stand-in and 0-3 %
    with it at weight 2; 0-19 % of robot-windows fell in both; the final evaluation (always in the walk world) 100 %
    falls.  So the disturbances are not the main blocker: even without them the learner does not take up the
    stepping signal.
- **Result** (20 Nov): the missing derivative was a real gap, and the stand-in closes it: following it, weight
  comes off a foot.  Paid only for unloading kept 0.06 s, what the optimiser makes of it from standing is the first
  half of a step (a weight shift held 0.1-0.2 s, at a walking rhythm) instead of rocking; but the foot does not come
  up within 0.8 s, and the learner does not take it up in 30 min, in the walk world or in a quiet one.  No Kaggle
  run.  Next (the user, 20 Nov): a gait clock (W5c), the way Booster trains its own T1 (Booster Gym, arXiv
  2506.15132): two clock inputs (cos and sin) and a reward for stepping in time with them, the inputs at 0 while
  standing; the firmware agent drafted for Eric.  Separately, W5 did not even stand under the walk world's
  disturbances (78 % falls when asked to stand).

### W5c: a gait clock  **Examined 20 Nov on the laptop: in the stepping check a foot comes up; the learner did not step yet.**  Repeat: `steps/W5c.sh check | lift | laptop`
- **Why:** W5b gave stepping a slope; the learner spent it on shifting weight, never on a step, even in a quiet world.
  Walking needs a beat, and the learner cannot find one from Eric's rewards.  The user chose a gait clock on 20 Nov,
  the way Booster trains its own T1 (Booster Gym, arXiv 2506.15132: the actor gets `(cos 2πft, sin 2πft)`, a
  feet-swing reward pays for "stepping with both legs according to the set cycle", the cycle is 0 for standing).
- **What** (Full Body `0297d78`; Model Based RL `ac7efe0`, `077f02f`):
  - The clock runs while the request, clamped as the network sees it, asks for a move: |(vx, vy)| + |wz| above 0.2,
    Eric's `air_time` gate.  It moves on by dt / 0.46 s per control step; asked to stand, it is back at 0 and both
    inputs are 0, so every walk starts with the same foot.
  - Period and swing share are walk_v9's own on the pH plant at 0.5 m/s (79 cycles: 0.465 s, each swing 0.166 s, 36 %
    of the cycle).  walk_v9 changes its rhythm with speed: 0.37 s at 0.3 m/s (feet in the air 28 %), 0.53 s at
    0.8 m/s (47 %); the clock keeps one.
  - The network gets sin and cos of the clock after the 75 inputs (77).  The term `gait_clock`, per foot
    `swing x unload + (1 - swing) x (1 - unload)` averaged over the feet, only while the clock runs: its slope tells
    each foot at every moment to unload (its turn) or to carry.  Not Eric's (`SHAPING`): never in the scored full
    reward.
  - `GradWalk(gait_clock=W)`, `train.py --gait-clock W`, `probe_step lift --term clock` (lift.json keeps the traces).
- **The firmware, a draft for Eric** (K1_Deploy, local branch `walk-clock`, `cc16204`, not pushed): `WalkClockAgent`
  in `src/agents/sprint.cpp` (the sprint's 75 inputs, gains, decode and limits, plus the clock from a step counter;
  the clock inputs in the run log), registered as `walk_clock` in `factory.cpp`; the twin's `WalkClock`
  (`k1_deploy/agents/sprint.py`, reset at every start) and its test.  Full Body's `rl/walk/tests` check that the twin
  and the training give the same two inputs at every step, through stops and restarts.
- **The stepping check with the clock** (as W5b's; `outputs/steps/W5c/`; standing still scores 0.64 in time):

  | Term | Weight | Adam steps | In time with the clock | Weight off a foot | Longest off the ground | Robots up 0.1 s | Falls |
  |---|---|---|---|---|---|---|---|
  | kept unload (W5b) | 2 | 100 | 0.57 | 86 % | 0.02 s | 0 of 4 | 0 |
  | gait clock | 2 | 40 | 0.74 | 59 % | 0 | 0 of 4 | 0 |
  | gait clock | 2 | 100 | 0.78 | 80 % | 0 | 0 of 4 | 0 |
  | gait clock | 4 | 100 | 0.88 | 94 % | 0.12 s | 2 of 4 | 0 |

  The clock's weight 4 pulls each foot about as hard as the stand-in's 2 (the clock averages over the two feet).
  With the clock the weight shifts follow the beat (the kept unload's keep their own rhythm), the lean drops
  (17° → 6°), and at weight 4 the first lift of any check: robot 0's right foot is off the ground for 0.10 s, robot
  2's for 0.12 s, inside their windows; no hop, no fall.  The bar set before W5b's check (a foot up 0.1 s in 2 of 4
  robots, no fall, hop or crouch) is met.
- **The learner, 30 min on the laptop** (`steps/W5c.sh laptop`: weight 4, 77 inputs, the walk world, no in-run
  evaluation: it still takes 75): 49 updates, 15,680 robot-steps; feet in the air 1.3 % over the first 10 updates,
  1.5 % over the last 20 (W5: 1-3 %); the clock's reward 0.037 → 0.031 per robot and step; 131 falls.  The bar for
  Kaggle (feet in the air at least 6 % over the last 20, the clock's reward rising) is not met: no Kaggle run.
- **Result:** physics and the clock's reward can make a step: when the timing is known, the optimiser lifts a foot.
  The learner has not found it in 30 min.  Its window (0.4 s) is shorter than one gait cycle (0.46 s), so it never
  sees a whole step, and 15,680 robot-steps are 0.5 % of W5's run.  The course page has a chapter for Eric (chapter 10).
- **Ready for a real run** (20 Nov afternoon): the 77 inputs reach the in-run evaluation (C++ plant), the export (the
  walk_clock agent's file, 77 -> 22), the video and the scoring (pH plant and the firmware's loop in MuJoCo, with its
  own clock counter).  Tests: both plants give the same 77 inputs while walk_v9 walks; a clock run is evaluated,
  exported and runs in MuJoCo.  `steps/W5c.sh launch` shows the Kaggle run `grad-walk-clock-v1` (W5's settings and
  only `--gait-clock 4`); it starts only with the user's yes.  Open after it: longer windows (W4).
- **With the learner's own view** (the stepping check with a 0.4 s window and term 2 at its end, as the learner sees
  it, instead of 0.8 s; weight 4, 100 Adam steps; `outputs/steps/W5c/lift-clock-4-window20/`): the question was
  whether the short view, with term 2 charging a robot caught mid-step, pulls it back to standing.  It does not: all 4
  robots lift the right foot for 0.06-0.08 s inside its window while the left foot carries nearly the whole robot,
  then the weight goes back and the left foot unloads in its window; in time 0.86, the weight fully off a foot, feet
  in the air 9 % of the window (0.8 s view: 3 %), the trunk 0.10 m forward with 1.8° of lean (0.8 s view: 5.8°), no
  fall, no hop.  The bar (0.1 s in 2 of 4 robots) is missed by its letter, 0.08 s at most: the short view gives
  shorter lifts.  So the learner's objective points to stepping; what is missing is training.

### W6: kick
- First the derivative through the ball contact (a check like W0).  The ball's 3x3 solve in the Python plant
  (`torch.linalg.inv` in `ph_k1/plant.py`) must go through `ph_k1/solve.py`, or the T4 fails on it.
- `mbrl/grad/kick_env.py` from `Full Body/rl/kick/env.py`: kick from standing first, then walk and kick; every stage
  starts from this method's own previous stage.
- The kick's reward comes late (speed and direction of the ball after the foot hit it): longer windows and the
  learned value of W3.  This is the hardest case for the method; a negative result is possible and is reported.

## Older packages for the comparison rows (PPO on pH and the planner)

### Package 6: walking, both arms (medium + ~7 h of Kaggle per run)
- Do: `Full Body/rl/walk/{env,train,evaluate}.py`: Eric's walk task and rewards on the pH plant, ported from
  `RL_Core/rl_core/envs/k1_skill.py` and `k1_skill_cfg.py` as `rl/kick/env.py` was ported from k1_kick; the contract
  is `rl/walk/contract.py`. Snapshot/restore on the env. Check of the environment only: Eric's walk policy
  (`policies/walk_teacher.pt`) follows the commanded speed in it.
- Runs: `walk-ph-v1` (PPO from scratch) and `mb-walk-v1` (the learner of package 4).

### Package 7: the kick, both arms (medium each)
- Do: snapshot/restore on `KickEnv` (the ball and the camera's buffers included); a "from standing" setting
  (`KickConfig(stand_prob=1.0)` and no walk commands); starting a network from the arm's own walk checkpoint with
  the new inputs at zero weight (PPO: `rl/kick/train.py --init-actor` handles equal shapes only, extend it).
- Runs: kick from standing, then walk + kick, for both arms; evaluation with `rl/kick/evaluate.py`.
- The 166th input (seconds since the request) is optional (`--phase-input`); a policy that uses it needs one line in
  the firmware's kick_walk agent.
