# Thesis experiments: model-based RL on the pH plant, against the team's PPO in MuJoCo

This file is the working plan of the thesis experiments: the rules, what exists, and the next work packages. Each
package has a goal, the files it touches, what to reuse, and a check that says it is done. Update the status tables
when a package is finished.

## The question and the rules

**This thesis** = the port-Hamiltonian plant as the model + model-based RL on it.  **The work it is compared with**
= how the team trains today: model-free PPO in MuJoCo (Eric Plass's k1_kick, k1_sprint, RL_Core).  The two are kept
apart everywhere: in the names, the tables and the code.

| Row | Physics | Learning | Whose | Role |
|---|---|---|---|---|
| **Thesis approach** | pH plant (Full Body, fitted to robot logs with `mbrl/fit.py`) | model-based: at every control step a planner looks a short time ahead in the pH model; a policy network and a value network learn together with it (`mbrl/mpc`) | this thesis | the result |
| **Baseline** | MuJoCo | model-free PPO (rsl-rl) | the team (Eric's policies and training) | what the thesis is measured against; never a part of the thesis approach |
| Ablation: PPO on pH | pH plant | model-free PPO (Full Body `rl/<skill>/train.py`) | this thesis | separates the two changes: is a difference due to the physics (pH instead of MuJoCo) or to the learning (planning instead of PPO)? |

In this file "model-based arm" = the thesis approach, "PPO arm" = the ablation (PPO on pH) unless MuJoCo is named.

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
| 1 | stand under pushes | firmware `stand`, 72 -> 22 (`rl/stand`) | done: `stand-ph-v1` (3000 iterations, ~295 M steps, 7 h on a T4, no falls) | open (packages 3-5) |
| 2 | walk on velocity commands, standing included | sprint family, 75 -> 22 | open (package 6) | open (package 6) |
| 3 | kick from standing | kick_walk, 165 (166 with the request clock) -> 22 | open (package 7) | open (package 7) |
| 4 | walk, stay and kick | kick_walk | open (package 7) | open (package 7) |

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

| model-based, Kaggle `mb-stand-v1`, best checkpoint (`policies/mb-stand-v1_260.bin`, 1.6 M executed + 64 M model steps, 2.2 h on a T4) | 2.3 % | 9.94 s | 14.0 W | 1.3° |
| model-based, Kaggle `mb-stand-v1`, end of the run (`policies/mb-stand-v1_708.bin`, 4.35 M executed + 174 M model steps, 6 h) | 10.2 % | 9.73 s | 24.6 W | 2.1° |

Without pushes and noise (`--quiet`): PPO 0 % falls, 1.0 W; model-based 0 % falls at 13.3 W (best) and 20.9 W (end).
In MuJoCo (`rl/stand/eval_mujoco.py --seconds 20`, quiet and `--push 0.3`): all three stand the 20 s; largest tilt
PPO 0.4° / 0.7°, model-based best 3.9° / 3.9°, model-based end 1.5° / 2.1°.

**Stage 1 as it stands (16 Nov): the model-based arm learns to stand from scratch with about 1 M executed steps
(PPO: about 14 M), but its policy is not as good as PPO's: 7 times the power, and it still falls under pushes.
It is not a result for the thesis table yet.**

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
- Status: an investigated alternative.  It becomes a method only if the `--value` run lowers the power without
  falls, and then it still needs derivative code in the C++/CUDA plant (or the Python plant on a GPU).

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
