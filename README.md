# K1 Model-Based RL: the pH plant fitted to the real robot

Part of the master thesis *Physics-informed Reinforcement Learning for Soccer Robots* (Akos Eross, HTWK Leipzig).
This repository turns the hand-built port-Hamiltonian plant of the Booster K1
([K1-Full-Body-pH](https://github.com/akuduku/K1-Full-Body-pH), submodule `external/full-body`) into a
**model fitted to robot data** and trains the RL skills against that model.

Model-based RL (Stanford CS224R, Lecture 11) in three steps, and what each step means here:

| Lecture 11 | Here |
|---|---|
| 1. collect data `D` with some policy π₀ | Logs of the real K1 (q, gyro, accelerometer, commanded targets at 500 Hz), recorded with the policies that already work on it |
| 2. learn a model `f(s,a) ≈ s'` | The model is the pH plant (physics: "approximately known model", slide 10). Its unknown parameters are **fitted to the logs** (system identification with CEM). An optional **learned residual in pH form** covers what the physics misses |
| 3. use the model to get a policy | PPO in the fitted plant (Full Body's `rl/stand`, `rl/getup`). Every robot draws one **member of the parameter ensemble** (the CEM elites) at each reset, so the policy cannot exploit one wrong model (PDDM, PETS) |
| iterate: avoid distribution mismatch | Run the new policy on the robot, log it, add the log to `D`, refit, retrain |

The reward model is not learned. It comes from the plant's Hamiltonian and energy ledger.

## Layout

```
mbrl/
  paths.py      where Full Body (../Full Body, or the submodule) and the robot logs are
  params.py     the fitted parameter vector theta (24): contact, per joint group damping / torque ceiling /
                driver kp, kd factors, transport delay -> the plant's per-robot rows
  data.py       robot CSV -> 2 ms grid, smoothed joint speeds, held commands, trunk tilt (Mahony filter);
                KNOWN = the logs and the agent settings (gain preset, slew) their replay needs
  state.py      a plant robot in a logged state (stance-point leg kinematics for the trunk speed)
  replay.py     logged commands from logged states, 0.4 s windows, every (candidate, window) one robot of a batch
  fit.py        CEM system identification -> fits/<name>.json (mean, std, elites, errors before/after)
  report.py     today's plant vs fit (vs fit + residual): report.md, kstep.png, traces.png
  residual.py   learned pH residual: -d(x) theta_dot (d >= 0, passive) + bounded torque u(x), ensemble of K MLPs
  model.py      attach(plant, fit, residual): parameter ensemble + residual on any env's plant
  train.py      Full Body's rl/<task>/train.py on the fitted model (same harness, W&B, deploy.bin)
tests/          parameter rows, data, replay self-consistency, ensemble/residual attach, env wrapper
fits/           fitted models (JSON), residual/  residual ensembles (.pt)
kaggle/         Kaggle kernel train_mb.py (TASK=stand|getup kaggle/run.sh, fetch.sh); the GITHUB_TOKEN in the
                private dataset <user>/k1-secrets needs read access to this repo AND K1-Full-Body-pH
colab/          colab-train jobs: k1-stand-mb.yaml, k1-getup-mb.yaml (+ setup_gpu.sh, also used by Kaggle)
```

The plant feature this needs lives in Full Body: `plant.robot_params` (B, 49), one row of physical parameters
per robot (`ph_k1/robot_params.py`, C++ `config.hpp RP_*`). It is tested against the Python reference.

## Run

```bash
git clone --recursive https://github.com/akuduku/K1-Model-Based-RL.git     # or next to "Full Body" (used if present)
python -m pytest tests -q                                                  # ~10 s, CPU
python -m mbrl.fit --smoke                                                 # a few seconds
python -m mbrl.fit --train getup_2236,getup_front_2250 --val "" --out fits/getup-both-v1.json   # ~5 min CPU
python -m mbrl.report fits/getup-both-v1.json                              # outputs/getup-both-v1/
python -m mbrl.residual --fit fits/getup-both-v1.json --train getup_2236,getup_front_2250 --val "" \
       --out residual/getup-both-v1.pt
python -m mbrl.train stand --fit fits/getup-both-v1.json -- --device cpu --num-envs 8 --iterations 2 --no-wandb  # smoke
TASK=stand kaggle/run.sh                                                   # the real training (Kaggle T4, 12 h session)
TASK=stand kaggle/fetch.sh                                                 # wait, download the kernel output
colab-train launch --browser colab/k1-stand-mb.yaml                        # or on Colab
```

The logs are not in git: `MBRL_LOGS` points to the folder that holds `30102026/...` (default:
`MA/Karls Repo/Logs` next to this repository).

## First results (2 get-up logs of K1-13, 25 Oct 2026)

k-step prediction: logged commands from logged states, 0.4 s windows, RMSE of the joint angles in degrees.
Loss = weighted joint MSE + 0.02 · gyro MSE + tilt error.

**Generalisation test** (`fits/getup-v1.json`: fitted on the back get-up, checked on the front get-up):

| log | model | loss | head | arm | hip | knee | ankle |
|---|---|---|---|---|---|---|---|
| back (train) | today's plant | 0.223 | 11.0 | 10.1 | 7.1 | 4.7 | 5.1 |
| back (train) | fit | 0.190 | 11.3 | 8.8 | 7.0 | 4.8 | 5.2 |
| front (held out) | today's plant | 0.651 | 12.6 | 14.5 | 7.8 | 7.1 | 6.6 |
| front (held out) | fit | 0.658 | 13.5 | 13.1 | 7.8 | 7.3 | 6.9 |

**Fit on both logs** (`fits/getup-both-v1.json`, the one the trainings use): loss 0.417 → 0.313 (−25 %), front
get-up 0.651 → 0.441 (trunk tilt error 0.25 → 0.16). The fit makes the floor contact softer (k_n × 0.16) and more
slippery (μ × 0.42), adds 0.1–0.2 N m s/rad of joint damping, and finds a transport delay of 19 ms. That delay
agrees with the earlier replay of the front get-up (~20 ms).

What this says:
- The plant already predicts 0.4 s ahead far better than "nothing moves" (leg error 5° vs 22°). The fitted
  parameters improve the get-up it saw, but one get-up does not carry over to the other. This is the lecture's
  point that **data coverage matters**: the parameters that matter differ between lying on the back and on the
  front.
- The residual (`residual/`) improves the joint angles a little, but not the overall loss, from ~500 samples. It
  is ready for more data; the trainings use the physics fit alone for now.
- A replay of the plant's own synthetic log (tests) leaves ~1° of error with the true parameters. That is the
  floor set by the state estimate (smoothed joint speeds, trunk speed from the stance foot, tilt filter).
- The first ~1.9 s of both logs are excluded: the logger writes the start pose as the command while the joints
  move to the get-up start (the driver was not following the logged command then).

## Data we need (request for Eric)

The same CSV the get-up logs have (`t, q0..q21, gx gy gz, ax ay az, cmd_t, cmd0..cmd21`, 500 Hz, MJCF joint
order). If the logger can, add `dq0..dq21` and `tau0..tau21` (`tau_est` of the motor state) and the kp/kd it sent.
Please record the agent and gain set of every run (they go into `mbrl/data.py KNOWN`).

1. **Hanging robot, one joint group at a time:** chirp (0.2→5 Hz, ±0.2 rad) and steps (±0.1/0.3 rad) on hips,
   knees, ankles, arms. This separates motor, driver and delay from contact.
2. **stand-ph-v1 on the robot**, standing and under pushes (it was trained on today's plant: its log shows
   exactly where that plant is wrong).
3. **The working skills:** walk (genesis_walk), kick, goalie, dribble, a few minutes each. They give coverage of
   the states the robot really visits.
4. **More get-ups**, front and back, ~10 each.
5. **After every new policy of ours:** its own log, which goes back into the fit (the data loop).

## Open

- **Get-up A/B baseline.** The first get-up run on the fitted model (`getup-mb-v1`) also has the new reward
  curriculum (Full Body b5d0c8a: no movement costs and fixed gains until it stands up). To separate the effect of
  the fit from that of the reward fix, the same training on today's plant is needed (`getup-ph-v2`), compared on
  success rate, time to stand, battery energy per get-up and the MuJoCo check.

## Limits

- The trunk position and linear velocity are not measured. They come from leg kinematics with the lowest contact
  point as a static stance point, which is wrong in the fast phases of a get-up.
- No self-collision in the plant (arms against the trunk during get-ups).
- The torque residual is held for 20 ms and computed from the robot's state at the start of the step.
- Two logs, one robot, one day.
