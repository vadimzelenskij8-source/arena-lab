# arena-lab

Population-based PPO experiments in a deterministic card-battle simulator
([crforge](https://github.com/voonhous/crforge), Apache-2.0, cloned at build time).

- `train/ppo.py` — residual PPO worker (starts at the scripted policy's level, learns where to deviate)
- `train/select_best.py` — keeps the best worker of each generation by a fixed evaluation
- `train/report.py` — builds `report.html` on the `results` branch
- `.github/workflows/train.yml` — up to 20 parallel workers per generation, a new generation every 6 hours
