# ddr-ml

Training pipeline for my ddr binary image classifier. convnextv2-large from timm on top
of the local dataset, stratified k-fold cv scored by AUC, an optuna sweep built directly
into train.py, and mid-epoch pause/resume so a run can be interrupted without losing
progress (which is what half of this code is doing).

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Assumes a CUDA gpu. It falls back to cpu (exp.device=cpu) but the dataloader cache and
the prefetch pipeline are tuned around a gpu, and nobody has ever waited for a run to
finish on cpu.

## Dataset

`ddr_binary_dataset/` with one folder per class, jpg/png inside:

```
ddr_binary_dataset/
├── some_class/
│   ├── img_0001.jpg
│   └── ...
└── other_class/
    └── ...
```

Class folders are sorted alphabetically into label indices. The dataset is gitignored,
so it has to be placed next to train.py. Images are resized longest-side to
`data.img_sz` (600) and randomly cropped from there.

## Training

```
python train.py
```

Everything lives in `conf/config.yaml` and gets overridden on the command line, hydra
dot syntax:

```
python train.py exp.folds=1 optim.lr=3e-5 data.img_sz=512
```

The UI is curses: `P` pauses and resumes. Pausing moves model, EMA and optimizer state
to cpu and frees VRAM while keeping the dataloader caches warm (useful on a shared
box). You can also `touch checkpoints/<hash>/PAUSE` to pause from another shell.
Ctrl+C stops after the current batch; rerunning the same command picks up where the
last epoch left off. Checkpoints are keyed by a hash of the resolved config under
`checkpoints/<hash>/` (`last.pth`, `resume_marker.json`, `result.json`), so changing
any parameter starts a fresh run and changing nothing resumes. Hydra also writes a
full config dump and `train.log` under `outputs/<date>/<time>/`.

## Hyperparameter search

The search space is `conf/hpo/space.yaml`, edit the ranges there. To sweep:

```
./optuna_op.sh
```

which is just train.py with `+hpo=space` and a shortened schedule so trials finish
fast. Trials are stored in `optunahpo.db`, so if something dies mid-trial (reboot,
oom, ctrl+c) the same command re-enqueues the last unfinished trial and resumes it
from its checkpoint. If a trial is stuck in RUNNING state in the db and that doesn't
pick it up, `python enqueue_running.py` force-enqueues its params. If the failure
rate goes over `hpo.max_failure_rate` the whole study bails out.

Watch progress with the optuna dashboard:

```
./optuna_view_dashboard.sh    # http://localhost:9090
```

## Notes

- Batch size is small (5) because the images are big; grad accumulation of 2 doubles
  it effectively. The dataset caches decoded images in worker memory with a total
  budget (`data.cache_max_gb_total`) and shrinks the cache when system ram goes over
  ~92%, which needs psutil.
- Loss is asymmetric focal (ASL) by default, paired with a class-balanced sampler,
  because this set is imbalanced. mixup/cutmix and EMA are on and in the search space.
- Default optimizer is AdamP. AdamP and channels_last don't play nice, so
  channels_last is silently disabled when adamp is selected.
