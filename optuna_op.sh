#!/usr/bin/env bash
python train.py \
  +hpo=space \
  scheduler.t_initial=10 \
  early_stop.patience=2 \
  exp.folds=1 \
  checkpoint.resume=auto
