python train.py --config-path conf --config-name config \
  +hpo=space \
  scheduler.t_initial=10 \
  early_stop.patience=2 \
  exp.folds=1 \
  checkpoint.resume=auto