python train.py --config-path conf --config-name config -m \
  +hpo=space \
  hydra.sweeper.n_trials=50 \
  scheduler.t_initial=10 \
  early_stop.patience=2 \
  exp.folds=1
