# enqueue_running.py
import optuna
from optuna.trial import TrialState

STORAGE = "sqlite:///optuna.db"   # must match your CLI
STUDY   = "DDR_ML"                # must match your CLI

study = optuna.create_study(storage=STORAGE, study_name=STUDY, load_if_exists=True)
running = [t for t in study.get_trials(deepcopy=False) if t.state == TrialState.RUNNING]
if not running:
    print("No RUNNING trials.")
else:
    t0 = sorted(running, key=lambda t: t.number)[0]
    study.enqueue_trial(t0.params)
    print(f"Enqueued params of Trial {t0.number}: {t0.params}")