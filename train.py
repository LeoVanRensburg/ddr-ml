# train.py
from __future__ import annotations

import os
import time
import warnings
import curses

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from rich import traceback
import json, hashlib
from hydra import utils as hydra_utils
import copy
import optuna
from optuna.trial import TrialState
from omegaconf import open_dict

import engine

def _param_hash_for_ckpt(cfg: DictConfig) -> str:
    # Hash the resolved config minus volatile fields, to derive a stable dir per param set.
    c = OmegaConf.to_container(cfg, resolve=True)
    # Remove Hydra runtime section and the checkpoint.dir itself to avoid recursion.
    c.pop("hydra", None)
    if "checkpoint" in c and isinstance(c["checkpoint"], dict) and "dir" in c["checkpoint"]:
        c["checkpoint"] = {k: v for k, v in c["checkpoint"].items() if k != "dir"}
    blob = json.dumps(c, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode()).hexdigest()[:12]

def _ensure_stable_checkpoint_dir(cfg: DictConfig) -> None:
    """
    For Option B. Place checkpoints/marker under a dir that depends only on the params,
    not on the Hydra run directory. This lets a re-enqueued trial pick them up.
    """
    # Base directory (project-root relative), configurable; default "checkpoints".
    base = getattr(cfg.checkpoint, "base_dir", "checkpoints")
    # Make absolute relative to original working directory (not Hydra's run dir).
    base_abs = hydra_utils.to_absolute_path(base)
    key = _param_hash_for_ckpt(cfg)
    new_dir = os.path.join(base_abs, key)
    os.makedirs(new_dir, exist_ok=True)
    cfg.checkpoint.dir = new_dir

def _setup_logger_to_hydra_run_dir():
    """Attach loguru sink to the current Hydra run directory if available."""
    if HydraConfig.initialized():
        run_dir = HydraConfig.get().runtime.output_dir
        os.makedirs(run_dir, exist_ok=True)
        logger.add(
            os.path.join(run_dir, "train.log"),
            enqueue=True,
            backtrace=True,
            diagnose=False,
        )

def _normalize_storage_uri(storage: str, base_ckpt_dir: str) -> str:
    """
    Ensure Optuna storage persists across runs.
    - If empty/None -> create sqlite DB under checkpoint.base_dir.
    - If sqlite:///relative/path.db -> absolutize relative path and ensure parent dir exists.
    - Otherwise (e.g., postgres/mysql URIs) pass through unchanged.
    """
    s = (storage or "").strip()
    if s.lower() in ("", "none", "null"):
        base_abs = os.path.abspath(hydra_utils.to_absolute_path(base_ckpt_dir))
        os.makedirs(base_abs, exist_ok=True)
        return "sqlite:///" + os.path.join(base_abs, "optuna.db")

    if s.startswith("sqlite:///"):
        raw = s[len("sqlite:///"):]
        abs_path = os.path.abspath(hydra_utils.to_absolute_path(raw))
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        return "sqlite:///" + abs_path

    return s

def _device_from_cfg(cfg: DictConfig) -> torch.device:
    want = str(cfg.exp.device).lower()
    if torch.cuda.is_available() and want.startswith("cuda"):
        return torch.device("cuda")
    return torch.device("cpu")

def _oc_update(cfg: DictConfig, dotted: str, value):
    # OmegaConf.update supports dotted keys
    OmegaConf.update(cfg, dotted, value, merge=True)

# --- suggest params from cfg.hpo.params (list of {name,type,...}) ---
def _apply_trial_to_cfg(trial: optuna.Trial, base_cfg: DictConfig) -> DictConfig:
    """
    Expects cfg.hpo.params like:
      hpo:
        params:
          - {name: "optim.lr", type: "float", low: 1e-5, high: 1e-2, log: true}
          - {name: "model.drop_rate", type: "float", low: 0.0, high: 0.5}
          - {name: "optim.weight_decay", type: "float", low: 0.0, high: 0.1}
          - {name: "scheduler.min_lr", type: "float", low: 1e-6, high: 1e-3, log: true}
          - {name: "mixup_cutmix.mixup_alpha", type: "float", low: 0.0, high: 0.4}
          - {name: "mixup_cutmix.cutmix_alpha", type: "float", low: 0.0, high: 0.4}
          - {name: "data.batch_size", type: "categorical", choices: [16, 24, 32]}
    """
    cfg = copy.deepcopy(base_cfg)
    params = getattr(cfg, "hpo", {}).get("params", [])
    for p in params:
        name = p["name"]
        ptype = p["type"].lower()
        if ptype == "float":
            v = trial.suggest_float(name, float(p["low"]), float(p["high"]), log=bool(p.get("log", False)))
        elif ptype == "int":
            v = trial.suggest_int(name, int(p["low"]), int(p["high"]),
                                  step=int(p.get("step", 1)), log=bool(p.get("log", False)))
        elif ptype == "categorical":
            v = trial.suggest_categorical(name, list(p["choices"]))
        else:
            raise ValueError(f"Unsupported hpo param type: {ptype} for {name}")
        _oc_update(cfg, name, v)
    return cfg

def _build_sampler(sconf):
    if not sconf:
        return None
    name = str(getattr(sconf, "name", "tpe")).lower()
    seed = sconf.get("seed", None)
    multivariate = bool(sconf.get("multivariate", False))
    if name in ("tpe", "tpesampler"):
        return optuna.samplers.TPESampler(seed=seed, multivariate=multivariate)
    raise ValueError(f"Unsupported sampler: {name}")

def _stop_on_failure_rate(threshold: float):
    def _cb(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        done = [t for t in study.trials if t.state in (TrialState.COMPLETE, TrialState.FAIL, TrialState.PRUNED)]
        if not done:
            return
        fails = sum(1 for t in done if t.state == TrialState.FAIL)
        if fails / max(len(done), 1) > threshold:
            study.stop()  # graceful stop
    return _cb

def run_training_nocurses(cfg: DictConfig, rethrow_errors: bool = False) -> float:
    # logging/determinism/device
    try:
        logger.remove()
    except Exception:
        pass
    _setup_logger_to_hydra_run_dir()
    logger.info("CONFIG:\n" + OmegaConf.to_yaml(cfg))

    engine.set_seed(cfg.exp.seed)
    device = _device_from_cfg(cfg)
    if cfg.exp.determinism:
        torch.use_deterministic_algorithms(True)

    # stable per-param checkpoint dir
    _ensure_stable_checkpoint_dir(cfg)

    # data plumbing (folds + paths + kornia)
    (train_loader_dummy, val_loader_dummy, folds,
     all_paths, targets, kornia_tf) = engine.build_dataloaders(cfg, device)

    # shared epoch counter ONLY for training; validation stays cache-warm across epochs
    tr_shared_epoch = engine.create_shared_epoch(0)

    # Initial loaders (first fold) for scheduler sizing
    tr0, va0 = folds[0]
    train_loader = engine._make_loader(cfg, all_paths, targets, tr0, is_train=True,  shared_epoch=tr_shared_epoch)
    val_loader   = engine._make_loader(cfg, all_paths, targets, va0, is_train=False, shared_epoch=None)

    model, ema = engine.build_model(cfg, device)
    criterion = engine.build_loss(cfg).to(device)
    optimizer = engine.build_optimizer(cfg, model)
    scheduler = engine.build_scheduler(cfg, optimizer, len(train_loader))
    mixup_fn = engine.MixupCutmixScheduler(cfg)

    start_epoch, start_fold, best_auc = engine.load_checkpoint(cfg, model, ema, optimizer, scheduler)
    logger.info("Starting HPO trial" + (f" (resume at epoch {start_epoch})" if start_epoch > 0 else ""))

    pause_path = os.path.join(cfg.checkpoint.dir, "PAUSE")

    try:
        for fold_idx, (tr_idx, val_idx) in enumerate(folds[start_fold:], start=start_fold):
            # loaders for this fold
            train_loader = engine._make_loader(cfg, all_paths, targets, tr_idx, is_train=True,  shared_epoch=tr_shared_epoch)
            val_loader   = engine._make_loader(cfg, all_paths, targets, val_idx, is_train=False, shared_epoch=None)

            epochs_no_improve = 0
            local_best_auc = best_auc

            for epoch in range(start_epoch, cfg.scheduler.t_initial):
                # write resume marker BEFORE epoch
                engine.write_resume_marker(cfg, epoch=epoch, fold_idx=fold_idx, best_auc=best_auc)

                engine.train_one_epoch(
                    cfg=cfg, model=model, ema=ema, criterion=criterion,
                    optimizer=optimizer, scheduler=scheduler, loader=train_loader,
                    epoch=epoch, device=device, kornia_tf=kornia_tf,
                    mixup_fn=mixup_fn, stdscr=None, pause_file=pause_path
                )

                try:
                    val_auc = engine.validate(cfg, model, ema, criterion, val_loader, epoch, device)
                except FloatingPointError:
                    val_auc = -1.0

                best_auc = engine.save_checkpoint(
                    cfg, model, ema, optimizer, scheduler,
                    epoch=epoch, fold_idx=fold_idx, val_auc=val_auc, best_auc=best_auc
                )

                # epoch completed successfully; clear resume marker
                engine.clear_resume_marker(cfg)

                # bump ONLY training shared epoch so train workers drop caches for next epoch
                engine.bump_shared_epoch(tr_shared_epoch)
                # (no validation bump; keep val cache warm)

                if val_auc > local_best_auc:
                    local_best_auc = val_auc
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= cfg.early_stop.patience:
                    logger.info(f"Early stopping after {epochs_no_improve} non-improving epochs.")
                    break

            start_epoch = 0  # next fold starts at 0

    except Exception as e:
        logger.error(f"Training error in HPO trial: {e}")
        if rethrow_errors:
            raise
        return float("-inf")  # bad objective for maximize

    return float(best_auc)



# --- Optuna 4.x driver ---
def run_hpo(cfg: DictConfig) -> float:
    # Force sequential HPO so curses can own the TTY per trial.
    with open_dict(cfg.hpo):
        cfg.hpo.n_jobs = 1

    hcfg = cfg.hpo
    sampler = _build_sampler(getattr(hcfg, "sampler", None))
    fail_on_error = bool(getattr(hcfg, "fail_on_error", True))

    # Normalize storage so trials persist across separate program launches.
    base_ckpt_dir = getattr(cfg.checkpoint, "base_dir", "checkpoints")
    storage_uri = _normalize_storage_uri(str(getattr(hcfg, "storage", "")), base_ckpt_dir)

    study = optuna.create_study(
        study_name=str(hcfg.study_name),
        storage=storage_uri,
        load_if_exists=True,
        direction=str(hcfg.direction),
        sampler=sampler,
    )

    # Optional: clean up checkpoints only for COMPLETED trials.
    def _cleanup_ckpt_on_complete(enabled: bool):
        def _cb(study_: optuna.Study, t: optuna.trial.FrozenTrial):
            if not enabled:
                return
            from optuna.trial import TrialState as _TS
            if t.state == _TS.COMPLETE:
                ckpt_dir_ = t.user_attrs.get("ckpt_dir")
                if ckpt_dir_:
                    try:
                        import shutil
                        shutil.rmtree(ckpt_dir_, ignore_errors=True)
                        logger.info(f"[Trial {t.number}] cleaned checkpoint dir: {ckpt_dir_}")
                    except Exception as e:
                        logger.warning(f"[Trial {t.number}] checkpoint cleanup failed: {e}")
        return _cb

    callbacks = []
    if getattr(hcfg, "max_failure_rate", None) is not None:
        callbacks.append(_stop_on_failure_rate(float(hcfg.max_failure_rate)))
    callbacks.append(_cleanup_ckpt_on_complete(bool(getattr(hcfg, "cleanup_on_complete", False))))

    # Enqueue the most recent unfinished trial's params (to resume next).
    try:
        from optuna.trial import TrialState as TS
        unfinished = [t for t in study.get_trials(deepcopy=False)
                      if t.state in (TS.RUNNING, TS.FAIL, TS.PRUNED)]
        for t in sorted(unfinished, key=lambda x: x.number, reverse=True):
            ckpt_dir = t.user_attrs.get("ckpt_dir")
            if not ckpt_dir:
                continue
            marker = os.path.join(ckpt_dir, "resume_marker.json")
            lastpth = os.path.join(ckpt_dir, "last.pth")
            if os.path.isfile(marker) or os.path.isfile(lastpth):
                logger.info(f"Enqueuing params from unfinished trial {t.number} for resume.")
                study.enqueue_trial(t.params)
                break
    except Exception as e:
        logger.warning(f"Resume enqueue skipped due to error: {e}")

    def objective(trial: optuna.Trial) -> float:
        # Apply trial params to a fresh config
        cfg_trial = _apply_trial_to_cfg(trial, cfg)

        # Compute stable, param-hashed checkpoint dir now (visible + persistent)
        _ensure_stable_checkpoint_dir(cfg_trial)
        ckpt_dir = cfg_trial.checkpoint.dir
        trial.set_user_attr("ckpt_dir", ckpt_dir)
        logger.info(f"[Trial {trial.number}] ckpt_dir={ckpt_dir}")

        try:
            # Run with curses so 'P' pause works; engine handles epoch-level resume.
            best_auc = curses.wrapper(run_training, cfg_trial)
            return float(best_auc)
        except KeyboardInterrupt:
            # Mark interruption; keep ckpt for precise resume on next launch.
            trial.set_user_attr("interrupted", True)
            raise
        except Exception:
            if fail_on_error:
                raise
            logger.exception(f"[Trial {trial.number}] continuing after error; returning -inf")
            return float("-inf")

    # Let KeyboardInterrupt propagate (so the program stops immediately and leaves artifacts intact).
    study.optimize(objective, n_trials=int(hcfg.n_trials), n_jobs=1, callbacks=callbacks)

    best = study.best_trial
    logger.info(f"Best trial AUC={best.value:.6f}; params={best.params}")

    # Persist a summary in Hydra run dir
    out_dir = HydraConfig.get().runtime.output_dir if HydraConfig.initialized() else os.getcwd()
    try:
        with open(os.path.join(out_dir, "optuna_best.json"), "w") as f:
            json.dump({"value": float(best.value), "params": best.params}, f, indent=2)
    except Exception:
        pass

    return float(best.value)

def run_training(stdscr: "curses._CursesWindow", cfg: DictConfig) -> float:
    """
    Curses-wrapped training entry. Keeps validation cache warm by not tying it to a shared epoch.
    """
    # ── Curses screen setup ────────────────────────────────────────────────
    stdscr.nodelay(True)
    curses.cbreak()
    stdscr.clear()
    max_y, max_x = stdscr.getmaxyx()

    def update_line(row: int, text: str) -> None:
        if row >= max_y:
            return
        display = text[: max_x - 1].ljust(max_x - 1)
        try:
            stdscr.addstr(row, 0, display)
            stdscr.refresh()
        except curses.error:
            pass

    # ── Logging / determinism / device ────────────────────────────────────
    try:
        logger.remove()  # silence console; logs go to file only
    except Exception:
        pass
    _setup_logger_to_hydra_run_dir()
    logger.info("CONFIG:\n" + OmegaConf.to_yaml(cfg))

    engine.set_seed(cfg.exp.seed)
    device = _device_from_cfg(cfg)
    if cfg.exp.determinism:
        torch.use_deterministic_algorithms(True)

    # ── Stable per-param checkpoint dir ───────────────────────────────────
    _ensure_stable_checkpoint_dir(cfg)

    # ── Build data & model plumbing ───────────────────────────────────────
    (train_loader_dummy, val_loader_dummy, folds,
     all_paths, targets, kornia_tf) = engine.build_dataloaders(cfg, device)

    # shared epoch counter ONLY for training; validation stays cache-warm
    tr_shared_epoch = engine.create_shared_epoch(0)

    model, ema = engine.build_model(cfg, device)
    criterion = engine.build_loss(cfg).to(device)
    optimizer = engine.build_optimizer(cfg, model)

    # size scheduler on first fold
    tr0, va0 = folds[0]
    train_loader = engine._make_loader(cfg, all_paths, targets, tr0, is_train=True,  shared_epoch=tr_shared_epoch)
    scheduler = engine.build_scheduler(cfg, optimizer, len(train_loader))
    mixup_fn = engine.MixupCutmixScheduler(cfg)

    start_epoch, start_fold, best_auc = engine.load_checkpoint(
        cfg, model, ema, optimizer, scheduler
    )

    pause_path = os.path.join(cfg.checkpoint.dir, "PAUSE")

    # Accurate, explicit status line about resumption.
    if start_epoch > 0 or os.path.isfile(engine._marker_path(cfg)):
        hdr = f"RESUMING at epoch {start_epoch + 1} / {cfg.scheduler.t_initial}, fold {start_fold + 1}."
    else:
        hdr = "Starting fresh."
    update_line(0, f"{hdr} Press 'P' to pause/resume.")
    logger.info(hdr + " Press 'P' to pause/resume.")

    try:
        for fold_idx, (tr_idx, val_idx) in enumerate(folds[start_fold:], start=start_fold):
            stdscr.erase(); stdscr.refresh()

            fold_msg = f"===== Fold {fold_idx + 1}/{cfg.exp.folds} ====="
            update_line(1, fold_msg)
            logger.info(fold_msg)

            train_loader = engine._make_loader(cfg, all_paths, targets, tr_idx, is_train=True,  shared_epoch=tr_shared_epoch)
            # Validation loader WITHOUT a shared epoch => cache persists across epochs
            val_loader   = engine._make_loader(cfg, all_paths, targets, val_idx, is_train=False, shared_epoch=None)

            epochs_no_improve = 0
            local_best_auc = best_auc

            for epoch in range(start_epoch, cfg.scheduler.t_initial):
                stdscr.erase(); stdscr.refresh()
                update_line(0, f"Epoch {epoch + 1}/{cfg.scheduler.t_initial} running (Fold {fold_idx + 1}). Press 'P' to pause/resume.")
                update_line(1, fold_msg)

                # Write resume marker BEFORE the epoch runs.
                engine.write_resume_marker(cfg, epoch=epoch, fold_idx=fold_idx, best_auc=best_auc)

                engine.train_one_epoch(
                    cfg=cfg, model=model, ema=ema, criterion=criterion,
                    optimizer=optimizer, scheduler=scheduler, loader=train_loader,
                    epoch=epoch, device=device, kornia_tf=kornia_tf,
                    mixup_fn=mixup_fn, stdscr=stdscr, pause_file=pause_path
                )

                try:
                    val_auc = engine.validate(cfg, model, ema, criterion, val_loader, epoch, device)
                    update_line(3, f"Epoch {epoch + 1}: Val AUC {val_auc:.4f}")
                except FloatingPointError:
                    val_auc = -1.0
                    update_line(3, f"Epoch {epoch + 1}: Val AUC ERROR (FloatingPointError)")

                best_auc = engine.save_checkpoint(
                    cfg, model, ema, optimizer, scheduler,
                    epoch=epoch, fold_idx=fold_idx, val_auc=val_auc, best_auc=best_auc
                )

                # Epoch completed successfully; clear marker.
                engine.clear_resume_marker(cfg)

                # bump ONLY training shared epoch so train workers drop caches for next epoch
                engine.bump_shared_epoch(tr_shared_epoch)
                # (no validation bump; keep val cache warm)

                if val_auc > local_best_auc:
                    local_best_auc = val_auc
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= cfg.early_stop.patience:
                    msg = f"Early stopping after {epochs_no_improve} non-improving epochs."
                    update_line(4, msg)
                    logger.info(msg)
                    break

            start_epoch = 0  # next fold restarts from epoch 0

    except Exception as e:
        logger.error(f"Training error: {e}")
        update_line(max_y - 2, f"ERROR: {e}. See train.log.")
        time.sleep(3)
        return best_auc
    finally:
        # Final guard: ensure allocator is drained before returning control
        try:
            if _device_from_cfg(cfg).type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            pass
        import gc as _gc
        _gc.collect()
        engine._malloc_trim()

    return best_auc


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    if cfg.logging.rich_traceback:
        traceback.install()

    # HPO path (Optuna 4.x)
    if getattr(cfg, "hpo", None) and bool(getattr(cfg.hpo, "enable", False)):
        best = run_hpo(cfg)
        return best

    # Single-run path (keeps your curses UI)
    best_auc = curses.wrapper(run_training, cfg)
    try:
        os.makedirs(cfg.checkpoint.dir, exist_ok=True)
        with open(os.path.join(cfg.checkpoint.dir, "result.json"), "w") as f:
            json.dump({"best_auc": float(best_auc)}, f)
    except Exception:
        pass
    return best_auc


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
