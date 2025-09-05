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

import engine


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


def _device_from_cfg(cfg: DictConfig) -> torch.device:
    want = str(cfg.exp.device).lower()
    if torch.cuda.is_available() and want.startswith("cuda"):
        return torch.device("cuda")
    return torch.device("cpu")


def run_training(stdscr: "curses._CursesWindow", cfg: DictConfig) -> float:
    """
    Curses-wrapped training entry. Clears the terminal at each epoch.
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

    # ── Build data & model plumbing ───────────────────────────────────────
    (train_loader, val_loader, folds,
     all_paths, targets, kornia_tf) = engine.build_dataloaders(cfg, device)

    model, ema = engine.build_model(cfg, device)
    criterion = engine.build_loss(cfg).to(device)
    optimizer = engine.build_optimizer(cfg, model)
    scheduler = engine.build_scheduler(cfg, optimizer, len(train_loader))
    mixup_fn = engine.MixupCutmixScheduler(cfg)

    start_epoch, start_fold, best_auc = engine.load_checkpoint(
        cfg, model, ema, optimizer, scheduler
    )

    update_line(0, "Training started. Press 'P' to pause/resume.")
    logger.info("Training started. Press 'P' in the active terminal to pause/resume.")

    try:
        for fold_idx, (tr_idx, val_idx) in enumerate(folds[start_fold:], start=start_fold):
            # Clear screen at the start of each fold.
            stdscr.erase(); stdscr.refresh()

            fold_msg = f"===== Fold {fold_idx + 1}/{cfg.exp.folds} ====="
            update_line(1, fold_msg)
            logger.info(fold_msg)

            train_loader = engine._make_loader(cfg, all_paths, targets, tr_idx, is_train=True)
            val_loader   = engine._make_loader(cfg, all_paths, targets, val_idx, is_train=False)

            epochs_no_improve = 0
            local_best_auc = best_auc

            for epoch in range(start_epoch, cfg.scheduler.t_initial):
                # Clear terminal between epochs.
                stdscr.erase(); stdscr.refresh()
                update_line(0, "Training started. Press 'P' to pause/resume.")
                update_line(1, fold_msg)
                update_line(2, f"Epoch {epoch + 1}/{cfg.scheduler.t_initial} running (Fold {fold_idx + 1})...")

                engine.train_one_epoch(
                    cfg=cfg, model=model, ema=ema, criterion=criterion,
                    optimizer=optimizer, scheduler=scheduler, loader=train_loader,
                    epoch=epoch, device=device, kornia_tf=kornia_tf,
                    mixup_fn=mixup_fn, stdscr=stdscr
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

            start_epoch = 0

    except Exception as e:
        logger.error(f"Training error: {e}")
        update_line(max_y - 2, f"ERROR: {e}. See train.log.")
        time.sleep(3)
        raise

    return best_auc


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    if cfg.logging.rich_traceback:
        traceback.install()
    return curses.wrapper(run_training, cfg)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
