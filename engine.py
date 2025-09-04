# engine.py
from __future__ import annotations

import math
import os
import random
import time
from typing import List, Sequence

import cv2
cv2.setNumThreads(0)

import numpy as np
import torch
import albumentations as A
import gc

from hydra import utils as hydra_utils
from loguru import logger
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import timm
import kornia.augmentation as K
from tqdm import tqdm


# ───────────────────────────── utils ──────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _autocast_device(device: torch.device) -> str:
    return "cuda" if device.type == "cuda" else "cpu"

def _cpu_state_dict(sd: dict) -> dict:
    """Detach + move all tensors in a state_dict to CPU to keep checkpoint size/RAM low."""
    out = {}
    for k, v in sd.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu()
        else:
            out[k] = v
    return out

# ─────────────────────── Dataset & transforms ─────────────────────
class AlbumentationsDataset(Dataset):
    """
    Memory-safe pipeline: CPU does resize/pad/crop. Outputs uint8 CHW tensors.
    Normalization and heavy aug run on GPU.
    """
    def __init__(self, img_paths: Sequence[str], labels: Sequence[int], cfg):
        self.paths: List[str] = list(img_paths)
        self.labels: List[int] = list(labels)
        self.cfg = cfg
        self.tf_cpu = self._build_cpu_pipeline(cfg)
        self.use_cache = bool(cfg.data.cache_images)
        if self.use_cache:
            self._cache = [self._read_img(p) for p in self.paths]

    @staticmethod
    def _read_img(path: str) -> np.ndarray:
        # Suppress C-level libjpeg stderr (e.g., "Corrupt JPEG data: ...") during decode.
        import os, sys
        from contextlib import contextmanager

        @contextmanager
        def _silence_stderr_fd():
            devnull = open(os.devnull, "w")
            try:
                fd = sys.stderr.fileno()
            except Exception:
                fd = None
            if fd is None:
                try:
                    yield
                finally:
                    devnull.close()
                return
            saved = os.dup(fd)
            os.dup2(devnull.fileno(), fd)
            try:
                yield
            finally:
                os.dup2(saved, fd)
                os.close(saved)
                devnull.close()

        with _silence_stderr_fd():
            img = cv2.imread(path, cv2.IMREAD_COLOR)

        if img is None:
            # Fallback: decode with PIL (also silenced)
            try:
                with _silence_stderr_fd():
                    from PIL import Image, ImageFile
                    ImageFile.LOAD_TRUNCATED_IMAGES = True
                    with open(path, "rb") as f:
                        pil_img = Image.open(f)
                        pil_img.load()
                        pil_img = pil_img.convert("RGB")
                return np.array(pil_img)
            except Exception:
                raise FileNotFoundError(f"Failed to read image at {path}")

        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _build_cpu_pipeline(self, cfg):
        return A.Compose([
            A.LongestMaxSize(max_size=int(cfg.data.img_sz * 1.15)),
            A.PadIfNeeded(
                min_height=cfg.data.img_sz,
                min_width=cfg.data.img_sz,
                border_mode=cv2.BORDER_REFLECT_101,
            ),
            A.RandomCrop(cfg.data.img_sz, cfg.data.img_sz),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = self._cache[idx] if self.use_cache else self._read_img(self.paths[idx])
        img = self.tf_cpu(image=img)["image"]  # HWC uint8
        x = torch.from_numpy(img).permute(2, 0, 1).contiguous()  # CHW uint8 contiguous
        y = int(self.labels[idx])
        return x, y


# ───────────────────────── kornia GPU aug ─────────────────────────
def build_kornia(cfg):
    p_cpu = cfg.data.aug_cpu
    p_gpu = cfg.data.aug_gpu
    return K.AugmentationSequential(
        K.RandomHorizontalFlip(p=float(p_cpu.horizontal_flip)),
        K.RandomVerticalFlip(p=float(p_cpu.vertical_flip)),
        K.RandomAffine(
            degrees=float(p_cpu.rotation_deg),
            translate=(0.1, 0.1),
            scale=(0.85, 1.15),
            p=1.0,
        ),
        K.ColorJitter(
            brightness=float(p_cpu.color.brightness),
            contrast=float(p_cpu.color.contrast),
            saturation=float(p_cpu.color.saturation),
            hue=float(p_cpu.color.hue),
            p=1.0,
        ),
        K.RandomErasing(
            p=float(p_gpu.random_erasing_p),
            scale=tuple(p_gpu.random_erasing_scale),
            ratio=tuple(p_gpu.random_erasing_ratio),
            same_on_batch=False,
        ),
        K.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        data_keys=["input"],
        same_on_batch=False,
    )


# ───────────────────── loss variants ───────────────────────────────
class AsymmetricLoss(nn.Module):
    """
    Numerically stable asymmetric focal-style BCE.
    Computes all sensitive ops in fp32 using softplus identities, so it is safe
    under autocast/AMP and avoids log(0)/Inf/NaN.

    log(sigmoid(x))     = -softplus(-x)
    log(1 - sigmoid(x)) = -softplus(x)
    """
    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 0.0):
        super().__init__()
        self.gn = float(gamma_neg)
        self.gp = float(gamma_pos)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Accept targets as class indices or one-hot/multi-label.
        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=logits.shape[1])

        # Do all math in fp32 on the same device, regardless of autocast dtype.
        logits_f = logits.float()
        target_f = target.to(dtype=torch.float32, device=logits_f.device)

        # Probabilities and focal weight.
        p = torch.sigmoid(logits_f)  # in (0,1) for finite logits
        pt = p * target_f + (1.0 - p) * (1.0 - target_f)
        gamma = self.gp * target_f + self.gn * (1.0 - target_f)
        weight = (1.0 - pt).pow(gamma)

        # Stable log-probs via softplus (no log(0)).
        logp = -torch.nn.functional.softplus(-logits_f)  # log(sigmoid)
        log1mp = -torch.nn.functional.softplus(logits_f) # log(1 - sigmoid)

        loss = -weight * (target_f * logp + (1.0 - target_f) * log1mp)
        return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0):
        super().__init__()
        self.g = gamma
        self.a = alpha
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ce = self.ce(x, y)
        pt = torch.exp(-ce)
        return (self.a * (1 - pt) ** self.g * ce).mean()


class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.s = smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(x, dim=-1)
        nll = -log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        smooth = -log_probs.mean(dim=-1)
        return (1 - self.s) * nll + self.s * smooth


class CombinedFocalLSCE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.focal = FocalLoss(gamma=2.0, alpha=1.0)
        self.lsce = LabelSmoothingCE(smoothing=0.1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.focal(x, y) + self.lsce(x, y)


def build_loss(cfg) -> nn.Module:
    if cfg.loss.type == "asymmetric":
        return AsymmetricLoss(cfg.loss.gamma_neg, cfg.loss.gamma_pos)
    return CombinedFocalLSCE(cfg)


# ─────────────────────── model / ema ───────────────────────────────
def build_model(cfg, device: torch.device):
    model = timm.create_model(
        cfg.model.name,
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained,
        drop_rate=cfg.model.drop_rate,
        drop_path_rate=cfg.model.drop_path_rate,
    ).to(device)

    # Keep standard memory format to prevent .view() issues from 3rd-party code.
    try:
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    except Exception:
        pass

    if getattr(cfg.exp, "torch_compile", False):
        # model = torch.compile(model, mode="max-autotune")
        pass

    ema = timm.utils.ModelEmaV2(model, decay=cfg.ema.decay, device=device) if cfg.ema.enable else None
    return model, ema


# ─────────────────── optimizer & scheduler ─────────────────────────
def build_optimizer(cfg, model: nn.Module):
    """
    Build optimizer with defensive defaults so third-party optimizers (e.g., AdamP)
    get all the param group keys they expect across versions.
    """
    opt_type = str(cfg.optim.type).lower()

    if opt_type.endswith("adamp"):
        # AdamP expects 'nesterov' in each param group in some versions.
        # Pull optional fields from cfg if present; otherwise use safe defaults.
        try:
            from adamp import AdamP
        except Exception as e:
            raise ImportError(
                "optim.type=adamp requested but 'adamp' package is not available."
            ) from e

        kwargs = dict(
            lr=cfg.optim.lr,
            betas=(cfg.optim.beta1, cfg.optim.beta2),
            weight_decay=cfg.optim.weight_decay,
        )
        # Optional AdamP-specific knobs with safe fallbacks
        if hasattr(cfg.optim, "eps"):
            kwargs["eps"] = cfg.optim.eps
        if hasattr(cfg.optim, "delta"):
            kwargs["delta"] = cfg.optim.delta
        if hasattr(cfg.optim, "wd_ratio"):
            kwargs["wd_ratio"] = cfg.optim.wd_ratio

        # Critical: ensure 'nesterov' exists in defaults (and thus in param groups)
        kwargs["nesterov"] = bool(getattr(cfg.optim, "nesterov", False))

        base_opt = AdamP(model.parameters(), **kwargs)

        # Harden param groups in case the underlying lib changes behavior
        for g in base_opt.param_groups:
            g.setdefault("nesterov", bool(getattr(cfg.optim, "nesterov", False)))

    else:
        # Default to AdamW
        base_opt = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.optim.lr,
            betas=(cfg.optim.beta1, cfg.optim.beta2),
            weight_decay=cfg.optim.weight_decay,
            eps=getattr(cfg.optim, "eps", 1e-8),
        )

    # Optional Lookahead wrapper (kept as-is)
    if opt_type.startswith("lookahead"):
        Lookahead = None
        try:
            from timm.optim.lookahead import Lookahead
        except Exception:
            try:
                from lookahead import Lookahead
            except Exception:
                Lookahead = None
        if Lookahead is not None:
            base_opt = Lookahead(base_opt)
        else:
            logger.warning("Lookahead requested but implementation not found; using base optimizer.")
    return base_opt


def build_scheduler(cfg, optimizer, steps_per_epoch: int):
    if cfg.scheduler.type == "onecycle":
        return OneCycleLR(
            optimizer,
            max_lr=cfg.optim.lr,
            total_steps=steps_per_epoch * cfg.scheduler.t_initial,
            pct_start=cfg.scheduler.warmup_epochs / max(cfg.scheduler.t_initial, 1),
            final_div_factor=cfg.optim.lr / max(cfg.scheduler.min_lr, 1e-12),
            anneal_strategy="cos",
        )
    from timm.scheduler import CosineLRScheduler
    cosine_kwargs = dict(
        optimizer=optimizer,
        t_initial=cfg.scheduler.t_initial,
        lr_min=cfg.scheduler.min_lr,
        warmup_t=cfg.scheduler.warmup_epochs,
        warmup_lr_init=cfg.optim.lr * 0.1,
        cycle_limit=1,
    )
    if "cycle_mul" in CosineLRScheduler.__init__.__code__.co_varnames:
        cosine_kwargs["cycle_mul"] = 1.0
    else:
        cosine_kwargs["t_mul"] = 1.0
    return CosineLRScheduler(**cosine_kwargs)


def _scheduler_step_per_batch(scheduler, epoch: int, step: int, num_steps: int):
    if isinstance(scheduler, OneCycleLR):
        scheduler.step()
        return
    if hasattr(scheduler, "step_update"):
        scheduler.step_update(epoch * num_steps + step)
        return
    # else: do nothing; epoch-level step happens later


# ─────────────────────── data loaders ─────────────────────────────
def build_dataloaders(cfg, device: torch.device):
    cfg.data.root = hydra_utils.to_absolute_path(cfg.data.root)
    all_paths, targets = _scan_dataset(cfg)

    if cfg.exp.folds == 1:
        tr_idx, val_idx = train_test_split(
            np.arange(len(all_paths)),
            test_size=0.2,
            stratify=targets,
            random_state=cfg.exp.seed,
        )
        folds = [(tr_idx, val_idx)]
    else:
        skf = StratifiedKFold(n_splits=cfg.exp.folds, shuffle=True, random_state=cfg.exp.seed)
        folds = list(skf.split(all_paths, targets))

    kornia_tf = build_kornia(cfg)

    tr_idx, val_idx = folds[0]
    tr_dl = _make_loader(cfg, all_paths, targets, tr_idx, is_train=True)
    val_dl = _make_loader(cfg, all_paths, targets, val_idx, is_train=False)
    return tr_dl, val_dl, folds, all_paths, targets, kornia_tf


def _scan_dataset(cfg):
    cls_to_idx = {}
    all_paths, targets = [], []
    root = cfg.data.root
    for class_name in sorted(os.listdir(root)):
        cls_dir = os.path.join(root, class_name)
        if not os.path.isdir(cls_dir):
            continue
        cls_idx = len(cls_to_idx)
        cls_to_idx[class_name] = cls_idx
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                all_paths.append(os.path.join(cls_dir, fname))
                targets.append(cls_idx)
    if not all_paths:
        raise RuntimeError(f"No images found under {root}")
    return np.array(all_paths), np.array(targets)


def _make_loader(cfg, paths, targets, indices, is_train: bool):
    subset_paths = [paths[i] for i in indices]
    subset_labels = [int(targets[i]) for i in indices]
    ds = AlbumentationsDataset(subset_paths, subset_labels, cfg)

    sampler, shuffle = None, is_train
    if is_train and cfg.data.balance_sampler:
        label_counts = np.bincount(subset_labels)
        weights = 1.0 / (label_counts[subset_labels] + 1e-12)
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
        )
        shuffle = False

    loader_kwargs = dict(
        batch_size=(cfg.data.batch_size if is_train else cfg.data.val_batch_size),
        sampler=sampler,
        shuffle=shuffle,
        num_workers=cfg.exp.num_workers,
        pin_memory=bool(cfg.exp.pin_memory),
        drop_last=is_train,
    )
    if cfg.exp.num_workers and cfg.exp.num_workers > 0:
        loader_kwargs.update(
            persistent_workers=bool(getattr(cfg.exp, "persistent_workers", False)),
            prefetch_factor=int(getattr(cfg.exp, "prefetch_factor", 2)),
        )
    return DataLoader(ds, **loader_kwargs)


# ─────────────────────── mixup/cutmix ─────────────────────────────
class MixupCutmixScheduler:
    def __init__(self, cfg):
        self.cfg = cfg
        self.alpha_m = float(cfg.mixup_cutmix.mixup_alpha)
        self.alpha_c = float(cfg.mixup_cutmix.cutmix_alpha)

    @staticmethod
    def lam(alpha: float) -> float:
        if alpha <= 0:
            return 1.0
        return max(np.random.beta(alpha, alpha), 1e-2)

    def __call__(self, x: torch.Tensor, y: torch.Tensor, epoch_frac: float):
        if (not self.cfg.mixup_cutmix.enable) or (random.random() > self.cfg.mixup_cutmix.prob):
            return x, y, y, 1.0

        if epoch_frac >= self.cfg.mixup_cutmix.switch_epoch:
            a_m, a_c = 0.0, 0.0
        else:
            a_m, a_c = self.alpha_m, self.alpha_c

        if random.random() < 0.5:  # mixup
            lam = self.lam(a_m)
            idx = torch.randperm(x.size(0), device=x.device)
            x = lam * x + (1.0 - lam) * x[idx]
            return x, y, y[idx], float(lam)

        # cutmix
        lam = self.lam(a_c)
        y1, x1, y2, x2 = self._rand_bbox(x.size(), lam)
        idx = torch.randperm(x.size(0), device=x.device)
        x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
        lam = 1.0 - ((y2 - y1) * (x2 - x1) / (x.size(-1) * x.size(-2)))
        return x, y, y[idx], float(lam)

    @staticmethod
    def _rand_bbox(size, lam):
        H, W = size[2], size[3]
        cut_rat = math.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        x1 = np.clip(cx - cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y2 = np.clip(cy + cut_h // 2, 0, H)
        return y1, x1, y2, x2


# ───────────────────── train / validate ─────────────────────────
def train_one_epoch(
    cfg,
    model: nn.Module,
    ema,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loader: DataLoader,
    epoch: int,
    device: torch.device,
    kornia_tf,
    mixup_fn: MixupCutmixScheduler,
    stdscr=None,
) -> None:
    model.train()
    num_steps = len(loader)

    scaler = torch.amp.GradScaler(enabled=cfg.exp.amp, growth_interval=15000)
    is_paused = False

    bar = tqdm(
        loader,
        desc=f"Epoch {epoch + 1}/{cfg.scheduler.t_initial}",
        total=num_steps,
        dynamic_ncols=True,
        leave=False,
    )

    running_loss = 0.0
    shown_steps = 0

    for step, (x, y) in enumerate(bar, start=1):
        # Pause handling
        if stdscr is not None:
            key = stdscr.getch()
            if key in (ord("p"), ord("P")):
                is_paused = not is_paused
            while is_paused:
                try:
                    stdscr.addstr(0, 0, "[PAUSED] Press 'P' to resume...".ljust(80))
                    stdscr.refresh()
                except Exception:
                    pass
                time.sleep(0.2)
                key = stdscr.getch()
                if key in (ord("p"), ord("P")):
                    is_paused = False

        # x: uint8 CHW on CPU
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        # convert to float and run GPU augment + normalize
        x = x.float().div_(255.0)
        x = kornia_tf(x)
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).contiguous()

        epoch_frac = (epoch + step / max(num_steps, 1)) / max(cfg.scheduler.t_initial, 1)
        x, y_a, y_b, lam = mixup_fn(x, y, epoch_frac)
        x = x.contiguous()

        with torch.autocast(device_type=_autocast_device(device), enabled=cfg.exp.amp):
            # 💡 FIX: Add a tiny amount of noise to prevent zero-variance batches.
            # This is a robust way to prevent division-by-zero in normalization layers.
            if cfg.exp.amp:
                x.add_(torch.randn_like(x), alpha=1e-6)

            logits = model(x)
            logits.clamp_(-15.0, 15.0) # Keep the previous fix as well
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
            loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)

        # Skip non-finite loss to prevent parameter corruption
        if not torch.isfinite(loss):
            logger.warning(f"Non-finite loss at epoch {epoch + 1} step {step}; batch skipped.")
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        # Unscale to check/clip grads safely
        scaler.unscale_(optimizer)
        if cfg.optim.grad_clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip_norm)

        # Abort step if any gradient is non-finite
        bad_grad = False
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                bad_grad = True
                break
        if bad_grad:
            logger.warning(f"Non-finite gradient at epoch {epoch + 1} step {step}; batch skipped.")
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            continue

        scaler.step(optimizer)
        scaler.update()

        _scheduler_step_per_batch(scheduler, epoch, step, num_steps)

        if ema is not None:
            ema.update(model)

        if step % 10 == 0:
            running_loss += loss.detach().cpu().item()
            shown_steps += 1
            bar.set_postfix(
                loss=f"{running_loss / max(shown_steps,1):.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}"
            )

    if (not isinstance(scheduler, OneCycleLR)) and hasattr(scheduler, "step") and not hasattr(scheduler, "step_update"):
        scheduler.step(epoch + 1)


def validate(cfg, model: nn.Module, ema, criterion: nn.Module, loader: DataLoader, epoch: int, device: torch.device) -> float:
    model_to_eval = ema.module if ema is not None else model
    model_to_eval.eval()

    loss_meter = 0.0
    preds, gts = [], []

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).reshape(1, 3, 1, 1)

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x = x.float().div_(255.0)
            x = (x - mean) / std
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).contiguous()

            with torch.autocast(device_type=_autocast_device(device), enabled=cfg.exp.amp):
                logits = model_to_eval(x)
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
                loss = criterion(logits, y)

            if not torch.isfinite(loss):
                logger.warning(f"Non-finite val loss at epoch {epoch + 1}; batch skipped.")
                continue

            loss_meter += float(loss.detach().cpu().item())

            # Probability extraction:
            # - If multi-class (C>=2), use softmax and take class-1 probability (binary AUC convention).
            # - If single-logit, use sigmoid.
            if logits.ndim == 2 and logits.size(1) >= 2:
                prob = torch.softmax(logits, dim=1)[:, 1]
            else:
                prob = torch.sigmoid(logits.squeeze(-1))

            prob = torch.nan_to_num(prob, nan=0.5, posinf=1.0, neginf=0.0)

            preds.append(prob.float().cpu().numpy())
            gts.append(y.int().cpu().numpy())

    if not preds:
        logger.warning("No valid validation batches; returning AUC=0.5")
        return 0.5

    preds = np.concatenate(preds)
    gts = np.concatenate(gts)

    try:
        auc = roc_auc_score(gts, preds)
    except Exception:
        logger.warning("AUC computation failed (degenerate targets/preds); returning AUC=0.5")
        auc = 0.5

    logger.info(f"Epoch {epoch + 1}: val loss {loss_meter / max(len(loader), 1):.4f} | AUC {auc:.4f}")
    return float(auc)

# ───────────────────── checkpoint I/O ──────────────────────────────
def _ckpt_path(cfg, name: str) -> str:
    return os.path.join(cfg.checkpoint.dir, name)

def save_checkpoint(cfg, model, ema, optimizer, scheduler, epoch: int, fold_idx: int, val_auc: float, best_auc: float) -> float:
    """
    Memory-light checkpointing:
      - Always write a light 'last.pth' containing model+EMA weights and minimal metadata.
      - Optionally write a separate heavy 'last_heavy.pth' (optimizer/scheduler) every N epochs.
      - On improvement, write model-only best files: 'best_model.pth' and 'ema_best_model.pth'.
    """
    os.makedirs(cfg.checkpoint.dir, exist_ok=True)

    new_best = max(best_auc, val_auc)

    # 1) Light, resumable "last" (model + EMA + minimal metadata), all on CPU tensors.
    light = {
        "model": _cpu_state_dict(model.state_dict()),
        "ema": (_cpu_state_dict(ema.state_dict()) if ema else None),  # keep EMA object state for proper resume
        "epoch": int(epoch),
        "fold_idx": int(fold_idx),
        "best_auc": float(new_best),
    }
    torch.save(light, _ckpt_path(cfg, "last.pth"))
    del light
    gc.collect()

    # 2) Heavy bits (optimizer/scheduler) saved sparsely if configured.
    save_heavy_every = int(getattr(cfg.checkpoint, "save_heavy_every", 0) or 0)
    if save_heavy_every > 0 and ((epoch + 1) % save_heavy_every == 0):
        heavy = {
            "optimizer": optimizer.state_dict(),
            "scheduler": (scheduler.state_dict() if hasattr(scheduler, "state_dict") else None),
            "epoch": int(epoch),
            "fold_idx": int(fold_idx),
            "best_auc": float(new_best),
        }
        torch.save(heavy, _ckpt_path(cfg, "last_heavy.pth"))
        del heavy
        gc.collect()

    # 3) Best model-only snapshots (weights only; small; no optimizer).
    if val_auc > best_auc:
        logger.info(f"New best AUC: {val_auc:.4f}. Saving model-only checkpoints.")
        torch.save(_cpu_state_dict(model.state_dict()), _ckpt_path(cfg, "best_model.pth"))
        if ema:
            # store EMA *weights* (the averaged model parameters) as the best EMA model
            torch.save(_cpu_state_dict(ema.module.state_dict()), _ckpt_path(cfg, "ema_best_model.pth"))
        return val_auc

    return best_auc

def load_checkpoint(cfg, model, ema, optimizer, scheduler):
    """
    Robust resume:
      - Supports the new light 'last.pth' (model+EMA only) plus optional 'last_heavy.pth'.
      - Backward-compatible with legacy heavy 'last.pth' that already contains optimizer/scheduler.
      - Also tolerates model-only files if user points resume at them (starts optimizers fresh).
    """
    if str(cfg.checkpoint.resume) == "none":
        return 0, 0, -1.0

    base = (_ckpt_path(cfg, "last.pth") if str(cfg.checkpoint.resume) == "auto"
            else str(cfg.checkpoint.resume))
    if not os.path.isfile(base):
        logger.warning(f"Checkpoint not found at {base}, starting from scratch.")
        return 0, 0, -1.0

    try:
        state = torch.load(base, map_location="cpu")

        # Load model weights (handle both dict-with-model and raw state_dict cases).
        if isinstance(state, dict) and "model" in state:
            model.load_state_dict(state["model"])
        else:
            model.load_state_dict(state)

        # Load EMA if available and requested.
        if ema is not None:
            ema_state = (state.get("ema") if isinstance(state, dict) else None)
            if ema_state is not None:
                ema.load_state_dict(ema_state)

        # Try to load optimizer/scheduler either from this file (legacy heavy last.pth)
        # or from a separate heavy sidecar file.
        loaded_heavy = False
        if isinstance(state, dict) and state.get("optimizer") is not None:
            optimizer.load_state_dict(state["optimizer"])
            if state.get("scheduler") is not None and hasattr(scheduler, "load_state_dict"):
                scheduler.load_state_dict(state["scheduler"])
            loaded_heavy = True
        else:
            heavy_path = _ckpt_path(cfg, "last_heavy.pth")
            if os.path.isfile(heavy_path):
                heavy = torch.load(heavy_path, map_location="cpu")
                optimizer.load_state_dict(heavy["optimizer"])
                if heavy.get("scheduler") is not None and hasattr(scheduler, "load_state_dict"):
                    scheduler.load_state_dict(heavy["scheduler"])
                loaded_heavy = True

        # Harden param groups for AdamP edge-case (keeps compatibility across versions).
        for g in optimizer.param_groups:
            g.setdefault("nesterov", bool(getattr(cfg.optim, "nesterov", False)))

        # Epoch/fold/best metadata if present; otherwise default.
        epoch_in_state = (state.get("epoch") if isinstance(state, dict) else None)
        fold_in_state  = (state.get("fold_idx") if isinstance(state, dict) else None)
        best_in_state  = (state.get("best_auc") if isinstance(state, dict) else None)

        start_epoch = int(epoch_in_state if epoch_in_state is not None else -1) + 1
        start_fold  = int(fold_in_state if fold_in_state is not None else 0)
        best_auc    = float(best_in_state if best_in_state is not None else -1.0)

        # If the checkpoint already completed the planned epochs, restart fresh.
        if start_epoch >= int(cfg.scheduler.t_initial):
            logger.info("Checkpoint is at/after planned epochs; restarting fresh.")
            return 0, 0, -1.0

        source = base if (isinstance(state, dict) and "model" in state) else "model-only file"
        extra  = " + heavy" if loaded_heavy else ""
        logger.info(f"Resumed from {source}{extra} (epoch {start_epoch}, fold {start_fold + 1})")
        return start_epoch, start_fold, best_auc

    except Exception as e:
        logger.error(f"Failed to load checkpoint from {base}: {e}")
        logger.warning("Starting from scratch.")
        return 0, 0, -1.0
