# """
# Post-hoc analysis exporter for the thyroid nodule case-level model.

# This script does NOT train the model and does NOT modify the original training
# script. It imports the original model definition, loads saved fold checkpoints,
# reruns inference on each independent test fold, and exports:

# 1. all_case_predictions.csv
# 2. fold_hard_subset_metrics.csv
# 3. hard_subset_summary.csv
# 4. representative ROI-attention heatmaps
# 5. representative attention-guided log-SPD heatmaps

# ROI attention is visualized as a heatmap over key-frame / ROI indices, rather
# than by overlaying text on the original frames.
# """

# from __future__ import annotations

# import csv
# import argparse
# from importlib.machinery import SourceFileLoader
# import importlib.util
# import shutil
# from pathlib import Path
# from typing import Any, Dict, List, Tuple

# import matplotlib

# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# import torch.nn.functional as F
# from sklearn.metrics import (
#     accuracy_score,
#     average_precision_score,
#     precision_recall_fscore_support,
#     roc_auc_score,
# )
# from sklearn.model_selection import StratifiedKFold, train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from torch.utils.data import DataLoader, Dataset


# # Default source used only when --source is not provided. This can be either
# # the original training script or the project directory that contains it.
# SOURCE_SCRIPT = Path(r"/data1/syq/HYH/工作留存/工作三/spd_mu_aux+V3")

# # Output folder. By default this goes under the same roi_output folder used by
# # the original script after it is imported.
# ANALYSIS_SUBDIR = "posthoc_analysis_exports"

# # Hard cases are selected by the top cross-view dispersion scores.
# HARD_SUBSET_RATIO = 0.30

# # Set this to True if you want to export ROI-attention and log-SPD heatmaps
# # for every test case by default. The command-line flag --export-all-heatmaps
# # can also enable this behavior for a single run.
# EXPORT_ALL_HEATMAPS = False

# # Set this to True if you want to additionally export fold-level mean heatmaps
# # and five-fold mean heatmaps. ROI attention is resampled to fixed bins before
# # averaging because different cases may contain different numbers of ROIs.
# EXPORT_AVERAGE_HEATMAPS = False
# ATTENTION_AVERAGE_BINS = 20

# # Set this to True if you want one final five-model averaged heatmap for each
# # case. Each fold checkpoint is applied to every case, and the ROI attention
# # weights/log-SPD matrices from all available fold models are averaged per case.
# EXPORT_PER_CASE_FIVE_FOLD_AVERAGE_HEATMAPS = True
# EXPORT_PER_CASE_ATTENTION_OVERLAY_MONTAGE = True
# EXPORT_PER_CASE_ATTENTION_BARPLOT = True
# ROI_MONTAGE_COLUMNS = 5

# # Representative figures for the paper.
# REPRESENTATIVE_CATEGORIES = [
#     "benign_correct",
#     "malignant_correct",
#     "high_dispersion_correct",
#     "false_case",
# ]


# def resolve_source_script(source: str | Path) -> Path:
#     """Resolve a Python source file. If a directory is provided, pick a likely training script."""
#     source_path = Path(source)
#     if source_path.is_file():
#         return source_path
#     if not source_path.is_dir():
#         raise FileNotFoundError(f"Source path does not exist: {source_path}")

#     preferred_names = [
#         "终版1.py",
#         "final.py",
#         "train.py",
#         "main.py",
#     ]
#     for name in preferred_names:
#         candidate = source_path / name
#         if candidate.is_file():
#             return candidate

#     py_files = sorted(source_path.glob("*.py"))
#     if len(py_files) == 1:
#         return py_files[0]
#     if not py_files:
#         raise FileNotFoundError(f"No .py files found under source directory: {source_path}")

#     names = "\n".join(str(p) for p in py_files)
#     raise RuntimeError(
#         "Multiple .py files found. Please pass the exact training script path with --source.\n"
#         f"Candidates:\n{names}"
#     )


# def load_training_module(script_path: Path):
#     if not script_path.is_file():
#         raise RuntimeError(
#             "Cannot import source because it is not a file. "
#             f"Please pass the exact training script path with --source, got: {script_path}"
#         )
#     loader = SourceFileLoader("thyroid_training_module", str(script_path))
#     spec = importlib.util.spec_from_loader("thyroid_training_module", loader)
#     if spec is None or spec.loader is None:
#         raise RuntimeError(f"Cannot import source script: {script_path}")
#     module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(module)
#     return module


# def parse_args():
#     parser = argparse.ArgumentParser(
#         description=(
#             "Export post-hoc prediction tables, hard-case metrics, ROI-attention "
#             "heatmaps, and log-SPD heatmaps from trained fold checkpoints."
#         )
#     )
#     parser.add_argument(
#         "--source",
#         default=str(SOURCE_SCRIPT),
#         help="Path to the original training script or the project directory that contains it.",
#     )
#     parser.add_argument(
#         "--analysis-subdir",
#         default=ANALYSIS_SUBDIR,
#         help="Subdirectory created under ROI_OUTPUT_DIR for exported analysis results.",
#     )
#     parser.add_argument(
#         "--hard-ratio",
#         type=float,
#         default=HARD_SUBSET_RATIO,
#         help="Ratio of high-dispersion test cases selected in each fold for hard-case analysis.",
#     )
#     parser.add_argument(
#         "--export-all-heatmaps",
#         action="store_true",
#         default=EXPORT_ALL_HEATMAPS,
#         help="Export ROI-attention and log-SPD heatmaps for every test case.",
#     )
#     parser.add_argument(
#         "--export-average-heatmaps",
#         action="store_true",
#         default=EXPORT_AVERAGE_HEATMAPS,
#         help="Export fold-level and five-fold mean ROI-attention/log-SPD heatmaps.",
#     )
#     parser.add_argument(
#         "--attention-average-bins",
#         type=int,
#         default=ATTENTION_AVERAGE_BINS,
#         help="Number of normalized bins used when averaging variable-length ROI attention weights.",
#     )
#     parser.add_argument(
#         "--export-per-case-five-fold-average-heatmaps",
#         action="store_true",
#         default=EXPORT_PER_CASE_FIVE_FOLD_AVERAGE_HEATMAPS,
#         help=(
#             "Apply each fold checkpoint to every case and export one five-model "
#             "averaged ROI-attention/log-SPD heatmap for each case."
#         ),
#     )
#     parser.add_argument(
#         "--export-per-case-attention-overlay-montage",
#         action="store_true",
#         default=EXPORT_PER_CASE_ATTENTION_OVERLAY_MONTAGE,
#         help="Export ROI thumbnail montages with semi-transparent color overlays based on averaged attention weights.",
#     )
#     parser.add_argument(
#         "--export-per-case-attention-barplot",
#         action="store_true",
#         default=EXPORT_PER_CASE_ATTENTION_BARPLOT,
#         help="Export bar plots of five-model averaged ROI attention weights for each case.",
#     )
#     parser.add_argument(
#         "--roi-montage-columns",
#         type=int,
#         default=ROI_MONTAGE_COLUMNS,
#         help="Number of ROI thumbnails per row in the attention overlay montage.",
#     )
#     return parser.parse_args()


# def image_exts(module) -> List[str]:
#     exts = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]
#     if getattr(module, "ALLOW_PNG", True):
#         exts += ["*.png", "*.PNG"]
#     return exts


# def list_images_in_case_dir(module, case_dir: Path) -> List[Path]:
#     imgs: List[Path] = []
#     for pat in image_exts(module):
#         imgs.extend(case_dir.glob(pat))
#     return sorted(imgs)


# def precompute_case_roi_features(module, case_dir: Path, max_rois: int) -> np.ndarray:
#     imgs = list_images_in_case_dir(module, case_dir)
#     roi_feats: List[np.ndarray] = []
#     if imgs:
#         for p in imgs[:max_rois]:
#             try:
#                 from PIL import Image

#                 with Image.open(p) as im:
#                     roi_feats.append(module.extract_radiomics_features_simplified(im))
#             except Exception:
#                 continue
#     if not roi_feats:
#         roi_feats = [np.zeros(182, dtype=np.float32)]
#     return np.stack(roi_feats).astype(np.float32)


# def list_case_roi_paths(module, case_dir: Path, max_rois: int) -> List[str]:
#     return [str(p) for p in list_images_in_case_dir(module, case_dir)[:max_rois]]


# def fit_train_scaler(module, train_case_dirs: List[Path], max_rois: int) -> StandardScaler:
#     feats_all: List[np.ndarray] = []
#     for d in train_case_dirs:
#         feats_all.append(precompute_case_roi_features(module, d, max_rois=max_rois))
#     scaler = StandardScaler()
#     scaler.fit(np.concatenate(feats_all, axis=0))
#     return scaler


# class AnalysisDataset(Dataset):
#     def __init__(self, module, case_dirs: List[Path], labels: List[int], scaler: StandardScaler, max_rois: int):
#         self.module = module
#         self.case_dirs = case_dirs
#         self.labels = labels
#         self.scaler = scaler
#         self.max_rois = max_rois
#         self.data = self._precompute_all()

#     def _precompute_all(self) -> List[Dict[str, Any]]:
#         out: List[Dict[str, Any]] = []
#         for d, y in zip(self.case_dirs, self.labels):
#             x = precompute_case_roi_features(self.module, d, max_rois=self.max_rois)
#             x = self.scaler.transform(x).astype(np.float32)
#             x = np.clip(x, -self.module.SCALER_CLIP, self.module.SCALER_CLIP)
#             x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
#             out.append(
#                 {
#                     "roi_features": torch.from_numpy(x),
#                     "label": torch.tensor(int(y), dtype=torch.long),
#                     "case_name": d.name,
#                     "case_dir": str(d),
#                     "roi_paths": list_case_roi_paths(self.module, d, max_rois=self.max_rois),
#                 }
#             )
#         return out

#     def __len__(self) -> int:
#         return len(self.data)

#     def __getitem__(self, idx: int) -> Dict[str, Any]:
#         return self.data[idx]


# def analysis_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
#     return {
#         "roi_features": [item["roi_features"] for item in batch],
#         "label": torch.stack([item["label"] for item in batch]),
#         "case_name": [item["case_name"] for item in batch],
#         "case_dir": [item["case_dir"] for item in batch],
#         "roi_paths": [item["roi_paths"] for item in batch],
#     }


# def model_forward_with_details(module, model, roi_list: List[torch.Tensor]):
#     """Reproduce the original forward pass while collecting analysis tensors."""
#     device = module.DEVICE
#     mu_list, spd_list = [], []
#     alpha_list, alpha_spd_list, z_refined_list, valid_mask_list, dispersion_list = [], [], [], [], []

#     for x in roi_list:
#         x = x.to(device)
#         with torch.no_grad():
#             roi_variance = torch.var(x, dim=-1)
#             valid_mask = (roi_variance > 1e-5).float()
#             if valid_mask.sum() == 0:
#                 valid_mask[0] = 1.0

#         z = model.encoder(x)
#         z = z * valid_mask.unsqueeze(-1)

#         if model.cafr is not None and module.USE_MVCA:
#             z_refined = model.cafr(z, valid_mask=valid_mask)
#             z_refined = z_refined * valid_mask.unsqueeze(-1)
#         else:
#             z_refined = z

#         if module.MU_AGGREGATION_MODE == "attn" and model.attn is not None:
#             alpha, mu = model.attn(z_refined, valid_mask=valid_mask)
#             quality = torch.norm(z_refined, p=2, dim=1)
#             quality = quality / (quality.mean().detach() + 1e-8)
#             quality = torch.clamp(quality, min=0.3, max=2.0)
#             alpha_q = alpha * quality
#             alpha_q = alpha_q / (alpha_q.sum() + 1e-8)
#             spd = model.build_spd_second_moment(z_refined, alpha_q)
#         else:
#             v_count = valid_mask.sum().clamp(min=1.0)
#             mu = z_refined.sum(dim=0) / v_count
#             alpha = valid_mask / (valid_mask.sum() + 1e-8)
#             alpha_q = alpha
#             spd = model.build_spd_second_moment(z_refined, alpha_q)

#         dispersion = (alpha_q * torch.sum((z_refined - mu.unsqueeze(0)) ** 2, dim=1)).sum()

#         mu_list.append(mu)
#         spd_list.append(spd)
#         alpha_list.append(alpha.detach().cpu())
#         alpha_spd_list.append(alpha_q.detach().cpu())
#         z_refined_list.append(z_refined.detach().cpu())
#         valid_mask_list.append(valid_mask.detach().cpu())
#         dispersion_list.append(float(dispersion.detach().cpu()))

#     mu_batch = torch.stack(mu_list, dim=0)
#     spd_batch = torch.stack(spd_list, dim=0)
#     log_spd = module.robust_logm_spd_batch(spd_batch, mode="cpu")
#     v = module.spd_triu_vector(log_spd)

#     if module.USE_MU_IN_CLASSIFIER:
#         mu_projected = model.mu_projector(mu_batch)
#         feat = torch.cat([mu_projected, v], dim=1)
#     else:
#         feat = v

#     logits = model.head(feat)
#     return logits, {
#         "alpha": alpha_list,
#         "alpha_spd": alpha_spd_list,
#         "z_refined": z_refined_list,
#         "valid_mask": valid_mask_list,
#         "dispersion": dispersion_list,
#         "mu": mu_batch.detach().cpu(),
#         "spd": spd_batch.detach().cpu(),
#         "log_spd": log_spd.detach().cpu(),
#         "feat": feat.detach().cpu(),
#     }


# def safe_name(name: str) -> str:
#     return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(name))


# def safe_binary_metrics(labels: List[int], preds: List[int], probs: List[float]) -> Dict[str, float]:
#     if len(labels) == 0:
#         return {"n": 0, "accuracy": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan, "auc": np.nan, "ap": np.nan}
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
#     out = {
#         "n": len(labels),
#         "accuracy": accuracy_score(labels, preds),
#         "precision": precision,
#         "recall": recall,
#         "f1": f1,
#         "auc": np.nan,
#         "ap": np.nan,
#     }
#     if len(set(labels)) == 2:
#         out["auc"] = roc_auc_score(labels, probs)
#         out["ap"] = average_precision_score(labels, probs)
#     return out


# def save_roi_attention_heatmap(case_name: str, alpha: np.ndarray, out_path: Path):
#     """Visualize ROI attention as a true heatmap over key-frame indices."""
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     alpha = np.asarray(alpha, dtype=np.float32)
#     if alpha.size == 0:
#         return

#     fig_width = max(6.5, min(14.0, 0.55 * alpha.size + 2.0))
#     fig, ax = plt.subplots(figsize=(fig_width, 2.4))
#     im = ax.imshow(alpha.reshape(1, -1), cmap="YlOrRd", aspect="auto", vmin=0.0, vmax=max(float(alpha.max()), 1e-8))
#     ax.set_yticks([0])
#     ax.set_yticklabels(["Attention"])
#     ax.set_xticks(np.arange(alpha.size))
#     ax.set_xticklabels([str(i + 1) for i in range(alpha.size)], fontsize=8)
#     ax.set_xlabel("Key-frame / ROI index")
#     ax.set_title(f"ROI attention heatmap: {case_name}", fontsize=11)

#     for i, value in enumerate(alpha):
#         text_color = "white" if value > alpha.max() * 0.55 else "black"
#         ax.text(i, 0, f"{value:.2f}", ha="center", va="center", fontsize=7, color=text_color)

#     cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.035, pad=0.02)
#     cbar.set_label("Attention weight")
#     fig.tight_layout()
#     fig.savefig(out_path, dpi=240)
#     plt.close(fig)


# def save_roi_attention_2x2_heatmap(case_name: str, alpha: np.ndarray, out_path: Path):
#     """Visualize exactly four ROI attention weights as a 2 x 2 heatmap."""
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     alpha = np.asarray(alpha, dtype=np.float32).reshape(-1)
#     if alpha.size != 4:
#         return

#     matrix = alpha.reshape(2, 2)
#     fig, ax = plt.subplots(figsize=(5.2, 4.6))
#     im = ax.imshow(
#         matrix,
#         cmap="Reds",
#         aspect="equal",
#         vmin=max(0.0, float(alpha.min()) * 0.95),
#         vmax=max(float(alpha.max()), 1e-8),
#     )
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_title(f"ROI attention heatmap: {case_name}", fontsize=12)

#     for idx, value in enumerate(alpha):
#         row, col = divmod(idx, 2)
#         text_color = "white" if value > alpha.max() * 0.55 else "black"
#         ax.text(
#             col,
#             row,
#             f"ROI{idx + 1}\n{value:.3f}",
#             ha="center",
#             va="center",
#             fontsize=11,
#             color=text_color,
#         )

#     cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
#     cbar.set_label("Attention weight")
#     fig.tight_layout()
#     fig.savefig(out_path, dpi=240)
#     plt.close(fig)


# def save_spd_heatmap(case_name: str, matrix: np.ndarray, out_path: Path):
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     fig, ax = plt.subplots(figsize=(6.2, 5.4))
#     im = ax.imshow(matrix, cmap="coolwarm", aspect="auto")
#     ax.set_title(f"Attention-guided log-SPD heatmap: {case_name}", fontsize=11)
#     ax.set_xlabel("Latent dimension")
#     ax.set_ylabel("Latent dimension")
#     fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#     fig.tight_layout()
#     fig.savefig(out_path, dpi=240)
#     plt.close(fig)


# def resample_attention(alpha: np.ndarray, n_bins: int) -> np.ndarray:
#     """Resample variable-length ROI attention weights to fixed normalized bins."""
#     alpha = np.asarray(alpha, dtype=np.float32).reshape(-1)
#     if alpha.size == 0:
#         return np.zeros(n_bins, dtype=np.float32)
#     if alpha.size == 1:
#         return np.full(n_bins, float(alpha[0]), dtype=np.float32)

#     src_x = np.linspace(0.0, 1.0, alpha.size)
#     dst_x = np.linspace(0.0, 1.0, n_bins)
#     return np.interp(dst_x, src_x, alpha).astype(np.float32)


# def save_average_roi_attention_heatmap(title: str, alpha_mean: np.ndarray, out_path: Path):
#     """Visualize averaged ROI attention after normalization to fixed ROI-position bins."""
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     alpha_mean = np.asarray(alpha_mean, dtype=np.float32)
#     if alpha_mean.size == 0:
#         return

#     fig_width = max(7.5, min(14.0, 0.42 * alpha_mean.size + 2.2))
#     fig, ax = plt.subplots(figsize=(fig_width, 2.4))
#     im = ax.imshow(
#         alpha_mean.reshape(1, -1),
#         cmap="YlOrRd",
#         aspect="auto",
#         vmin=0.0,
#         vmax=max(float(alpha_mean.max()), 1e-8),
#     )
#     ax.set_yticks([0])
#     ax.set_yticklabels(["Mean attention"])
#     ax.set_xticks(np.arange(alpha_mean.size))
#     ax.set_xticklabels([str(i + 1) for i in range(alpha_mean.size)], fontsize=8)
#     ax.set_xlabel("Normalized ROI-position bin")
#     ax.set_title(title, fontsize=11)

#     for i, value in enumerate(alpha_mean):
#         text_color = "white" if value > alpha_mean.max() * 0.55 else "black"
#         ax.text(i, 0, f"{value:.2f}", ha="center", va="center", fontsize=7, color=text_color)

#     cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.035, pad=0.02)
#     cbar.set_label("Mean attention weight")
#     fig.tight_layout()
#     fig.savefig(out_path, dpi=240)
#     plt.close(fig)


# def save_roi_attention_barplot(case_name: str, alpha: np.ndarray, out_path: Path):
#     """Save a quantitative bar plot of ROI-level attention weights."""
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     alpha = np.asarray(alpha, dtype=np.float32).reshape(-1)
#     if alpha.size == 0:
#         return

#     fig_width = max(7.0, min(14.0, 0.45 * alpha.size + 2.5))
#     fig, ax = plt.subplots(figsize=(fig_width, 3.2))
#     x = np.arange(alpha.size)
#     colors = plt.cm.YlOrRd(alpha / max(float(alpha.max()), 1e-8))
#     ax.bar(x + 1, alpha, color=colors, edgecolor="#333333", linewidth=0.4)
#     ax.set_xlabel("ROI index")
#     ax.set_ylabel("Five-model mean attention weight")
#     ax.set_title(f"ROI attention bar plot: {case_name}", fontsize=11)
#     ax.set_xticks(x + 1)
#     ax.set_xticklabels([str(i + 1) for i in x], fontsize=8)
#     ax.set_ylim(0, max(float(alpha.max()) * 1.18, 1e-3))
#     ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)

#     for i, value in enumerate(alpha):
#         ax.text(i + 1, value, f"{value:.2f}", ha="center", va="bottom", fontsize=7)

#     fig.tight_layout()
#     fig.savefig(out_path, dpi=240)
#     plt.close(fig)


# def save_roi_attention_overlay_montage(
#     case_name: str,
#     roi_paths: List[str],
#     alpha: np.ndarray,
#     out_path: Path,
#     n_cols: int = 5,
# ):
#     """Overlay ROI-level attention weights on ROI thumbnails as translucent color blocks."""
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     alpha = np.asarray(alpha, dtype=np.float32).reshape(-1)
#     if alpha.size == 0 or not roi_paths:
#         return

#     n = min(len(roi_paths), alpha.size)
#     roi_paths = roi_paths[:n]
#     alpha = alpha[:n]
#     n_cols = max(1, int(n_cols))
#     n_rows = int(np.ceil(n / n_cols))

#     fig_width = min(3.0 * n_cols, 15.0)
#     fig_height = 3.0 * n_rows + 0.6
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
#     axes = np.asarray(axes).reshape(-1)

#     a_min = float(alpha.min())
#     a_max = float(alpha.max())
#     denom = max(a_max - a_min, 1e-8)
#     norm_values = (alpha - a_min) / denom
#     cmap = plt.cm.YlOrRd

#     for idx, ax in enumerate(axes):
#         ax.axis("off")
#         if idx >= n:
#             continue
#         try:
#             from PIL import Image

#             with Image.open(roi_paths[idx]) as im:
#                 img = np.asarray(im.convert("L"))
#         except Exception:
#             img = np.zeros((160, 160), dtype=np.uint8)

#         ax.imshow(img, cmap="gray", vmin=0, vmax=255)
#         overlay = np.ones((*img.shape, 4), dtype=np.float32)
#         overlay_color = cmap(norm_values[idx])
#         overlay[..., 0] = overlay_color[0]
#         overlay[..., 1] = overlay_color[1]
#         overlay[..., 2] = overlay_color[2]
#         overlay[..., 3] = 0.18 + 0.52 * float(norm_values[idx])
#         ax.imshow(overlay)
#         ax.set_title(f"ROI {idx + 1}: {alpha[idx]:.2f}", fontsize=8, pad=2)

#     fig.suptitle(
#         f"ROI-level attention overlay montage: {case_name}",
#         fontsize=12,
#         y=0.995,
#     )
#     sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=a_min, vmax=a_max))
#     sm.set_array([])
#     cbar = fig.colorbar(sm, ax=axes[:n], orientation="horizontal", fraction=0.035, pad=0.035)
#     cbar.set_label("Five-model mean attention weight")
#     fig.tight_layout(rect=[0, 0.05, 1, 0.96])
#     fig.savefig(out_path, dpi=240)
#     plt.close(fig)


# def export_average_heatmaps(payloads: List[Dict[str, Any]], out_dir: Path, attention_bins: int):
#     """Export fold-level means and the overall five-fold mean heatmaps."""
#     if not payloads:
#         return
#     if attention_bins <= 1:
#         raise ValueError("--attention-average-bins must be greater than 1.")

#     out_dir.mkdir(parents=True, exist_ok=True)

#     def export_subset(name: str, subset: List[Dict[str, Any]]):
#         if not subset:
#             return
#         mean_attention = np.mean(
#             [resample_attention(p["alpha"], attention_bins) for p in subset],
#             axis=0,
#         )
#         mean_log_spd = np.mean([np.asarray(p["log_spd"], dtype=np.float32) for p in subset], axis=0)

#         save_average_roi_attention_heatmap(
#             f"Mean ROI attention heatmap: {name}",
#             mean_attention,
#             out_dir / f"{safe_name(name)}_mean_roi_attention_heatmap.png",
#         )
#         save_spd_heatmap(
#             f"Mean log-SPD: {name}",
#             mean_log_spd,
#             out_dir / f"{safe_name(name)}_mean_log_spd_heatmap.png",
#         )

#     for fold in sorted({int(p["fold"]) for p in payloads}):
#         export_subset(f"fold{fold}", [p for p in payloads if int(p["fold"]) == fold])
#     export_subset("five_fold_overall", payloads)


# def compute_five_model_case_mean(payloads: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, float]:
#     """Average ROI attention, log-SPD matrix, and malignant probability across fold models."""
#     if not payloads:
#         raise ValueError("Cannot average an empty payload list.")

#     lengths = [len(p["alpha"]) for p in payloads if len(p["alpha"]) > 0]
#     if not lengths:
#         raise ValueError("Cannot average payloads without valid attention weights.")

#     target_len = max(lengths)
#     if len(set(lengths)) == 1:
#         mean_alpha = np.mean([np.asarray(p["alpha"], dtype=np.float32) for p in payloads], axis=0)
#     else:
#         mean_alpha = np.mean(
#             [resample_attention(p["alpha"], target_len) for p in payloads],
#             axis=0,
#         )

#     mean_log_spd = np.mean(
#         [np.asarray(p["log_spd"], dtype=np.float32) for p in payloads],
#         axis=0,
#     )
#     mean_prob = float(np.mean([p["prob_malignant"] for p in payloads]))
#     return mean_alpha, mean_log_spd, mean_prob


# def export_per_case_five_fold_average_heatmaps(
#     payloads_by_case: Dict[str, List[Dict[str, Any]]],
#     out_dir: Path,
#     export_overlay_montage: bool = True,
#     export_barplot: bool = True,
#     montage_columns: int = 5,
# ):
#     """Export one five-model averaged interpretation set for each case."""
#     out_dir.mkdir(parents=True, exist_ok=True)

#     for case_name, payloads in sorted(payloads_by_case.items()):
#         if not payloads:
#             continue
#         try:
#             mean_alpha, mean_log_spd, mean_prob = compute_five_model_case_mean(payloads)
#         except ValueError:
#             continue

#         label = payloads[0]["label"]
#         label_name = "malignant" if label == 1 else "benign"
#         pred_name = "malignant" if mean_prob >= 0.5 else "benign"
#         prefix = (
#             f"label-{label_name}_"
#             f"meanpred-{pred_name}_"
#             f"models-{len(payloads)}_"
#             f"{safe_name(case_name)}"
#         )

#         save_roi_attention_heatmap(
#             f"{case_name} | five-model mean",
#             mean_alpha,
#             out_dir / f"{prefix}_five_fold_mean_roi_attention_heatmap.png",
#         )
#         save_spd_heatmap(
#             f"{case_name} | five-model mean",
#             mean_log_spd,
#             out_dir / f"{prefix}_five_fold_mean_log_spd_heatmap.png",
#         )
#         if export_overlay_montage:
#             roi_paths = payloads[0].get("roi_paths", [])
#             save_roi_attention_overlay_montage(
#                 f"{case_name} | five-model mean",
#                 roi_paths,
#                 mean_alpha,
#                 out_dir / f"{prefix}_five_fold_mean_roi_attention_overlay_montage.png",
#                 n_cols=montage_columns,
#             )
#         if export_barplot:
#             save_roi_attention_barplot(
#                 f"{case_name} | five-model mean",
#                 mean_alpha,
#                 out_dir / f"{prefix}_five_fold_mean_roi_attention_barplot.png",
#             )


# def choose_representatives(payloads: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
#     selected: Dict[str, Dict[str, Any]] = {}
#     four_roi_payloads = [
#         p for p in payloads
#         if int(p.get("num_valid_rois", 0)) == 4
#         and len(p.get("alpha", [])) == 4
#     ]

#     if four_roi_payloads:
#         payloads = four_roi_payloads

#     benign_correct = [p for p in payloads if p["label"] == 0 and p["pred"] == 0]
#     if benign_correct:
#         selected["benign_correct"] = max(benign_correct, key=lambda p: 1.0 - p["prob_malignant"])

#     malignant_correct = [p for p in payloads if p["label"] == 1 and p["pred"] == 1]
#     if malignant_correct:
#         selected["malignant_correct"] = max(malignant_correct, key=lambda p: p["prob_malignant"])

#     correct_cases = [p for p in payloads if p["label"] == p["pred"]]
#     if correct_cases:
#         selected["high_dispersion_correct"] = max(correct_cases, key=lambda p: p["dispersion"])

#     false_cases = [p for p in payloads if p["label"] != p["pred"]]
#     if false_cases:
#         selected["false_case"] = max(false_cases, key=lambda p: p["dispersion"])

#     return selected


# def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]):
#     path.parent.mkdir(parents=True, exist_ok=True)
#     with open(path, "w", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerows(rows)


# def main():
#     args = parse_args()
#     if not (0 < args.hard_ratio <= 1):
#         raise ValueError("--hard-ratio must be in the interval (0, 1].")

#     source_script = resolve_source_script(args.source)
#     print(f"Using source script: {source_script}")

#     module = load_training_module(source_script)
#     out_root = module.ROI_OUTPUT_DIR / args.analysis_subdir
#     out_root.mkdir(parents=True, exist_ok=True)

#     labels_path, label_items = module.read_labels(module.BASE_ROOT)
#     cases = [c for c, _ in label_items]
#     labels = [y for _, y in label_items]

#     le = LabelEncoder()
#     labels_encoded = le.fit_transform(labels)
#     case_dirs = np.array([module.ROI_OUTPUT_DIR / c for c in cases])
#     labels_encoded = np.array(labels_encoded)

#     skf = StratifiedKFold(n_splits=module.N_FOLDS, shuffle=True, random_state=module.RANDOM_STATE)
#     all_records: List[Dict[str, Any]] = []
#     all_payloads: List[Dict[str, Any]] = []
#     fold_hard_metrics: List[Dict[str, Any]] = []
#     per_case_model_payloads: Dict[str, List[Dict[str, Any]]] = {}

#     for fold, (train_src_idx, test_idx) in enumerate(skf.split(case_dirs, labels_encoded)):
#         ckpt_path = module.ROI_OUTPUT_DIR / f"model_fold_{fold + 1}.pth"
#         if not ckpt_path.exists():
#             print(f"[Skip] Missing checkpoint: {ckpt_path}")
#             continue

#         fold_train_src_dirs = case_dirs[train_src_idx]
#         fold_train_src_labels = labels_encoded[train_src_idx]
#         fold_test_dirs = case_dirs[test_idx].tolist()
#         fold_test_labels = labels_encoded[test_idx].tolist()

#         train_dirs, _, train_labels, _ = train_test_split(
#             fold_train_src_dirs,
#             fold_train_src_labels,
#             test_size=module.VAL_SIZE,
#             random_state=module.RANDOM_STATE,
#             stratify=fold_train_src_labels,
#         )

#         scaler = fit_train_scaler(module, train_dirs.tolist(), max_rois=module.MAX_ROIS)
#         test_ds = AnalysisDataset(module, fold_test_dirs, fold_test_labels, scaler=scaler, max_rois=module.MAX_ROIS)
#         test_loader = DataLoader(
#             test_ds,
#             batch_size=module.BATCH_SIZE,
#             shuffle=False,
#             num_workers=0,
#             pin_memory=True,
#             collate_fn=analysis_collate_fn,
#         )

#         model = module.End2EndRadiomicsSPD(in_dim=182, z_dim=module.Z_DIM, attn_hidden=module.ATTN_HIDDEN, num_classes=len(le.classes_)).to(module.DEVICE)
#         state = torch.load(ckpt_path, map_location=module.DEVICE)
#         model.load_state_dict(state)
#         model.eval()

#         if args.export_per_case_five_fold_average_heatmaps:
#             all_case_ds = AnalysisDataset(
#                 module,
#                 case_dirs.tolist(),
#                 labels_encoded.tolist(),
#                 scaler=scaler,
#                 max_rois=module.MAX_ROIS,
#             )
#             all_case_loader = DataLoader(
#                 all_case_ds,
#                 batch_size=module.BATCH_SIZE,
#                 shuffle=False,
#                 num_workers=0,
#                 pin_memory=True,
#                 collate_fn=analysis_collate_fn,
#             )

#             with torch.no_grad():
#                 for batch in all_case_loader:
#                     labels_tensor = batch["label"].to(module.DEVICE)
#                     logits, details = model_forward_with_details(module, model, batch["roi_features"])
#                     probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
#                     preds = logits.argmax(dim=1).detach().cpu().numpy()
#                     labels_np = labels_tensor.detach().cpu().numpy()

#                     for i, case_name in enumerate(batch["case_name"]):
#                         valid_mask = details["valid_mask"][i].numpy()
#                         valid_n = int(valid_mask.sum())
#                         alpha = details["alpha"][i].numpy()
#                         alpha_valid = alpha[:valid_n] if valid_n > 0 else alpha

#                         per_case_model_payloads.setdefault(case_name, []).append(
#                             {
#                                 "fold_model": fold + 1,
#                                 "case_name": case_name,
#                                 "label": int(labels_np[i]),
#                                 "pred": int(preds[i]),
#                                 "prob_malignant": float(probs[i]),
#                                 "alpha": alpha_valid,
#                                 "log_spd": details["log_spd"][i].numpy(),
#                                 "roi_paths": batch["roi_paths"][i],
#                             }
#                         )

#         fold_records: List[Dict[str, Any]] = []
#         with torch.no_grad():
#             for batch in test_loader:
#                 labels_tensor = batch["label"].to(module.DEVICE)
#                 logits, details = model_forward_with_details(module, model, batch["roi_features"])
#                 probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
#                 preds = logits.argmax(dim=1).detach().cpu().numpy()
#                 labels_np = labels_tensor.detach().cpu().numpy()

#                 for i, case_name in enumerate(batch["case_name"]):
#                     valid_mask = details["valid_mask"][i].numpy()
#                     valid_n = int(valid_mask.sum())
#                     alpha = details["alpha"][i].numpy()
#                     alpha_valid = alpha[:valid_n] if valid_n > 0 else alpha

#                     rec = {
#                         "fold": fold + 1,
#                         "case_name": case_name,
#                         "label": int(labels_np[i]),
#                         "pred": int(preds[i]),
#                         "prob_malignant": float(probs[i]),
#                         "correct": int(labels_np[i] == preds[i]),
#                         "num_valid_rois": valid_n,
#                         "dispersion": float(details["dispersion"][i]),
#                         "max_attention": float(np.max(alpha_valid)) if len(alpha_valid) else np.nan,
#                         "attention_entropy": float(-np.sum(alpha_valid * np.log(alpha_valid + 1e-8))) if len(alpha_valid) else np.nan,
#                     }
#                     fold_records.append(rec)
#                     all_records.append(rec)
#                     all_payloads.append(
#                         {
#                             **rec,
#                             "alpha": alpha_valid,
#                             "log_spd": details["log_spd"][i].numpy(),
#                             "roi_paths": batch["roi_paths"][i],
#                         }
#                     )

#         n_hard = max(1, int(np.ceil(len(fold_records) * args.hard_ratio)))
#         hard_records = sorted(fold_records, key=lambda r: r["dispersion"], reverse=True)[:n_hard]
#         hard_metrics = safe_binary_metrics(
#             [r["label"] for r in hard_records],
#             [r["pred"] for r in hard_records],
#             [r["prob_malignant"] for r in hard_records],
#         )
#         hard_metrics.update({"fold": fold + 1, "selection": "top_dispersion", "hard_ratio": args.hard_ratio})
#         fold_hard_metrics.append(hard_metrics)
#         print(f"[Fold {fold + 1}] exported records: {len(fold_records)}, hard cases: {n_hard}")

#     record_fields = [
#         "fold",
#         "case_name",
#         "label",
#         "pred",
#         "prob_malignant",
#         "correct",
#         "num_valid_rois",
#         "dispersion",
#         "max_attention",
#         "attention_entropy",
#     ]
#     write_csv(out_root / "all_case_predictions.csv", all_records, record_fields)

#     hard_fields = ["fold", "selection", "hard_ratio", "n", "accuracy", "precision", "recall", "f1", "auc", "ap"]
#     write_csv(out_root / "fold_hard_subset_metrics.csv", fold_hard_metrics, hard_fields)

#     summary = {}
#     for key in ["accuracy", "precision", "recall", "f1", "auc", "ap"]:
#         vals = [m[key] for m in fold_hard_metrics if not np.isnan(m[key])]
#         summary[f"{key}_mean"] = float(np.mean(vals)) if vals else np.nan
#         summary[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
#     summary["hard_ratio"] = args.hard_ratio
#     summary["n_folds"] = len(fold_hard_metrics)
#     write_csv(out_root / "hard_subset_summary.csv", [summary], list(summary.keys()))

#     rep_dir = out_root / "representative_cases"
#     selected_all: Dict[str, Dict[str, Any]] = {}

#     def export_representative_payload(category: str, payload: Dict[str, Any], group_name: str):
#         case_dir = rep_dir / f"fold_{payload['fold']}" / safe_name(payload["case_name"])
#         prefix = f"{group_name}_{category}_fold{payload['fold']}"
#         save_roi_attention_heatmap(
#             payload["case_name"],
#             payload["alpha"],
#             case_dir / f"{prefix}_roi_attention_heatmap.png",
#         )
#         save_spd_heatmap(
#             payload["case_name"],
#             payload["log_spd"],
#             case_dir / f"{prefix}_log_spd_heatmap.png",
#         )
#         if len(payload.get("alpha", [])) == 4:
#             save_roi_attention_2x2_heatmap(
#                 payload["case_name"],
#                 payload["alpha"],
#                 case_dir / f"{prefix}_roi_attention_2x2_heatmap.png",
#             )
#         mean_payloads = per_case_model_payloads.get(payload["case_name"], [])
#         if mean_payloads:
#             try:
#                 mean_alpha, mean_log_spd, mean_prob = compute_five_model_case_mean(mean_payloads)
#             except ValueError:
#                 mean_alpha, mean_log_spd, mean_prob = None, None, None
#             if mean_alpha is not None and mean_log_spd is not None:
#                 mean_pred_name = "malignant" if mean_prob >= 0.5 else "benign"
#                 mean_prefix = f"{category}_five_fold_mean_pred-{mean_pred_name}_models-{len(mean_payloads)}"
#                 save_roi_attention_heatmap(
#                     f"{payload['case_name']} | five-fold mean",
#                     mean_alpha,
#                     case_dir / f"{mean_prefix}_roi_attention_heatmap.png",
#                 )
#                 save_spd_heatmap(
#                     f"{payload['case_name']} | five-fold mean",
#                     mean_log_spd,
#                     case_dir / f"{mean_prefix}_log_spd_heatmap.png",
#                 )
#                 if len(mean_alpha) == 4:
#                     save_roi_attention_2x2_heatmap(
#                         f"{payload['case_name']} | five-fold mean",
#                         mean_alpha,
#                         case_dir / f"{mean_prefix}_roi_attention_2x2_heatmap.png",
#                     )

#     for fold in sorted({int(p["fold"]) for p in all_payloads}):
#         fold_payloads = [p for p in all_payloads if int(p["fold"]) == fold]
#         selected_fold = choose_representatives(fold_payloads)
#         for category, payload in selected_fold.items():
#             key = f"fold{fold}_{category}"
#             selected_all[key] = payload
#             export_representative_payload(category, payload, group_name=f"fold{fold}")

#     selected_global = choose_representatives(all_payloads)
#     for category, payload in selected_global.items():
#         key = f"global_{category}"
#         selected_all[key] = payload
#         export_representative_payload(category, payload, group_name="global")

#     if args.export_all_heatmaps:
#         all_case_fig_dir = out_root / "all_case_heatmaps"
#         for payload in all_payloads:
#             correctness = "correct" if payload["label"] == payload["pred"] else "wrong"
#             label_name = "malignant" if payload["label"] == 1 else "benign"
#             pred_name = "malignant" if payload["pred"] == 1 else "benign"
#             prefix = (
#                 f"fold{payload['fold']}_"
#                 f"label-{label_name}_"
#                 f"pred-{pred_name}_"
#                 f"{correctness}_"
#                 f"{safe_name(payload['case_name'])}"
#             )

#             save_roi_attention_heatmap(
#                 payload["case_name"],
#                 payload["alpha"],
#                 all_case_fig_dir / f"{prefix}_roi_attention_heatmap.png",
#             )
#             save_spd_heatmap(
#                 payload["case_name"],
#                 payload["log_spd"],
#                 all_case_fig_dir / f"{prefix}_log_spd_heatmap.png",
#             )

#     if args.export_average_heatmaps:
#         export_average_heatmaps(
#             all_payloads,
#             out_root / "average_heatmaps",
#             attention_bins=args.attention_average_bins,
#         )

#     if args.export_per_case_five_fold_average_heatmaps:
#         export_per_case_five_fold_average_heatmaps(
#             per_case_model_payloads,
#             out_root / "per_case_five_fold_average_heatmaps",
#             export_overlay_montage=args.export_per_case_attention_overlay_montage,
#             export_barplot=args.export_per_case_attention_barplot,
#             montage_columns=args.roi_montage_columns,
#         )

#     # Save a copy of this analysis script for reproducibility.
#     try:
#         shutil.copy2(Path(__file__), out_root / "export_analysis_from_trained_models.py")
#     except Exception:
#         pass

#     print(f"\nDone. Analysis outputs saved to: {out_root}")
#     print("Representative cases:", ", ".join(selected_all.keys()) if selected_all else "none")


# if __name__ == "__main__":
#     main()






# """
# Post-hoc analysis exporter for the thyroid nodule case-level model.

# This script does NOT train the model and does NOT modify the original training
# script. It imports the original model definition, loads saved fold checkpoints,
# reruns inference on each independent test fold, and exports:

# 1. all_case_predictions.csv
# 2. fold_hard_subset_metrics.csv
# 3. hard_subset_summary.csv
# 4. representative ROI-attention heatmaps
# 5. representative attention-guided log-SPD heatmaps

# ROI attention is visualized as a heatmap over key-frame / ROI indices, rather
# than by overlaying text on the original frames.
# """

# from __future__ import annotations

# import csv
# import argparse
# from importlib.machinery import SourceFileLoader
# import importlib.util
# import shutil
# from pathlib import Path
# from typing import Any, Dict, List, Tuple

# import matplotlib

# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# import torch.nn.functional as F
# from sklearn.metrics import (
#     accuracy_score,
#     average_precision_score,
#     precision_recall_fscore_support,
#     roc_auc_score,
# )
# from sklearn.model_selection import StratifiedKFold, train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from torch.utils.data import DataLoader, Dataset


# # Default source used only when --source is not provided. This can be either
# # the original training script or the project directory that contains it.
# SOURCE_SCRIPT = Path(r"/data1/syq/HYH/工作留存/工作三/spd_mu_aux+V3")

# # Output folder. By default this goes under the same roi_output folder used by
# # the original script after it is imported.
# ANALYSIS_SUBDIR = "posthoc_analysis_exports"

# # Hard cases are selected by the top cross-view dispersion scores.
# HARD_SUBSET_RATIO = 0.30

# # Set this to True if you want to export ROI-attention and log-SPD heatmaps
# # for every test case by default. The command-line flag --export-all-heatmaps
# # can also enable this behavior for a single run.
# EXPORT_ALL_HEATMAPS = False

# # Set this to True if you want to additionally export fold-level mean heatmaps
# # and five-fold mean heatmaps. ROI attention is resampled to fixed bins before
# # averaging because different cases may contain different numbers of ROIs.
# EXPORT_AVERAGE_HEATMAPS = False
# ATTENTION_AVERAGE_BINS = 20

# # Set this to True if you want one final five-model averaged heatmap for each
# # case. Each fold checkpoint is applied to every case, and the ROI attention
# # weights/log-SPD matrices from all available fold models are averaged per case.
# EXPORT_PER_CASE_FIVE_FOLD_AVERAGE_HEATMAPS = True
# EXPORT_PER_CASE_ATTENTION_OVERLAY_MONTAGE = True
# EXPORT_PER_CASE_ATTENTION_BARPLOT = True
# ROI_MONTAGE_COLUMNS = 5

# # Representative figures for the paper.
# REPRESENTATIVE_CATEGORIES = [
#     "benign_correct",
#     "malignant_correct",
#     "high_dispersion_correct",
#     "false_case",
# ]


# def resolve_source_script(source: str | Path) -> Path:
#     """Resolve a Python source file. If a directory is provided, pick a likely training script."""
#     source_path = Path(source)
#     if source_path.is_file():
#         return source_path
#     if not source_path.is_dir():
#         raise FileNotFoundError(f"Source path does not exist: {source_path}")

#     preferred_names = [
#         "缁堢増1.py",
#         "final.py",
#         "train.py",
#         "main.py",
#     ]
#     for name in preferred_names:
#         candidate = source_path / name
#         if candidate.is_file():
#             return candidate

#     py_files = sorted(source_path.glob("*.py"))
#     if len(py_files) == 1:
#         return py_files[0]
#     if not py_files:
#         raise FileNotFoundError(f"No .py files found under source directory: {source_path}")

#     names = "\n".join(str(p) for p in py_files)
#     raise RuntimeError(
#         "Multiple .py files found. Please pass the exact training script path with --source.\n"
#         f"Candidates:\n{names}"
#     )


# def load_training_module(script_path: Path):
#     if not script_path.is_file():
#         raise RuntimeError(
#             "Cannot import source because it is not a file. "
#             f"Please pass the exact training script path with --source, got: {script_path}"
#         )
#     loader = SourceFileLoader("thyroid_training_module", str(script_path))
#     spec = importlib.util.spec_from_loader("thyroid_training_module", loader)
#     if spec is None or spec.loader is None:
#         raise RuntimeError(f"Cannot import source script: {script_path}")
#     module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(module)
#     return module


# def parse_args():
#     parser = argparse.ArgumentParser(
#         description=(
#             "Export post-hoc prediction tables, hard-case metrics, ROI-attention "
#             "heatmaps, and log-SPD heatmaps from trained fold checkpoints."
#         )
#     )
#     parser.add_argument(
#         "--source",
#         default=str(SOURCE_SCRIPT),
#         help="Path to the original training script or the project directory that contains it.",
#     )
#     parser.add_argument(
#         "--analysis-subdir",
#         default=ANALYSIS_SUBDIR,
#         help="Subdirectory created under ROI_OUTPUT_DIR for exported analysis results.",
#     )
#     parser.add_argument(
#         "--hard-ratio",
#         type=float,
#         default=HARD_SUBSET_RATIO,
#         help="Ratio of high-dispersion test cases selected in each fold for hard-case analysis.",
#     )
#     parser.add_argument(
#         "--export-all-heatmaps",
#         action="store_true",
#         default=EXPORT_ALL_HEATMAPS,
#         help="Export ROI-attention and log-SPD heatmaps for every test case.",
#     )
#     parser.add_argument(
#         "--export-average-heatmaps",
#         action="store_true",
#         default=EXPORT_AVERAGE_HEATMAPS,
#         help="Export fold-level and five-fold mean ROI-attention/log-SPD heatmaps.",
#     )
#     parser.add_argument(
#         "--attention-average-bins",
#         type=int,
#         default=ATTENTION_AVERAGE_BINS,
#         help="Number of normalized bins used when averaging variable-length ROI attention weights.",
#     )
#     parser.add_argument(
#         "--export-per-case-five-fold-average-heatmaps",
#         action="store_true",
#         default=EXPORT_PER_CASE_FIVE_FOLD_AVERAGE_HEATMAPS,
#         help=(
#             "Apply each fold checkpoint to every case and export one five-model "
#             "averaged ROI-attention/log-SPD heatmap for each case."
#         ),
#     )
#     parser.add_argument(
#         "--export-per-case-attention-overlay-montage",
#         action="store_true",
#         default=EXPORT_PER_CASE_ATTENTION_OVERLAY_MONTAGE,
#         help="Export ROI thumbnail montages with semi-transparent color overlays based on averaged attention weights.",
#     )
#     parser.add_argument(
#         "--export-per-case-attention-barplot",
#         action="store_true",
#         default=EXPORT_PER_CASE_ATTENTION_BARPLOT,
#         help="Export bar plots of five-model averaged ROI attention weights for each case.",
#     )
#     parser.add_argument(
#         "--roi-montage-columns",
#         type=int,
#         default=ROI_MONTAGE_COLUMNS,
#         help="Number of ROI thumbnails per row in the attention overlay montage.",
#     )
#     return parser.parse_args()


# def image_exts(module) -> List[str]:
#     exts = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]
#     if getattr(module, "ALLOW_PNG", True):
#         exts += ["*.png", "*.PNG"]
#     return exts


# def list_images_in_case_dir(module, case_dir: Path) -> List[Path]:
#     imgs: List[Path] = []
#     for pat in image_exts(module):
#         imgs.extend(case_dir.glob(pat))
#     return sorted(imgs)


# def precompute_case_roi_features(module, case_dir: Path, max_rois: int) -> np.ndarray:
#     imgs = list_images_in_case_dir(module, case_dir)
#     roi_feats: List[np.ndarray] = []
#     if imgs:
#         for p in imgs[:max_rois]:
#             try:
#                 from PIL import Image

#                 with Image.open(p) as im:
#                     roi_feats.append(module.extract_radiomics_features_simplified(im))
#             except Exception:
#                 continue
#     if not roi_feats:
#         roi_feats = [np.zeros(182, dtype=np.float32)]
#     return np.stack(roi_feats).astype(np.float32)


# def list_case_roi_paths(module, case_dir: Path, max_rois: int) -> List[str]:
#     return [str(p) for p in list_images_in_case_dir(module, case_dir)[:max_rois]]


# def fit_train_scaler(module, train_case_dirs: List[Path], max_rois: int) -> StandardScaler:
#     feats_all: List[np.ndarray] = []
#     for d in train_case_dirs:
#         feats_all.append(precompute_case_roi_features(module, d, max_rois=max_rois))
#     scaler = StandardScaler()
#     scaler.fit(np.concatenate(feats_all, axis=0))
#     return scaler


# class AnalysisDataset(Dataset):
#     def __init__(self, module, case_dirs: List[Path], labels: List[int], scaler: StandardScaler, max_rois: int):
#         self.module = module
#         self.case_dirs = case_dirs
#         self.labels = labels
#         self.scaler = scaler
#         self.max_rois = max_rois
#         self.data = self._precompute_all()

#     def _precompute_all(self) -> List[Dict[str, Any]]:
#         out: List[Dict[str, Any]] = []
#         for d, y in zip(self.case_dirs, self.labels):
#             x = precompute_case_roi_features(self.module, d, max_rois=self.max_rois)
#             x = self.scaler.transform(x).astype(np.float32)
#             x = np.clip(x, -self.module.SCALER_CLIP, self.module.SCALER_CLIP)
#             x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
#             out.append(
#                 {
#                     "roi_features": torch.from_numpy(x),
#                     "label": torch.tensor(int(y), dtype=torch.long),
#                     "case_name": d.name,
#                     "case_dir": str(d),
#                     "roi_paths": list_case_roi_paths(self.module, d, max_rois=self.max_rois),
#                 }
#             )
#         return out

#     def __len__(self) -> int:
#         return len(self.data)

#     def __getitem__(self, idx: int) -> Dict[str, Any]:
#         return self.data[idx]


# def analysis_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
#     return {
#         "roi_features": [item["roi_features"] for item in batch],
#         "label": torch.stack([item["label"] for item in batch]),
#         "case_name": [item["case_name"] for item in batch],
#         "case_dir": [item["case_dir"] for item in batch],
#         "roi_paths": [item["roi_paths"] for item in batch],
#     }


# def model_forward_with_details(module, model, roi_list: List[torch.Tensor]):
#     """Reproduce the original forward pass while collecting analysis tensors."""
#     device = module.DEVICE
#     mu_list, spd_list = [], []
#     alpha_list, alpha_spd_list, z_refined_list, valid_mask_list, dispersion_list = [], [], [], [], []

#     for x in roi_list:
#         x = x.to(device)
#         with torch.no_grad():
#             roi_variance = torch.var(x, dim=-1)
#             valid_mask = (roi_variance > 1e-5).float()
#             if valid_mask.sum() == 0:
#                 valid_mask[0] = 1.0

#         z = model.encoder(x)
#         z = z * valid_mask.unsqueeze(-1)

#         if model.cafr is not None and module.USE_MVCA:
#             z_refined = model.cafr(z, valid_mask=valid_mask)
#             z_refined = z_refined * valid_mask.unsqueeze(-1)
#         else:
#             z_refined = z

#         if module.MU_AGGREGATION_MODE == "attn" and model.attn is not None:
#             alpha, mu = model.attn(z_refined, valid_mask=valid_mask)
#             quality = torch.norm(z_refined, p=2, dim=1)
#             quality = quality / (quality.mean().detach() + 1e-8)
#             quality = torch.clamp(quality, min=0.3, max=2.0)
#             alpha_q = alpha * quality
#             alpha_q = alpha_q / (alpha_q.sum() + 1e-8)
#             spd = model.build_spd_second_moment(z_refined, alpha_q)
#         else:
#             v_count = valid_mask.sum().clamp(min=1.0)
#             mu = z_refined.sum(dim=0) / v_count
#             alpha = valid_mask / (valid_mask.sum() + 1e-8)
#             alpha_q = alpha
#             spd = model.build_spd_second_moment(z_refined, alpha_q)

#         dispersion = (alpha_q * torch.sum((z_refined - mu.unsqueeze(0)) ** 2, dim=1)).sum()

#         mu_list.append(mu)
#         spd_list.append(spd)
#         alpha_list.append(alpha.detach().cpu())
#         alpha_spd_list.append(alpha_q.detach().cpu())
#         z_refined_list.append(z_refined.detach().cpu())
#         valid_mask_list.append(valid_mask.detach().cpu())
#         dispersion_list.append(float(dispersion.detach().cpu()))

#     mu_batch = torch.stack(mu_list, dim=0)
#     spd_batch = torch.stack(spd_list, dim=0)
#     log_spd = module.robust_logm_spd_batch(spd_batch, mode="cpu")
#     v = module.spd_triu_vector(log_spd)

#     if module.USE_MU_IN_CLASSIFIER:
#         mu_projected = model.mu_projector(mu_batch)
#         feat = torch.cat([mu_projected, v], dim=1)
#     else:
#         feat = v

#     logits = model.head(feat)
#     return logits, {
#         "alpha": alpha_list,
#         "alpha_spd": alpha_spd_list,
#         "z_refined": z_refined_list,
#         "valid_mask": valid_mask_list,
#         "dispersion": dispersion_list,
#         "mu": mu_batch.detach().cpu(),
#         "spd": spd_batch.detach().cpu(),
#         "log_spd": log_spd.detach().cpu(),
#         "feat": feat.detach().cpu(),
#     }


# def safe_name(name: str) -> str:
#     return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(name))


# def safe_binary_metrics(labels: List[int], preds: List[int], probs: List[float]) -> Dict[str, float]:
#     if len(labels) == 0:
#         return {"n": 0, "accuracy": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan, "auc": np.nan, "ap": np.nan}
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
#     out = {
#         "n": len(labels),
#         "accuracy": accuracy_score(labels, preds),
#         "precision": precision,
#         "recall": recall,
#         "f1": f1,
#         "auc": np.nan,
#         "ap": np.nan,
#     }
#     if len(set(labels)) == 2:
#         out["auc"] = roc_auc_score(labels, probs)
#         out["ap"] = average_precision_score(labels, probs)
#     return out


# def save_roi_attention_heatmap(case_name: str, alpha: np.ndarray, out_path: Path):
#     """Visualize ROI attention as a true heatmap over key-frame indices."""
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     alpha = np.asarray(alpha, dtype=np.float32)
#     if alpha.size == 0:
#         return

#     fig_width = max(6.5, min(14.0, 0.55 * alpha.size + 2.0))
#     fig, ax = plt.subplots(figsize=(fig_width, 2.4))
#     im = ax.imshow(alpha.reshape(1, -1), cmap="YlOrRd", aspect="auto", vmin=0.0, vmax=max(float(alpha.max()), 1e-8))
#     ax.set_yticks([0])
#     ax.set_yticklabels(["Attention"])
#     ax.set_xticks(np.arange(alpha.size))
#     ax.set_xticklabels([str(i + 1) for i in range(alpha.size)], fontsize=8)
#     ax.set_xlabel("Key-frame / ROI index")
#     ax.set_title("ROIattentionheatmap", fontsize=11)

#     for i, value in enumerate(alpha):
#         text_color = "white" if value > alpha.max() * 0.55 else "black"
#         ax.text(i, 0, f"{value:.2f}", ha="center", va="center", fontsize=7, color=text_color)

#     cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.035, pad=0.02)
#     cbar.set_label("Attention weight")
#     fig.tight_layout()
#     fig.savefig(out_path, dpi=240)
#     plt.close(fig)


# def save_roi_attention_2x2_heatmap(case_name: str, alpha: np.ndarray, out_path: Path):
#     """Visualize exactly four ROI attention weights as a 2 x 2 heatmap."""
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     alpha = np.asarray(alpha, dtype=np.float32).reshape(-1)
#     if alpha.size != 4:
#         return

#     matrix = alpha.reshape(2, 2)
#     fig, ax = plt.subplots(figsize=(5.2, 4.6))
#     im = ax.imshow(
#         matrix,
#         cmap="Reds",
#         aspect="equal",
#         vmin=max(0.0, float(alpha.min()) * 0.95),
#         vmax=max(float(alpha.max()), 1e-8),
#     )
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_title("ROIattentionheatmap", fontsize=12)

#     for idx, value in enumerate(alpha):
#         row, col = divmod(idx, 2)
#         text_color = "white" if value > alpha.max() * 0.55 else "black"
#         ax.text(
#             col,
#             row,
#             f"ROI{idx + 1}\n{value:.3f}",
#             ha="center",
#             va="center",
#             fontsize=11,
#             color=text_color,
#         )

#     cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
#     cbar.set_label("Attention weight")
#     fig.tight_layout()
#     fig.savefig(out_path, dpi=240)
#     plt.close(fig)


# def save_spd_heatmap(case_name: str, matrix: np.ndarray, out_path: Path):
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     fig, ax = plt.subplots(figsize=(6.2, 5.4))
#     im = ax.imshow(matrix, cmap="coolwarm", aspect="auto")
#     ax.set_title(f"Attention-guided log-SPD heatmap: {case_name}", fontsize=11)
#     ax.set_xlabel("Latent dimension")
#     ax.set_ylabel("Latent dimension")
#     fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#     fig.tight_layout()
#     fig.savefig(out_path, dpi=240)
#     plt.close(fig)


# def resample_attention(alpha: np.ndarray, n_bins: int) -> np.ndarray:
#     """Resample variable-length ROI attention weights to fixed normalized bins."""
#     alpha = np.asarray(alpha, dtype=np.float32).reshape(-1)
#     if alpha.size == 0:
#         return np.zeros(n_bins, dtype=np.float32)
#     if alpha.size == 1:
#         return np.full(n_bins, float(alpha[0]), dtype=np.float32)

#     src_x = np.linspace(0.0, 1.0, alpha.size)
#     dst_x = np.linspace(0.0, 1.0, n_bins)
#     return np.interp(dst_x, src_x, alpha).astype(np.float32)


# def save_average_roi_attention_heatmap(title: str, alpha_mean: np.ndarray, out_path: Path):
#     """Visualize averaged ROI attention after normalization to fixed ROI-position bins."""
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     alpha_mean = np.asarray(alpha_mean, dtype=np.float32)
#     if alpha_mean.size == 0:
#         return

#     fig_width = max(7.5, min(14.0, 0.42 * alpha_mean.size + 2.2))
#     fig, ax = plt.subplots(figsize=(fig_width, 2.4))
#     im = ax.imshow(
#         alpha_mean.reshape(1, -1),
#         cmap="YlOrRd",
#         aspect="auto",
#         vmin=0.0,
#         vmax=max(float(alpha_mean.max()), 1e-8),
#     )
#     ax.set_yticks([0])
#     ax.set_yticklabels(["Mean attention"])
#     ax.set_xticks(np.arange(alpha_mean.size))
#     ax.set_xticklabels([str(i + 1) for i in range(alpha_mean.size)], fontsize=8)
#     ax.set_xlabel("Normalized ROI-position bin")
#     ax.set_title("ROIattentionheatmap", fontsize=11)

#     for i, value in enumerate(alpha_mean):
#         text_color = "white" if value > alpha_mean.max() * 0.55 else "black"
#         ax.text(i, 0, f"{value:.2f}", ha="center", va="center", fontsize=7, color=text_color)

#     cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.035, pad=0.02)
#     cbar.set_label("Mean attention weight")
#     fig.tight_layout()
#     fig.savefig(out_path, dpi=240)
#     plt.close(fig)


# def save_roi_attention_barplot(case_name: str, alpha: np.ndarray, out_path: Path):
#     """Save a quantitative bar plot of ROI-level attention weights."""
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     alpha = np.asarray(alpha, dtype=np.float32).reshape(-1)
#     if alpha.size == 0:
#         return

#     fig_width = max(7.0, min(14.0, 0.45 * alpha.size + 2.5))
#     fig, ax = plt.subplots(figsize=(fig_width, 3.2))
#     x = np.arange(alpha.size)
#     colors = plt.cm.YlOrRd(alpha / max(float(alpha.max()), 1e-8))
#     ax.bar(x + 1, alpha, color=colors, edgecolor="#333333", linewidth=0.4)
#     ax.set_xlabel("ROI index")
#     ax.set_ylabel("Five-model mean attention weight")
#     ax.set_title(f"ROI attention bar plot: {case_name}", fontsize=11)
#     ax.set_xticks(x + 1)
#     ax.set_xticklabels([str(i + 1) for i in x], fontsize=8)
#     ax.set_ylim(0, max(float(alpha.max()) * 1.18, 1e-3))
#     ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)

#     for i, value in enumerate(alpha):
#         ax.text(i + 1, value, f"{value:.2f}", ha="center", va="bottom", fontsize=7)

#     fig.tight_layout()
#     fig.savefig(out_path, dpi=240)
#     plt.close(fig)


# def save_roi_attention_overlay_montage(
#     case_name: str,
#     roi_paths: List[str],
#     alpha: np.ndarray,
#     out_path: Path,
#     n_cols: int = 5,
# ):
#     """Overlay ROI-level attention weights on ROI thumbnails as translucent color blocks."""
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     alpha = np.asarray(alpha, dtype=np.float32).reshape(-1)
#     if alpha.size == 0 or not roi_paths:
#         return

#     n = min(len(roi_paths), alpha.size)
#     roi_paths = roi_paths[:n]
#     alpha = alpha[:n]
#     n_cols = max(1, int(n_cols))
#     n_rows = int(np.ceil(n / n_cols))

#     fig_width = min(3.0 * n_cols, 15.0)
#     fig_height = 3.0 * n_rows + 0.6
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
#     axes = np.asarray(axes).reshape(-1)

#     a_min = float(alpha.min())
#     a_max = float(alpha.max())
#     denom = max(a_max - a_min, 1e-8)
#     norm_values = (alpha - a_min) / denom
#     cmap = plt.cm.YlOrRd

#     for idx, ax in enumerate(axes):
#         ax.axis("off")
#         if idx >= n:
#             continue
#         try:
#             from PIL import Image

#             with Image.open(roi_paths[idx]) as im:
#                 img = np.asarray(im.convert("L"))
#         except Exception:
#             img = np.zeros((160, 160), dtype=np.uint8)

#         ax.imshow(img, cmap="gray", vmin=0, vmax=255)
#         overlay = np.ones((*img.shape, 4), dtype=np.float32)
#         overlay_color = cmap(norm_values[idx])
#         overlay[..., 0] = overlay_color[0]
#         overlay[..., 1] = overlay_color[1]
#         overlay[..., 2] = overlay_color[2]
#         overlay[..., 3] = 0.18 + 0.52 * float(norm_values[idx])
#         ax.imshow(overlay)
#         ax.set_title(f"ROI {idx + 1}: {alpha[idx]:.2f}", fontsize=8, pad=2)

#     fig.suptitle(
#         f"ROI-level attention overlay montage: {case_name}",
#         fontsize=12,
#         y=0.995,
#     )
#     sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=a_min, vmax=a_max))
#     sm.set_array([])
#     cbar = fig.colorbar(sm, ax=axes[:n], orientation="horizontal", fraction=0.035, pad=0.035)
#     cbar.set_label("Five-model mean attention weight")
#     fig.tight_layout(rect=[0, 0.05, 1, 0.96])
#     fig.savefig(out_path, dpi=240)
#     plt.close(fig)


# def save_roi_index_mapping(
#     roi_paths: List[str],
#     alpha: np.ndarray,
#     out_path: Path,
#     source: str,
# ):
#     """Export the correspondence between displayed ROI indices and original ROI files."""
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     alpha = np.asarray(alpha, dtype=np.float32).reshape(-1)
#     n = min(len(roi_paths), alpha.size)
#     rows: List[Dict[str, Any]] = []
#     for idx in range(n):
#         roi_path = Path(roi_paths[idx])
#         rows.append(
#             {
#                 "source": source,
#                 "roi_label": f"ROI{idx + 1}",
#                 "roi_index": idx + 1,
#                 "roi_file": roi_path.name,
#                 "roi_path": str(roi_path),
#                 "attention_weight": float(alpha[idx]),
#             }
#         )
#     write_csv(
#         out_path,
#         rows,
#         ["source", "roi_label", "roi_index", "roi_file", "roi_path", "attention_weight"],
#     )


# def export_average_heatmaps(payloads: List[Dict[str, Any]], out_dir: Path, attention_bins: int):
#     """Export fold-level means and the overall five-fold mean heatmaps."""
#     if not payloads:
#         return
#     if attention_bins <= 1:
#         raise ValueError("--attention-average-bins must be greater than 1.")

#     out_dir.mkdir(parents=True, exist_ok=True)

#     def export_subset(name: str, subset: List[Dict[str, Any]]):
#         if not subset:
#             return
#         mean_attention = np.mean(
#             [resample_attention(p["alpha"], attention_bins) for p in subset],
#             axis=0,
#         )
#         mean_log_spd = np.mean([np.asarray(p["log_spd"], dtype=np.float32) for p in subset], axis=0)

#         save_average_roi_attention_heatmap(
#             f"Mean ROI attention heatmap: {name}",
#             mean_attention,
#             out_dir / f"{safe_name(name)}_mean_roi_attention_heatmap.png",
#         )
#         save_spd_heatmap(
#             f"Mean log-SPD: {name}",
#             mean_log_spd,
#             out_dir / f"{safe_name(name)}_mean_log_spd_heatmap.png",
#         )

#     for fold in sorted({int(p["fold"]) for p in payloads}):
#         export_subset(f"fold{fold}", [p for p in payloads if int(p["fold"]) == fold])
#     export_subset("five_fold_overall", payloads)


# def compute_five_model_case_mean(payloads: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, float]:
#     """Average ROI attention, log-SPD matrix, and malignant probability across fold models."""
#     if not payloads:
#         raise ValueError("Cannot average an empty payload list.")

#     lengths = [len(p["alpha"]) for p in payloads if len(p["alpha"]) > 0]
#     if not lengths:
#         raise ValueError("Cannot average payloads without valid attention weights.")

#     target_len = max(lengths)
#     if len(set(lengths)) == 1:
#         mean_alpha = np.mean([np.asarray(p["alpha"], dtype=np.float32) for p in payloads], axis=0)
#     else:
#         mean_alpha = np.mean(
#             [resample_attention(p["alpha"], target_len) for p in payloads],
#             axis=0,
#         )

#     mean_log_spd = np.mean(
#         [np.asarray(p["log_spd"], dtype=np.float32) for p in payloads],
#         axis=0,
#     )
#     mean_prob = float(np.mean([p["prob_malignant"] for p in payloads]))
#     return mean_alpha, mean_log_spd, mean_prob


# def export_per_case_five_fold_average_heatmaps(
#     payloads_by_case: Dict[str, List[Dict[str, Any]]],
#     out_dir: Path,
#     export_overlay_montage: bool = True,
#     export_barplot: bool = True,
#     montage_columns: int = 5,
# ):
#     """Export one five-model averaged interpretation set for each case."""
#     out_dir.mkdir(parents=True, exist_ok=True)

#     for case_name, payloads in sorted(payloads_by_case.items()):
#         if not payloads:
#             continue
#         try:
#             mean_alpha, mean_log_spd, mean_prob = compute_five_model_case_mean(payloads)
#         except ValueError:
#             continue

#         label = payloads[0]["label"]
#         label_name = "malignant" if label == 1 else "benign"
#         pred_name = "malignant" if mean_prob >= 0.5 else "benign"
#         prefix = (
#             f"label-{label_name}_"
#             f"meanpred-{pred_name}_"
#             f"models-{len(payloads)}_"
#             f"{safe_name(case_name)}"
#         )

#         save_roi_attention_heatmap(
#             f"{case_name} | five-model mean",
#             mean_alpha,
#             out_dir / f"{prefix}_five_fold_mean_roi_attention_heatmap.png",
#         )
#         save_spd_heatmap(
#             f"{case_name} | five-model mean",
#             mean_log_spd,
#             out_dir / f"{prefix}_five_fold_mean_log_spd_heatmap.png",
#         )
#         if export_overlay_montage:
#             roi_paths = payloads[0].get("roi_paths", [])
#             save_roi_attention_overlay_montage(
#                 f"{case_name} | five-model mean",
#                 roi_paths,
#                 mean_alpha,
#                 out_dir / f"{prefix}_five_fold_mean_roi_attention_overlay_montage.png",
#                 n_cols=montage_columns,
#             )
#         if export_barplot:
#             save_roi_attention_barplot(
#                 f"{case_name} | five-model mean",
#                 mean_alpha,
#                 out_dir / f"{prefix}_five_fold_mean_roi_attention_barplot.png",
#             )


# def choose_representatives(payloads: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
#     selected: Dict[str, Dict[str, Any]] = {}
#     four_roi_payloads = [
#         p for p in payloads
#         if int(p.get("num_valid_rois", 0)) == 4
#         and len(p.get("alpha", [])) == 4
#     ]

#     if four_roi_payloads:
#         payloads = four_roi_payloads

#     benign_correct = [p for p in payloads if p["label"] == 0 and p["pred"] == 0]
#     if benign_correct:
#         selected["benign_correct"] = max(benign_correct, key=lambda p: 1.0 - p["prob_malignant"])

#     malignant_correct = [p for p in payloads if p["label"] == 1 and p["pred"] == 1]
#     if malignant_correct:
#         selected["malignant_correct"] = max(malignant_correct, key=lambda p: p["prob_malignant"])

#     correct_cases = [p for p in payloads if p["label"] == p["pred"]]
#     if correct_cases:
#         selected["high_dispersion_correct"] = max(correct_cases, key=lambda p: p["dispersion"])

#     false_cases = [p for p in payloads if p["label"] != p["pred"]]
#     if false_cases:
#         selected["false_case"] = max(false_cases, key=lambda p: p["dispersion"])

#     return selected


# def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]):
#     path.parent.mkdir(parents=True, exist_ok=True)
#     with open(path, "w", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerows(rows)


# def main():
#     args = parse_args()
#     if not (0 < args.hard_ratio <= 1):
#         raise ValueError("--hard-ratio must be in the interval (0, 1].")

#     source_script = resolve_source_script(args.source)
#     print(f"Using source script: {source_script}")

#     module = load_training_module(source_script)
#     out_root = module.ROI_OUTPUT_DIR / args.analysis_subdir
#     out_root.mkdir(parents=True, exist_ok=True)

#     labels_path, label_items = module.read_labels(module.BASE_ROOT)
#     cases = [c for c, _ in label_items]
#     labels = [y for _, y in label_items]

#     le = LabelEncoder()
#     labels_encoded = le.fit_transform(labels)
#     case_dirs = np.array([module.ROI_OUTPUT_DIR / c for c in cases])
#     labels_encoded = np.array(labels_encoded)

#     skf = StratifiedKFold(n_splits=module.N_FOLDS, shuffle=True, random_state=module.RANDOM_STATE)
#     all_records: List[Dict[str, Any]] = []
#     all_payloads: List[Dict[str, Any]] = []
#     fold_hard_metrics: List[Dict[str, Any]] = []
#     per_case_model_payloads: Dict[str, List[Dict[str, Any]]] = {}

#     for fold, (train_src_idx, test_idx) in enumerate(skf.split(case_dirs, labels_encoded)):
#         ckpt_path = module.ROI_OUTPUT_DIR / f"model_fold_{fold + 1}.pth"
#         if not ckpt_path.exists():
#             print(f"[Skip] Missing checkpoint: {ckpt_path}")
#             continue

#         fold_train_src_dirs = case_dirs[train_src_idx]
#         fold_train_src_labels = labels_encoded[train_src_idx]
#         fold_test_dirs = case_dirs[test_idx].tolist()
#         fold_test_labels = labels_encoded[test_idx].tolist()

#         train_dirs, _, train_labels, _ = train_test_split(
#             fold_train_src_dirs,
#             fold_train_src_labels,
#             test_size=module.VAL_SIZE,
#             random_state=module.RANDOM_STATE,
#             stratify=fold_train_src_labels,
#         )

#         scaler = fit_train_scaler(module, train_dirs.tolist(), max_rois=module.MAX_ROIS)
#         test_ds = AnalysisDataset(module, fold_test_dirs, fold_test_labels, scaler=scaler, max_rois=module.MAX_ROIS)
#         test_loader = DataLoader(
#             test_ds,
#             batch_size=module.BATCH_SIZE,
#             shuffle=False,
#             num_workers=0,
#             pin_memory=True,
#             collate_fn=analysis_collate_fn,
#         )

#         model = module.End2EndRadiomicsSPD(in_dim=182, z_dim=module.Z_DIM, attn_hidden=module.ATTN_HIDDEN, num_classes=len(le.classes_)).to(module.DEVICE)
#         state = torch.load(ckpt_path, map_location=module.DEVICE)
#         model.load_state_dict(state)
#         model.eval()

#         if args.export_per_case_five_fold_average_heatmaps:
#             all_case_ds = AnalysisDataset(
#                 module,
#                 case_dirs.tolist(),
#                 labels_encoded.tolist(),
#                 scaler=scaler,
#                 max_rois=module.MAX_ROIS,
#             )
#             all_case_loader = DataLoader(
#                 all_case_ds,
#                 batch_size=module.BATCH_SIZE,
#                 shuffle=False,
#                 num_workers=0,
#                 pin_memory=True,
#                 collate_fn=analysis_collate_fn,
#             )

#             with torch.no_grad():
#                 for batch in all_case_loader:
#                     labels_tensor = batch["label"].to(module.DEVICE)
#                     logits, details = model_forward_with_details(module, model, batch["roi_features"])
#                     probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
#                     preds = logits.argmax(dim=1).detach().cpu().numpy()
#                     labels_np = labels_tensor.detach().cpu().numpy()

#                     for i, case_name in enumerate(batch["case_name"]):
#                         valid_mask = details["valid_mask"][i].numpy()
#                         valid_n = int(valid_mask.sum())
#                         alpha = details["alpha"][i].numpy()
#                         alpha_valid = alpha[:valid_n] if valid_n > 0 else alpha

#                         per_case_model_payloads.setdefault(case_name, []).append(
#                             {
#                                 "fold_model": fold + 1,
#                                 "case_name": case_name,
#                                 "label": int(labels_np[i]),
#                                 "pred": int(preds[i]),
#                                 "prob_malignant": float(probs[i]),
#                                 "alpha": alpha_valid,
#                                 "log_spd": details["log_spd"][i].numpy(),
#                                 "roi_paths": batch["roi_paths"][i],
#                             }
#                         )

#         fold_records: List[Dict[str, Any]] = []
#         with torch.no_grad():
#             for batch in test_loader:
#                 labels_tensor = batch["label"].to(module.DEVICE)
#                 logits, details = model_forward_with_details(module, model, batch["roi_features"])
#                 probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
#                 preds = logits.argmax(dim=1).detach().cpu().numpy()
#                 labels_np = labels_tensor.detach().cpu().numpy()

#                 for i, case_name in enumerate(batch["case_name"]):
#                     valid_mask = details["valid_mask"][i].numpy()
#                     valid_n = int(valid_mask.sum())
#                     alpha = details["alpha"][i].numpy()
#                     alpha_valid = alpha[:valid_n] if valid_n > 0 else alpha

#                     rec = {
#                         "fold": fold + 1,
#                         "case_name": case_name,
#                         "label": int(labels_np[i]),
#                         "pred": int(preds[i]),
#                         "prob_malignant": float(probs[i]),
#                         "correct": int(labels_np[i] == preds[i]),
#                         "num_valid_rois": valid_n,
#                         "dispersion": float(details["dispersion"][i]),
#                         "max_attention": float(np.max(alpha_valid)) if len(alpha_valid) else np.nan,
#                         "attention_entropy": float(-np.sum(alpha_valid * np.log(alpha_valid + 1e-8))) if len(alpha_valid) else np.nan,
#                     }
#                     fold_records.append(rec)
#                     all_records.append(rec)
#                     all_payloads.append(
#                         {
#                             **rec,
#                             "alpha": alpha_valid,
#                             "log_spd": details["log_spd"][i].numpy(),
#                             "roi_paths": batch["roi_paths"][i],
#                         }
#                     )

#         n_hard = max(1, int(np.ceil(len(fold_records) * args.hard_ratio)))
#         hard_records = sorted(fold_records, key=lambda r: r["dispersion"], reverse=True)[:n_hard]
#         hard_metrics = safe_binary_metrics(
#             [r["label"] for r in hard_records],
#             [r["pred"] for r in hard_records],
#             [r["prob_malignant"] for r in hard_records],
#         )
#         hard_metrics.update({"fold": fold + 1, "selection": "top_dispersion", "hard_ratio": args.hard_ratio})
#         fold_hard_metrics.append(hard_metrics)
#         print(f"[Fold {fold + 1}] exported records: {len(fold_records)}, hard cases: {n_hard}")

#     record_fields = [
#         "fold",
#         "case_name",
#         "label",
#         "pred",
#         "prob_malignant",
#         "correct",
#         "num_valid_rois",
#         "dispersion",
#         "max_attention",
#         "attention_entropy",
#     ]
#     write_csv(out_root / "all_case_predictions.csv", all_records, record_fields)

#     hard_fields = ["fold", "selection", "hard_ratio", "n", "accuracy", "precision", "recall", "f1", "auc", "ap"]
#     write_csv(out_root / "fold_hard_subset_metrics.csv", fold_hard_metrics, hard_fields)

#     summary = {}
#     for key in ["accuracy", "precision", "recall", "f1", "auc", "ap"]:
#         vals = [m[key] for m in fold_hard_metrics if not np.isnan(m[key])]
#         summary[f"{key}_mean"] = float(np.mean(vals)) if vals else np.nan
#         summary[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
#     summary["hard_ratio"] = args.hard_ratio
#     summary["n_folds"] = len(fold_hard_metrics)
#     write_csv(out_root / "hard_subset_summary.csv", [summary], list(summary.keys()))

#     rep_dir = out_root / "representative_cases"
#     selected_all: Dict[str, Dict[str, Any]] = {}

#     def export_representative_payload(category: str, payload: Dict[str, Any], group_name: str):
#         case_dir = rep_dir / f"fold_{payload['fold']}" / safe_name(payload["case_name"])
#         prefix = f"{group_name}_{category}_fold{payload['fold']}"
#         save_roi_attention_heatmap(
#             payload["case_name"],
#             payload["alpha"],
#             case_dir / f"{prefix}_roi_attention_heatmap.png",
#         )
#         save_spd_heatmap(
#             payload["case_name"],
#             payload["log_spd"],
#             case_dir / f"{prefix}_log_spd_heatmap.png",
#         )
#         if len(payload.get("alpha", [])) == 4:
#             save_roi_attention_2x2_heatmap(
#                 payload["case_name"],
#                 payload["alpha"],
#                 case_dir / f"{prefix}_roi_attention_2x2_heatmap.png",
#             )
#         save_roi_index_mapping(
#             payload.get("roi_paths", []),
#             payload["alpha"],
#             case_dir / f"{prefix}_roi_index_mapping.csv",
#             source="single_fold",
#         )
#         mean_payloads = per_case_model_payloads.get(payload["case_name"], [])
#         if mean_payloads:
#             try:
#                 mean_alpha, mean_log_spd, mean_prob = compute_five_model_case_mean(mean_payloads)
#             except ValueError:
#                 mean_alpha, mean_log_spd, mean_prob = None, None, None
#             if mean_alpha is not None and mean_log_spd is not None:
#                 mean_pred_name = "malignant" if mean_prob >= 0.5 else "benign"
#                 mean_prefix = f"{category}_five_fold_mean_pred-{mean_pred_name}_models-{len(mean_payloads)}"
#                 save_roi_attention_heatmap(
#                     f"{payload['case_name']} | five-fold mean",
#                     mean_alpha,
#                     case_dir / f"{mean_prefix}_roi_attention_heatmap.png",
#                 )
#                 save_spd_heatmap(
#                     f"{payload['case_name']} | five-fold mean",
#                     mean_log_spd,
#                     case_dir / f"{mean_prefix}_log_spd_heatmap.png",
#                 )
#                 if len(mean_alpha) == 4:
#                     save_roi_attention_2x2_heatmap(
#                         f"{payload['case_name']} | five-fold mean",
#                         mean_alpha,
#                         case_dir / f"{mean_prefix}_roi_attention_2x2_heatmap.png",
#                     )
#                 save_roi_index_mapping(
#                     mean_payloads[0].get("roi_paths", []),
#                     mean_alpha,
#                     case_dir / f"{mean_prefix}_roi_index_mapping.csv",
#                     source="five_fold_mean",
#                 )

#     for fold in sorted({int(p["fold"]) for p in all_payloads}):
#         fold_payloads = [p for p in all_payloads if int(p["fold"]) == fold]
#         selected_fold = choose_representatives(fold_payloads)
#         for category, payload in selected_fold.items():
#             key = f"fold{fold}_{category}"
#             selected_all[key] = payload
#             export_representative_payload(category, payload, group_name=f"fold{fold}")

#     selected_global = choose_representatives(all_payloads)
#     for category, payload in selected_global.items():
#         key = f"global_{category}"
#         selected_all[key] = payload
#         export_representative_payload(category, payload, group_name="global")

#     if args.export_all_heatmaps:
#         all_case_fig_dir = out_root / "all_case_heatmaps"
#         for payload in all_payloads:
#             correctness = "correct" if payload["label"] == payload["pred"] else "wrong"
#             label_name = "malignant" if payload["label"] == 1 else "benign"
#             pred_name = "malignant" if payload["pred"] == 1 else "benign"
#             prefix = (
#                 f"fold{payload['fold']}_"
#                 f"label-{label_name}_"
#                 f"pred-{pred_name}_"
#                 f"{correctness}_"
#                 f"{safe_name(payload['case_name'])}"
#             )

#             save_roi_attention_heatmap(
#                 payload["case_name"],
#                 payload["alpha"],
#                 all_case_fig_dir / f"{prefix}_roi_attention_heatmap.png",
#             )
#             save_spd_heatmap(
#                 payload["case_name"],
#                 payload["log_spd"],
#                 all_case_fig_dir / f"{prefix}_log_spd_heatmap.png",
#             )

#     if args.export_average_heatmaps:
#         export_average_heatmaps(
#             all_payloads,
#             out_root / "average_heatmaps",
#             attention_bins=args.attention_average_bins,
#         )

#     if args.export_per_case_five_fold_average_heatmaps:
#         export_per_case_five_fold_average_heatmaps(
#             per_case_model_payloads,
#             out_root / "per_case_five_fold_average_heatmaps",
#             export_overlay_montage=args.export_per_case_attention_overlay_montage,
#             export_barplot=args.export_per_case_attention_barplot,
#             montage_columns=args.roi_montage_columns,
#         )

#     # Save a copy of this analysis script for reproducibility.
#     try:
#         shutil.copy2(Path(__file__), out_root / "export_analysis_from_trained_models.py")
#     except Exception:
#         pass

#     print(f"\nDone. Analysis outputs saved to: {out_root}")
#     print("Representative cases:", ", ".join(selected_all.keys()) if selected_all else "none")


# if __name__ == "__main__":
#     main()




"""
Post-hoc analysis exporter for the thyroid nodule case-level model.

This script does NOT train the model and does NOT modify the original training
script. It imports the original model definition, loads saved fold checkpoints,
reruns inference on each independent test fold, and exports:

1. all_case_predictions.csv
2. fold_hard_subset_metrics.csv
3. hard_subset_summary.csv
4. representative ROI-attention heatmaps
5. representative attention-guided log-SPD heatmaps

ROI attention is visualized as a heatmap over key-frame / ROI indices, rather
than by overlaying text on the original frames.
"""

from __future__ import annotations

import csv
import argparse
from importlib.machinery import SourceFileLoader
import importlib.util
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    davies_bouldin_score,
    precision_recall_fscore_support,
    roc_auc_score,
    silhouette_score,
)
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset


# Default source used only when --source is not provided. This can be either
# the original training script or the project directory that contains it.
SOURCE_SCRIPT = Path(r"/data1/syq/HYH/工作留存/工作三/spd_mu_aux+V3")

# Output folder. By default this goes under the same roi_output folder used by
# the original script after it is imported.
ANALYSIS_SUBDIR = "posthoc_analysis_exports"

# Hard cases are selected by the top cross-view dispersion scores.
HARD_SUBSET_RATIO = 0.30

# Set this to True if you want to export ROI-attention and log-SPD heatmaps
# for every test case by default. The command-line flag --export-all-heatmaps
# can also enable this behavior for a single run.
EXPORT_ALL_HEATMAPS = False

# Set this to True if you want to additionally export fold-level mean heatmaps
# and five-fold mean heatmaps. ROI attention is resampled to fixed bins before
# averaging because different cases may contain different numbers of ROIs.
EXPORT_AVERAGE_HEATMAPS = False
ATTENTION_AVERAGE_BINS = 20

# Set this to True if you want one final five-model averaged heatmap for each
# case. Each fold checkpoint is applied to every case, and the ROI attention
# weights/log-SPD matrices from all available fold models are averaged per case.
EXPORT_PER_CASE_FIVE_FOLD_AVERAGE_HEATMAPS = True
EXPORT_PER_CASE_ATTENTION_OVERLAY_MONTAGE = True
EXPORT_PER_CASE_ATTENTION_BARPLOT = True
ROI_MONTAGE_COLUMNS = 5

# Export out-of-fold case-level features and draw t-SNE/UMAP distributions.
# These features are extracted only from the independent test cases of each
# fold, then concatenated across folds for visualization.
EXPORT_FEATURE_DISTRIBUTION = True
FEATURE_EMBEDDING_METHOD = "both"  # "tsne", "umap", or "both"

# Representative figures for the paper.
REPRESENTATIVE_CATEGORIES = [
    "benign_correct",
    "malignant_correct",
    "high_dispersion_correct",
    "false_case",
]


def resolve_source_script(source: str | Path) -> Path:
    """Resolve a Python source file. If a directory is provided, pick a likely training script."""
    source_path = Path(source)
    if source_path.is_file():
        return source_path
    if not source_path.is_dir():
        raise FileNotFoundError(f"Source path does not exist: {source_path}")

    preferred_names = [
        "终版1.py",
        "final.py",
        "train.py",
        "main.py",
    ]
    for name in preferred_names:
        candidate = source_path / name
        if candidate.is_file():
            return candidate

    py_files = sorted(source_path.glob("*.py"))
    if len(py_files) == 1:
        return py_files[0]
    if not py_files:
        raise FileNotFoundError(f"No .py files found under source directory: {source_path}")

    names = "\n".join(str(p) for p in py_files)
    raise RuntimeError(
        "Multiple .py files found. Please pass the exact training script path with --source.\n"
        f"Candidates:\n{names}"
    )


def load_training_module(script_path: Path):
    if not script_path.is_file():
        raise RuntimeError(
            "Cannot import source because it is not a file. "
            f"Please pass the exact training script path with --source, got: {script_path}"
        )
    loader = SourceFileLoader("thyroid_training_module", str(script_path))
    spec = importlib.util.spec_from_loader("thyroid_training_module", loader)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import source script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export post-hoc prediction tables, hard-case metrics, ROI-attention "
            "heatmaps, and log-SPD heatmaps from trained fold checkpoints."
        )
    )
    parser.add_argument(
        "--source",
        default=str(SOURCE_SCRIPT),
        help="Path to the original training script or the project directory that contains it.",
    )
    parser.add_argument(
        "--analysis-subdir",
        default=ANALYSIS_SUBDIR,
        help="Subdirectory created under ROI_OUTPUT_DIR for exported analysis results.",
    )
    parser.add_argument(
        "--hard-ratio",
        type=float,
        default=HARD_SUBSET_RATIO,
        help="Ratio of high-dispersion test cases selected in each fold for hard-case analysis.",
    )
    parser.add_argument(
        "--export-all-heatmaps",
        action="store_true",
        default=EXPORT_ALL_HEATMAPS,
        help="Export ROI-attention and log-SPD heatmaps for every test case.",
    )
    parser.add_argument(
        "--export-average-heatmaps",
        action="store_true",
        default=EXPORT_AVERAGE_HEATMAPS,
        help="Export fold-level and five-fold mean ROI-attention/log-SPD heatmaps.",
    )
    parser.add_argument(
        "--attention-average-bins",
        type=int,
        default=ATTENTION_AVERAGE_BINS,
        help="Number of normalized bins used when averaging variable-length ROI attention weights.",
    )
    parser.add_argument(
        "--export-per-case-five-fold-average-heatmaps",
        action="store_true",
        default=EXPORT_PER_CASE_FIVE_FOLD_AVERAGE_HEATMAPS,
        help=(
            "Apply each fold checkpoint to every case and export one five-model "
            "averaged ROI-attention/log-SPD heatmap for each case."
        ),
    )
    parser.add_argument(
        "--export-per-case-attention-overlay-montage",
        action="store_true",
        default=EXPORT_PER_CASE_ATTENTION_OVERLAY_MONTAGE,
        help="Export ROI thumbnail montages with semi-transparent color overlays based on averaged attention weights.",
    )
    parser.add_argument(
        "--export-per-case-attention-barplot",
        action="store_true",
        default=EXPORT_PER_CASE_ATTENTION_BARPLOT,
        help="Export bar plots of five-model averaged ROI attention weights for each case.",
    )
    parser.add_argument(
        "--roi-montage-columns",
        type=int,
        default=ROI_MONTAGE_COLUMNS,
        help="Number of ROI thumbnails per row in the attention overlay montage.",
    )
    parser.add_argument(
        "--export-feature-distribution",
        action="store_true",
        default=EXPORT_FEATURE_DISTRIBUTION,
        help="Export out-of-fold semantic/structural/fused features and draw t-SNE/UMAP distributions.",
    )
    parser.add_argument(
        "--feature-embedding-method",
        choices=["tsne", "umap", "both"],
        default=FEATURE_EMBEDDING_METHOD,
        help="Embedding method used for feature distribution visualization.",
    )
    return parser.parse_args()


def image_exts(module) -> List[str]:
    exts = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]
    if getattr(module, "ALLOW_PNG", True):
        exts += ["*.png", "*.PNG"]
    return exts


def list_images_in_case_dir(module, case_dir: Path) -> List[Path]:
    imgs: List[Path] = []
    for pat in image_exts(module):
        imgs.extend(case_dir.glob(pat))
    return sorted(imgs)


def precompute_case_roi_features(module, case_dir: Path, max_rois: int) -> np.ndarray:
    imgs = list_images_in_case_dir(module, case_dir)
    roi_feats: List[np.ndarray] = []
    if imgs:
        for p in imgs[:max_rois]:
            try:
                from PIL import Image

                with Image.open(p) as im:
                    roi_feats.append(module.extract_radiomics_features_simplified(im))
            except Exception:
                continue
    if not roi_feats:
        roi_feats = [np.zeros(182, dtype=np.float32)]
    return np.stack(roi_feats).astype(np.float32)


def list_case_roi_paths(module, case_dir: Path, max_rois: int) -> List[str]:
    return [str(p) for p in list_images_in_case_dir(module, case_dir)[:max_rois]]


def fit_train_scaler(module, train_case_dirs: List[Path], max_rois: int) -> StandardScaler:
    feats_all: List[np.ndarray] = []
    for d in train_case_dirs:
        feats_all.append(precompute_case_roi_features(module, d, max_rois=max_rois))
    scaler = StandardScaler()
    scaler.fit(np.concatenate(feats_all, axis=0))
    return scaler


class AnalysisDataset(Dataset):
    def __init__(self, module, case_dirs: List[Path], labels: List[int], scaler: StandardScaler, max_rois: int):
        self.module = module
        self.case_dirs = case_dirs
        self.labels = labels
        self.scaler = scaler
        self.max_rois = max_rois
        self.data = self._precompute_all()

    def _precompute_all(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for d, y in zip(self.case_dirs, self.labels):
            x = precompute_case_roi_features(self.module, d, max_rois=self.max_rois)
            x = self.scaler.transform(x).astype(np.float32)
            x = np.clip(x, -self.module.SCALER_CLIP, self.module.SCALER_CLIP)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            out.append(
                {
                    "roi_features": torch.from_numpy(x),
                    "label": torch.tensor(int(y), dtype=torch.long),
                    "case_name": d.name,
                    "case_dir": str(d),
                    "roi_paths": list_case_roi_paths(self.module, d, max_rois=self.max_rois),
                }
            )
        return out

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


def analysis_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "roi_features": [item["roi_features"] for item in batch],
        "label": torch.stack([item["label"] for item in batch]),
        "case_name": [item["case_name"] for item in batch],
        "case_dir": [item["case_dir"] for item in batch],
        "roi_paths": [item["roi_paths"] for item in batch],
    }


def model_forward_with_details(module, model, roi_list: List[torch.Tensor]):
    """Reproduce the original forward pass while collecting analysis tensors."""
    device = module.DEVICE
    mu_list, spd_list = [], []
    alpha_list, alpha_spd_list, z_refined_list, valid_mask_list, dispersion_list = [], [], [], [], []

    for x in roi_list:
        x = x.to(device)
        with torch.no_grad():
            roi_variance = torch.var(x, dim=-1)
            valid_mask = (roi_variance > 1e-5).float()
            if valid_mask.sum() == 0:
                valid_mask[0] = 1.0

        z = model.encoder(x)
        z = z * valid_mask.unsqueeze(-1)

        if model.cafr is not None and module.USE_MVCA:
            z_refined = model.cafr(z, valid_mask=valid_mask)
            z_refined = z_refined * valid_mask.unsqueeze(-1)
        else:
            z_refined = z

        if module.MU_AGGREGATION_MODE == "attn" and model.attn is not None:
            alpha, mu = model.attn(z_refined, valid_mask=valid_mask)
            quality = torch.norm(z_refined, p=2, dim=1)
            quality = quality / (quality.mean().detach() + 1e-8)
            quality = torch.clamp(quality, min=0.3, max=2.0)
            alpha_q = alpha * quality
            alpha_q = alpha_q / (alpha_q.sum() + 1e-8)
            spd = model.build_spd_second_moment(z_refined, alpha_q)
        else:
            v_count = valid_mask.sum().clamp(min=1.0)
            mu = z_refined.sum(dim=0) / v_count
            alpha = valid_mask / (valid_mask.sum() + 1e-8)
            alpha_q = alpha
            spd = model.build_spd_second_moment(z_refined, alpha_q)

        dispersion = (alpha_q * torch.sum((z_refined - mu.unsqueeze(0)) ** 2, dim=1)).sum()

        mu_list.append(mu)
        spd_list.append(spd)
        alpha_list.append(alpha.detach().cpu())
        alpha_spd_list.append(alpha_q.detach().cpu())
        z_refined_list.append(z_refined.detach().cpu())
        valid_mask_list.append(valid_mask.detach().cpu())
        dispersion_list.append(float(dispersion.detach().cpu()))

    mu_batch = torch.stack(mu_list, dim=0)
    spd_batch = torch.stack(spd_list, dim=0)
    log_spd = module.robust_logm_spd_batch(spd_batch, mode="cpu")
    v = module.spd_triu_vector(log_spd)

    if module.USE_MU_IN_CLASSIFIER:
        mu_projected = model.mu_projector(mu_batch)
        feat = torch.cat([mu_projected, v], dim=1)
        semantic_feat = mu_projected
    else:
        feat = v
        semantic_feat = mu_batch

    logits = model.head(feat)
    return logits, {
        "alpha": alpha_list,
        "alpha_spd": alpha_spd_list,
        "z_refined": z_refined_list,
        "valid_mask": valid_mask_list,
        "dispersion": dispersion_list,
        "mu": mu_batch.detach().cpu(),
        "spd": spd_batch.detach().cpu(),
        "log_spd": log_spd.detach().cpu(),
        "semantic_feat": semantic_feat.detach().cpu(),
        "structural_feat": v.detach().cpu(),
        "feat": feat.detach().cpu(),
    }


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(name))


def safe_binary_metrics(labels: List[int], preds: List[int], probs: List[float]) -> Dict[str, float]:
    if len(labels) == 0:
        return {"n": 0, "accuracy": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan, "auc": np.nan, "ap": np.nan}
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    out = {
        "n": len(labels),
        "accuracy": accuracy_score(labels, preds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": np.nan,
        "ap": np.nan,
    }
    if len(set(labels)) == 2:
        out["auc"] = roc_auc_score(labels, probs)
        out["ap"] = average_precision_score(labels, probs)
    return out


def save_roi_attention_heatmap(case_name: str, alpha: np.ndarray, out_path: Path):
    """Visualize ROI attention as a true heatmap over key-frame indices."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    alpha = np.asarray(alpha, dtype=np.float32)
    if alpha.size == 0:
        return

    fig_width = max(6.5, min(14.0, 0.55 * alpha.size + 2.0))
    fig, ax = plt.subplots(figsize=(fig_width, 2.4))
    im = ax.imshow(alpha.reshape(1, -1), cmap="YlOrRd", aspect="auto", vmin=0.0, vmax=max(float(alpha.max()), 1e-8))
    ax.set_yticks([0])
    ax.set_yticklabels(["Attention"])
    ax.set_xticks(np.arange(alpha.size))
    ax.set_xticklabels([str(i + 1) for i in range(alpha.size)], fontsize=8)
    ax.set_xlabel("Key-frame / ROI index")
    ax.set_title("ROIattentionheatmap", fontsize=11)

    for i, value in enumerate(alpha):
        text_color = "white" if value > alpha.max() * 0.55 else "black"
        ax.text(i, 0, f"{value:.2f}", ha="center", va="center", fontsize=7, color=text_color)

    cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.035, pad=0.02)
    cbar.set_label("Attention weight")
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def save_roi_attention_2x2_heatmap(case_name: str, alpha: np.ndarray, out_path: Path):
    """Visualize exactly four ROI attention weights as a 2 x 2 heatmap."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    alpha = np.asarray(alpha, dtype=np.float32).reshape(-1)
    if alpha.size != 4:
        return

    matrix = alpha.reshape(2, 2)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(
        matrix,
        cmap="Reds",
        aspect="equal",
        vmin=max(0.0, float(alpha.min()) * 0.95),
        vmax=max(float(alpha.max()), 1e-8),
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("ROIattentionheatmap", fontsize=12)

    for idx, value in enumerate(alpha):
        row, col = divmod(idx, 2)
        text_color = "white" if value > alpha.max() * 0.55 else "black"
        ax.text(
            col,
            row,
            f"ROI{idx + 1}\n{value:.3f}",
            ha="center",
            va="center",
            fontsize=11,
            color=text_color,
        )

    cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    cbar.set_label("Attention weight")
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def save_spd_heatmap(case_name: str, matrix: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(matrix, cmap="coolwarm", aspect="auto")
    ax.set_title(f"Attention-guided log-SPD heatmap: {case_name}", fontsize=11)
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel("Latent dimension")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def resample_attention(alpha: np.ndarray, n_bins: int) -> np.ndarray:
    """Resample variable-length ROI attention weights to fixed normalized bins."""
    alpha = np.asarray(alpha, dtype=np.float32).reshape(-1)
    if alpha.size == 0:
        return np.zeros(n_bins, dtype=np.float32)
    if alpha.size == 1:
        return np.full(n_bins, float(alpha[0]), dtype=np.float32)

    src_x = np.linspace(0.0, 1.0, alpha.size)
    dst_x = np.linspace(0.0, 1.0, n_bins)
    return np.interp(dst_x, src_x, alpha).astype(np.float32)


def save_average_roi_attention_heatmap(title: str, alpha_mean: np.ndarray, out_path: Path):
    """Visualize averaged ROI attention after normalization to fixed ROI-position bins."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    alpha_mean = np.asarray(alpha_mean, dtype=np.float32)
    if alpha_mean.size == 0:
        return

    fig_width = max(7.5, min(14.0, 0.42 * alpha_mean.size + 2.2))
    fig, ax = plt.subplots(figsize=(fig_width, 2.4))
    im = ax.imshow(
        alpha_mean.reshape(1, -1),
        cmap="YlOrRd",
        aspect="auto",
        vmin=0.0,
        vmax=max(float(alpha_mean.max()), 1e-8),
    )
    ax.set_yticks([0])
    ax.set_yticklabels(["Mean attention"])
    ax.set_xticks(np.arange(alpha_mean.size))
    ax.set_xticklabels([str(i + 1) for i in range(alpha_mean.size)], fontsize=8)
    ax.set_xlabel("Normalized ROI-position bin")
    ax.set_title("ROIattentionheatmap", fontsize=11)

    for i, value in enumerate(alpha_mean):
        text_color = "white" if value > alpha_mean.max() * 0.55 else "black"
        ax.text(i, 0, f"{value:.2f}", ha="center", va="center", fontsize=7, color=text_color)

    cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.035, pad=0.02)
    cbar.set_label("Mean attention weight")
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def save_roi_attention_barplot(case_name: str, alpha: np.ndarray, out_path: Path):
    """Save a quantitative bar plot of ROI-level attention weights."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    alpha = np.asarray(alpha, dtype=np.float32).reshape(-1)
    if alpha.size == 0:
        return

    fig_width = max(7.0, min(14.0, 0.45 * alpha.size + 2.5))
    fig, ax = plt.subplots(figsize=(fig_width, 3.2))
    x = np.arange(alpha.size)
    colors = plt.cm.YlOrRd(alpha / max(float(alpha.max()), 1e-8))
    ax.bar(x + 1, alpha, color=colors, edgecolor="#333333", linewidth=0.4)
    ax.set_xlabel("ROI index")
    ax.set_ylabel("Five-model mean attention weight")
    ax.set_title(f"ROI attention bar plot: {case_name}", fontsize=11)
    ax.set_xticks(x + 1)
    ax.set_xticklabels([str(i + 1) for i in x], fontsize=8)
    ax.set_ylim(0, max(float(alpha.max()) * 1.18, 1e-3))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)

    for i, value in enumerate(alpha):
        ax.text(i + 1, value, f"{value:.2f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def save_roi_attention_overlay_montage(
    case_name: str,
    roi_paths: List[str],
    alpha: np.ndarray,
    out_path: Path,
    n_cols: int = 5,
):
    """Overlay ROI-level attention weights on ROI thumbnails as translucent color blocks."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    alpha = np.asarray(alpha, dtype=np.float32).reshape(-1)
    if alpha.size == 0 or not roi_paths:
        return

    n = min(len(roi_paths), alpha.size)
    roi_paths = roi_paths[:n]
    alpha = alpha[:n]
    n_cols = max(1, int(n_cols))
    n_rows = int(np.ceil(n / n_cols))

    fig_width = min(3.0 * n_cols, 15.0)
    fig_height = 3.0 * n_rows + 0.6
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    axes = np.asarray(axes).reshape(-1)

    a_min = float(alpha.min())
    a_max = float(alpha.max())
    denom = max(a_max - a_min, 1e-8)
    norm_values = (alpha - a_min) / denom
    cmap = plt.cm.YlOrRd

    for idx, ax in enumerate(axes):
        ax.axis("off")
        if idx >= n:
            continue
        try:
            from PIL import Image

            with Image.open(roi_paths[idx]) as im:
                img = np.asarray(im.convert("L"))
        except Exception:
            img = np.zeros((160, 160), dtype=np.uint8)

        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        overlay = np.ones((*img.shape, 4), dtype=np.float32)
        overlay_color = cmap(norm_values[idx])
        overlay[..., 0] = overlay_color[0]
        overlay[..., 1] = overlay_color[1]
        overlay[..., 2] = overlay_color[2]
        overlay[..., 3] = 0.18 + 0.52 * float(norm_values[idx])
        ax.imshow(overlay)
        ax.set_title(f"ROI {idx + 1}: {alpha[idx]:.2f}", fontsize=8, pad=2)

    fig.suptitle(
        f"ROI-level attention overlay montage: {case_name}",
        fontsize=12,
        y=0.995,
    )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=a_min, vmax=a_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[:n], orientation="horizontal", fraction=0.035, pad=0.035)
    cbar.set_label("Five-model mean attention weight")
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def save_roi_index_mapping(
    roi_paths: List[str],
    alpha: np.ndarray,
    out_path: Path,
    source: str,
):
    """Export the correspondence between displayed ROI indices and original ROI files."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    alpha = np.asarray(alpha, dtype=np.float32).reshape(-1)
    n = min(len(roi_paths), alpha.size)
    rows: List[Dict[str, Any]] = []
    for idx in range(n):
        roi_path = Path(roi_paths[idx])
        rows.append(
            {
                "source": source,
                "roi_label": f"ROI{idx + 1}",
                "roi_index": idx + 1,
                "roi_file": roi_path.name,
                "roi_path": str(roi_path),
                "attention_weight": float(alpha[idx]),
            }
        )
    write_csv(
        out_path,
        rows,
        ["source", "roi_label", "roi_index", "roi_file", "roi_path", "attention_weight"],
    )


def plot_feature_embedding_panels(
    features_dict: Dict[str, np.ndarray],
    labels: np.ndarray,
    method: str,
    out_path: Path,
) -> List[Dict[str, Any]]:
    """Draw side-by-side t-SNE/UMAP plots for semantic, structural, and fused features."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_samples = int(labels.shape[0])
    if n_samples < 3:
        print(f"[Skip] Need at least 3 samples for {method.upper()} visualization, got {n_samples}.")
        return []

    if method == "umap":
        try:
            import umap  # type: ignore
        except Exception as exc:
            print(f"[Skip] UMAP is unavailable ({exc}). Install umap-learn or use --feature-embedding-method tsne.")
            return []

    fig, axes = plt.subplots(1, len(features_dict), figsize=(15, 4.6), dpi=300)
    if len(features_dict) == 1:
        axes = [axes]

    benign_mask = labels == 0
    malignant_mask = labels == 1
    score_rows: List[Dict[str, Any]] = []

    for ax, (title, features) in zip(axes, features_dict.items()):
        features = np.asarray(features, dtype=np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features = StandardScaler().fit_transform(features)

        if method == "tsne":
            perplexity = min(30, max(2, (n_samples - 1) // 3))
            embedding = TSNE(
                n_components=2,
                perplexity=perplexity,
                learning_rate=200.0,
                init="pca",
                random_state=42,
            ).fit_transform(features)
            xlabel, ylabel = "t-SNE 1", "t-SNE 2"
        else:
            reducer = umap.UMAP(  # type: ignore[name-defined]
                n_components=2,
                n_neighbors=min(15, max(2, n_samples - 1)),
                min_dist=0.1,
                metric="euclidean",
                random_state=42,
            )
            embedding = reducer.fit_transform(features)
            xlabel, ylabel = "UMAP 1", "UMAP 2"

        if len(np.unique(labels)) > 1 and n_samples > len(np.unique(labels)):
            sil = float(silhouette_score(embedding, labels))
            dbi = float(davies_bouldin_score(embedding, labels))
        else:
            sil = np.nan
            dbi = np.nan

        score_rows.append(
            {
                "method": method,
                "feature_type": title,
                "silhouette_score": sil,
                "davies_bouldin_index": dbi,
            }
        )

        ax.scatter(
            embedding[benign_mask, 0],
            embedding[benign_mask, 1],
            c="#2F80ED",
            s=18,
            alpha=0.78,
            label="Benign",
            edgecolors="none",
        )
        ax.scatter(
            embedding[malignant_mask, 0],
            embedding[malignant_mask, 1],
            c="#EB5757",
            s=18,
            alpha=0.78,
            label="Malignant",
            edgecolors="none",
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.22, linewidth=0.6)

    axes[0].legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return score_rows


def export_feature_distribution(
    feature_rows: List[Dict[str, Any]],
    out_dir: Path,
    embedding_method: str,
):
    """Save out-of-fold case-level features and draw feature distribution plots."""
    if not feature_rows:
        print("[Skip] No out-of-fold features were collected for feature distribution visualization.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    semantic_features = np.stack([np.asarray(r["semantic_feature"], dtype=np.float32) for r in feature_rows], axis=0)
    structural_features = np.stack([np.asarray(r["structural_feature"], dtype=np.float32) for r in feature_rows], axis=0)
    fused_features = np.stack([np.asarray(r["fused_feature"], dtype=np.float32) for r in feature_rows], axis=0)
    labels = np.asarray([int(r["label"]) for r in feature_rows], dtype=np.int64)
    preds = np.asarray([int(r["pred"]) for r in feature_rows], dtype=np.int64)
    probs = np.asarray([float(r["prob_malignant"]) for r in feature_rows], dtype=np.float32)
    folds = np.asarray([int(r["fold"]) for r in feature_rows], dtype=np.int64)
    case_names = np.asarray([str(r["case_name"]) for r in feature_rows], dtype=object)

    np.savez_compressed(
        out_dir / "case_level_features_for_embedding.npz",
        semantic_features=semantic_features,
        structural_features=structural_features,
        fused_features=fused_features,
        labels=labels,
        preds=preds,
        probs=probs,
        folds=folds,
        case_names=case_names,
    )
    write_csv(
        out_dir / "case_level_features_metadata.csv",
        [
            {
                "fold": int(r["fold"]),
                "case_name": r["case_name"],
                "label": int(r["label"]),
                "pred": int(r["pred"]),
                "prob_malignant": float(r["prob_malignant"]),
                "correct": int(r["label"] == r["pred"]),
            }
            for r in feature_rows
        ],
        ["fold", "case_name", "label", "pred", "prob_malignant", "correct"],
    )

    features_dict = {
        "Semantic branch": semantic_features,
        "Structural branch": structural_features,
        "Fused representation": fused_features,
    }
    methods = ["tsne", "umap"] if embedding_method == "both" else [embedding_method]
    score_rows: List[Dict[str, Any]] = []
    for method in methods:
        score_rows.extend(
            plot_feature_embedding_panels(
                features_dict,
                labels,
                method,
                out_dir / f"{method}_case_level_feature_distribution.png",
            )
        )
    if score_rows:
        write_csv(
            out_dir / "feature_embedding_separation_scores.csv",
            score_rows,
            ["method", "feature_type", "silhouette_score", "davies_bouldin_index"],
        )


def export_average_heatmaps(payloads: List[Dict[str, Any]], out_dir: Path, attention_bins: int):
    """Export fold-level means and the overall five-fold mean heatmaps."""
    if not payloads:
        return
    if attention_bins <= 1:
        raise ValueError("--attention-average-bins must be greater than 1.")

    out_dir.mkdir(parents=True, exist_ok=True)

    def export_subset(name: str, subset: List[Dict[str, Any]]):
        if not subset:
            return
        mean_attention = np.mean(
            [resample_attention(p["alpha"], attention_bins) for p in subset],
            axis=0,
        )
        mean_log_spd = np.mean([np.asarray(p["log_spd"], dtype=np.float32) for p in subset], axis=0)

        save_average_roi_attention_heatmap(
            f"Mean ROI attention heatmap: {name}",
            mean_attention,
            out_dir / f"{safe_name(name)}_mean_roi_attention_heatmap.png",
        )
        save_spd_heatmap(
            f"Mean log-SPD: {name}",
            mean_log_spd,
            out_dir / f"{safe_name(name)}_mean_log_spd_heatmap.png",
        )

    for fold in sorted({int(p["fold"]) for p in payloads}):
        export_subset(f"fold{fold}", [p for p in payloads if int(p["fold"]) == fold])
    export_subset("five_fold_overall", payloads)


def compute_five_model_case_mean(payloads: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, float]:
    """Average ROI attention, log-SPD matrix, and malignant probability across fold models."""
    if not payloads:
        raise ValueError("Cannot average an empty payload list.")

    lengths = [len(p["alpha"]) for p in payloads if len(p["alpha"]) > 0]
    if not lengths:
        raise ValueError("Cannot average payloads without valid attention weights.")

    target_len = max(lengths)
    if len(set(lengths)) == 1:
        mean_alpha = np.mean([np.asarray(p["alpha"], dtype=np.float32) for p in payloads], axis=0)
    else:
        mean_alpha = np.mean(
            [resample_attention(p["alpha"], target_len) for p in payloads],
            axis=0,
        )

    mean_log_spd = np.mean(
        [np.asarray(p["log_spd"], dtype=np.float32) for p in payloads],
        axis=0,
    )
    mean_prob = float(np.mean([p["prob_malignant"] for p in payloads]))
    return mean_alpha, mean_log_spd, mean_prob


def export_per_case_five_fold_average_heatmaps(
    payloads_by_case: Dict[str, List[Dict[str, Any]]],
    out_dir: Path,
    export_overlay_montage: bool = True,
    export_barplot: bool = True,
    montage_columns: int = 5,
):
    """Export one five-model averaged interpretation set for each case."""
    out_dir.mkdir(parents=True, exist_ok=True)

    for case_name, payloads in sorted(payloads_by_case.items()):
        if not payloads:
            continue
        try:
            mean_alpha, mean_log_spd, mean_prob = compute_five_model_case_mean(payloads)
        except ValueError:
            continue

        label = payloads[0]["label"]
        label_name = "malignant" if label == 1 else "benign"
        pred_name = "malignant" if mean_prob >= 0.5 else "benign"
        prefix = (
            f"label-{label_name}_"
            f"meanpred-{pred_name}_"
            f"models-{len(payloads)}_"
            f"{safe_name(case_name)}"
        )

        save_roi_attention_heatmap(
            f"{case_name} | five-model mean",
            mean_alpha,
            out_dir / f"{prefix}_five_fold_mean_roi_attention_heatmap.png",
        )
        save_spd_heatmap(
            f"{case_name} | five-model mean",
            mean_log_spd,
            out_dir / f"{prefix}_five_fold_mean_log_spd_heatmap.png",
        )
        if export_overlay_montage:
            roi_paths = payloads[0].get("roi_paths", [])
            save_roi_attention_overlay_montage(
                f"{case_name} | five-model mean",
                roi_paths,
                mean_alpha,
                out_dir / f"{prefix}_five_fold_mean_roi_attention_overlay_montage.png",
                n_cols=montage_columns,
            )
        if export_barplot:
            save_roi_attention_barplot(
                f"{case_name} | five-model mean",
                mean_alpha,
                out_dir / f"{prefix}_five_fold_mean_roi_attention_barplot.png",
            )


def choose_representatives(payloads: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    selected: Dict[str, Dict[str, Any]] = {}
    four_roi_payloads = [
        p for p in payloads
        if int(p.get("num_valid_rois", 0)) == 4
        and len(p.get("alpha", [])) == 4
    ]

    if four_roi_payloads:
        payloads = four_roi_payloads

    benign_correct = [p for p in payloads if p["label"] == 0 and p["pred"] == 0]
    if benign_correct:
        selected["benign_correct"] = max(benign_correct, key=lambda p: 1.0 - p["prob_malignant"])

    malignant_correct = [p for p in payloads if p["label"] == 1 and p["pred"] == 1]
    if malignant_correct:
        selected["malignant_correct"] = max(malignant_correct, key=lambda p: p["prob_malignant"])

    correct_cases = [p for p in payloads if p["label"] == p["pred"]]
    if correct_cases:
        selected["high_dispersion_correct"] = max(correct_cases, key=lambda p: p["dispersion"])

    false_cases = [p for p in payloads if p["label"] != p["pred"]]
    if false_cases:
        selected["false_case"] = max(false_cases, key=lambda p: p["dispersion"])

    return selected


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if not (0 < args.hard_ratio <= 1):
        raise ValueError("--hard-ratio must be in the interval (0, 1].")

    source_script = resolve_source_script(args.source)
    print(f"Using source script: {source_script}")

    module = load_training_module(source_script)
    out_root = module.ROI_OUTPUT_DIR / args.analysis_subdir
    out_root.mkdir(parents=True, exist_ok=True)

    labels_path, label_items = module.read_labels(module.BASE_ROOT)
    cases = [c for c, _ in label_items]
    labels = [y for _, y in label_items]

    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    case_dirs = np.array([module.ROI_OUTPUT_DIR / c for c in cases])
    labels_encoded = np.array(labels_encoded)

    skf = StratifiedKFold(n_splits=module.N_FOLDS, shuffle=True, random_state=module.RANDOM_STATE)
    all_records: List[Dict[str, Any]] = []
    all_payloads: List[Dict[str, Any]] = []
    feature_rows: List[Dict[str, Any]] = []
    fold_hard_metrics: List[Dict[str, Any]] = []
    per_case_model_payloads: Dict[str, List[Dict[str, Any]]] = {}

    for fold, (train_src_idx, test_idx) in enumerate(skf.split(case_dirs, labels_encoded)):
        ckpt_path = module.ROI_OUTPUT_DIR / f"model_fold_{fold + 1}.pth"
        if not ckpt_path.exists():
            print(f"[Skip] Missing checkpoint: {ckpt_path}")
            continue

        fold_train_src_dirs = case_dirs[train_src_idx]
        fold_train_src_labels = labels_encoded[train_src_idx]
        fold_test_dirs = case_dirs[test_idx].tolist()
        fold_test_labels = labels_encoded[test_idx].tolist()

        train_dirs, _, train_labels, _ = train_test_split(
            fold_train_src_dirs,
            fold_train_src_labels,
            test_size=module.VAL_SIZE,
            random_state=module.RANDOM_STATE,
            stratify=fold_train_src_labels,
        )

        scaler = fit_train_scaler(module, train_dirs.tolist(), max_rois=module.MAX_ROIS)
        test_ds = AnalysisDataset(module, fold_test_dirs, fold_test_labels, scaler=scaler, max_rois=module.MAX_ROIS)
        test_loader = DataLoader(
            test_ds,
            batch_size=module.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=analysis_collate_fn,
        )

        model = module.End2EndRadiomicsSPD(in_dim=182, z_dim=module.Z_DIM, attn_hidden=module.ATTN_HIDDEN, num_classes=len(le.classes_)).to(module.DEVICE)
        state = torch.load(ckpt_path, map_location=module.DEVICE)
        model.load_state_dict(state)
        model.eval()

        if args.export_per_case_five_fold_average_heatmaps:
            all_case_ds = AnalysisDataset(
                module,
                case_dirs.tolist(),
                labels_encoded.tolist(),
                scaler=scaler,
                max_rois=module.MAX_ROIS,
            )
            all_case_loader = DataLoader(
                all_case_ds,
                batch_size=module.BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                collate_fn=analysis_collate_fn,
            )

            with torch.no_grad():
                for batch in all_case_loader:
                    labels_tensor = batch["label"].to(module.DEVICE)
                    logits, details = model_forward_with_details(module, model, batch["roi_features"])
                    probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
                    preds = logits.argmax(dim=1).detach().cpu().numpy()
                    labels_np = labels_tensor.detach().cpu().numpy()

                    for i, case_name in enumerate(batch["case_name"]):
                        valid_mask = details["valid_mask"][i].numpy()
                        valid_n = int(valid_mask.sum())
                        alpha = details["alpha"][i].numpy()
                        alpha_valid = alpha[:valid_n] if valid_n > 0 else alpha

                        per_case_model_payloads.setdefault(case_name, []).append(
                            {
                                "fold_model": fold + 1,
                                "case_name": case_name,
                                "label": int(labels_np[i]),
                                "pred": int(preds[i]),
                                "prob_malignant": float(probs[i]),
                                "alpha": alpha_valid,
                                "log_spd": details["log_spd"][i].numpy(),
                                "roi_paths": batch["roi_paths"][i],
                            }
                        )

        fold_records: List[Dict[str, Any]] = []
        with torch.no_grad():
            for batch in test_loader:
                labels_tensor = batch["label"].to(module.DEVICE)
                logits, details = model_forward_with_details(module, model, batch["roi_features"])
                probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
                preds = logits.argmax(dim=1).detach().cpu().numpy()
                labels_np = labels_tensor.detach().cpu().numpy()

                for i, case_name in enumerate(batch["case_name"]):
                    valid_mask = details["valid_mask"][i].numpy()
                    valid_n = int(valid_mask.sum())
                    alpha = details["alpha"][i].numpy()
                    alpha_valid = alpha[:valid_n] if valid_n > 0 else alpha

                    rec = {
                        "fold": fold + 1,
                        "case_name": case_name,
                        "label": int(labels_np[i]),
                        "pred": int(preds[i]),
                        "prob_malignant": float(probs[i]),
                        "correct": int(labels_np[i] == preds[i]),
                        "num_valid_rois": valid_n,
                        "dispersion": float(details["dispersion"][i]),
                        "max_attention": float(np.max(alpha_valid)) if len(alpha_valid) else np.nan,
                        "attention_entropy": float(-np.sum(alpha_valid * np.log(alpha_valid + 1e-8))) if len(alpha_valid) else np.nan,
                    }
                    fold_records.append(rec)
                    all_records.append(rec)
                    all_payloads.append(
                        {
                            **rec,
                            "alpha": alpha_valid,
                            "log_spd": details["log_spd"][i].numpy(),
                            "roi_paths": batch["roi_paths"][i],
                        }
                    )
                    feature_rows.append(
                        {
                            "fold": fold + 1,
                            "case_name": case_name,
                            "label": int(labels_np[i]),
                            "pred": int(preds[i]),
                            "prob_malignant": float(probs[i]),
                            "semantic_feature": details["semantic_feat"][i].numpy(),
                            "structural_feature": details["structural_feat"][i].numpy(),
                            "fused_feature": details["feat"][i].numpy(),
                        }
                    )

        n_hard = max(1, int(np.ceil(len(fold_records) * args.hard_ratio)))
        hard_records = sorted(fold_records, key=lambda r: r["dispersion"], reverse=True)[:n_hard]
        hard_metrics = safe_binary_metrics(
            [r["label"] for r in hard_records],
            [r["pred"] for r in hard_records],
            [r["prob_malignant"] for r in hard_records],
        )
        hard_metrics.update({"fold": fold + 1, "selection": "top_dispersion", "hard_ratio": args.hard_ratio})
        fold_hard_metrics.append(hard_metrics)
        print(f"[Fold {fold + 1}] exported records: {len(fold_records)}, hard cases: {n_hard}")

    record_fields = [
        "fold",
        "case_name",
        "label",
        "pred",
        "prob_malignant",
        "correct",
        "num_valid_rois",
        "dispersion",
        "max_attention",
        "attention_entropy",
    ]
    write_csv(out_root / "all_case_predictions.csv", all_records, record_fields)

    if args.export_feature_distribution:
        export_feature_distribution(
            feature_rows,
            out_root / "feature_distribution",
            embedding_method=args.feature_embedding_method,
        )

    hard_fields = ["fold", "selection", "hard_ratio", "n", "accuracy", "precision", "recall", "f1", "auc", "ap"]
    write_csv(out_root / "fold_hard_subset_metrics.csv", fold_hard_metrics, hard_fields)

    summary = {}
    for key in ["accuracy", "precision", "recall", "f1", "auc", "ap"]:
        vals = [m[key] for m in fold_hard_metrics if not np.isnan(m[key])]
        summary[f"{key}_mean"] = float(np.mean(vals)) if vals else np.nan
        summary[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    summary["hard_ratio"] = args.hard_ratio
    summary["n_folds"] = len(fold_hard_metrics)
    write_csv(out_root / "hard_subset_summary.csv", [summary], list(summary.keys()))

    rep_dir = out_root / "representative_cases"
    selected_all: Dict[str, Dict[str, Any]] = {}

    def export_representative_payload(category: str, payload: Dict[str, Any], group_name: str):
        case_dir = rep_dir / f"fold_{payload['fold']}" / safe_name(payload["case_name"])
        prefix = f"{group_name}_{category}_fold{payload['fold']}"
        save_roi_attention_heatmap(
            payload["case_name"],
            payload["alpha"],
            case_dir / f"{prefix}_roi_attention_heatmap.png",
        )
        save_spd_heatmap(
            payload["case_name"],
            payload["log_spd"],
            case_dir / f"{prefix}_log_spd_heatmap.png",
        )
        if len(payload.get("alpha", [])) == 4:
            save_roi_attention_2x2_heatmap(
                payload["case_name"],
                payload["alpha"],
                case_dir / f"{prefix}_roi_attention_2x2_heatmap.png",
            )
        save_roi_index_mapping(
            payload.get("roi_paths", []),
            payload["alpha"],
            case_dir / f"{prefix}_roi_index_mapping.csv",
            source="single_fold",
        )
        mean_payloads = per_case_model_payloads.get(payload["case_name"], [])
        if mean_payloads:
            try:
                mean_alpha, mean_log_spd, mean_prob = compute_five_model_case_mean(mean_payloads)
            except ValueError:
                mean_alpha, mean_log_spd, mean_prob = None, None, None
            if mean_alpha is not None and mean_log_spd is not None:
                mean_pred_name = "malignant" if mean_prob >= 0.5 else "benign"
                mean_prefix = f"{category}_five_fold_mean_pred-{mean_pred_name}_models-{len(mean_payloads)}"
                save_roi_attention_heatmap(
                    f"{payload['case_name']} | five-fold mean",
                    mean_alpha,
                    case_dir / f"{mean_prefix}_roi_attention_heatmap.png",
                )
                save_spd_heatmap(
                    f"{payload['case_name']} | five-fold mean",
                    mean_log_spd,
                    case_dir / f"{mean_prefix}_log_spd_heatmap.png",
                )
                if len(mean_alpha) == 4:
                    save_roi_attention_2x2_heatmap(
                        f"{payload['case_name']} | five-fold mean",
                        mean_alpha,
                        case_dir / f"{mean_prefix}_roi_attention_2x2_heatmap.png",
                    )
                save_roi_index_mapping(
                    mean_payloads[0].get("roi_paths", []),
                    mean_alpha,
                    case_dir / f"{mean_prefix}_roi_index_mapping.csv",
                    source="five_fold_mean",
                )

    for fold in sorted({int(p["fold"]) for p in all_payloads}):
        fold_payloads = [p for p in all_payloads if int(p["fold"]) == fold]
        selected_fold = choose_representatives(fold_payloads)
        for category, payload in selected_fold.items():
            key = f"fold{fold}_{category}"
            selected_all[key] = payload
            export_representative_payload(category, payload, group_name=f"fold{fold}")

    selected_global = choose_representatives(all_payloads)
    for category, payload in selected_global.items():
        key = f"global_{category}"
        selected_all[key] = payload
        export_representative_payload(category, payload, group_name="global")

    if args.export_all_heatmaps:
        all_case_fig_dir = out_root / "all_case_heatmaps"
        for payload in all_payloads:
            correctness = "correct" if payload["label"] == payload["pred"] else "wrong"
            label_name = "malignant" if payload["label"] == 1 else "benign"
            pred_name = "malignant" if payload["pred"] == 1 else "benign"
            prefix = (
                f"fold{payload['fold']}_"
                f"label-{label_name}_"
                f"pred-{pred_name}_"
                f"{correctness}_"
                f"{safe_name(payload['case_name'])}"
            )

            save_roi_attention_heatmap(
                payload["case_name"],
                payload["alpha"],
                all_case_fig_dir / f"{prefix}_roi_attention_heatmap.png",
            )
            save_spd_heatmap(
                payload["case_name"],
                payload["log_spd"],
                all_case_fig_dir / f"{prefix}_log_spd_heatmap.png",
            )

    if args.export_average_heatmaps:
        export_average_heatmaps(
            all_payloads,
            out_root / "average_heatmaps",
            attention_bins=args.attention_average_bins,
        )

    if args.export_per_case_five_fold_average_heatmaps:
        export_per_case_five_fold_average_heatmaps(
            per_case_model_payloads,
            out_root / "per_case_five_fold_average_heatmaps",
            export_overlay_montage=args.export_per_case_attention_overlay_montage,
            export_barplot=args.export_per_case_attention_barplot,
            montage_columns=args.roi_montage_columns,
        )

    # Save a copy of this analysis script for reproducibility.
    try:
        shutil.copy2(Path(__file__), out_root / "export_analysis_from_trained_models.py")
    except Exception:
        pass

    print(f"\nDone. Analysis outputs saved to: {out_root}")
    print("Representative cases:", ", ".join(selected_all.keys()) if selected_all else "none")


if __name__ == "__main__":
    main()

