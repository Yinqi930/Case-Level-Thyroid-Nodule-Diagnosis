
"""
End-to-End Radiomics + CAFR + SPD (Upgraded with Multi-Head Q-K Attention & Focal Loss)
Maintains full feature extraction pipeline (182 dims: GLCM, LBP, Gabor included).
Complete Implementation with 5-Fold Cross Validation.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
from PIL import Image

import scipy
from scipy.stats import skew, kurtosis, entropy
from scipy.ndimage import sobel

from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import regionprops
from skimage.filters import threshold_otsu
from skimage.measure import label as cc_label
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    f1_score
)

# ===================== Config =====================
BASE_ROOT = Path("/data1/syq/HYH/Dataset/Work3")
ROI_OUTPUT_DIR = BASE_ROOT / "roi_output"
ROI_STANDARD_SIZE = (224, 224)

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 2e-4
TEST_SIZE = 0.2
VAL_SIZE = 0.1
RANDOM_STATE = 30
N_FOLDS = 5  

Z_DIM = 64 # 64
ATTN_HIDDEN = 128 # 128
ATTN_TEMP = 0.5  # 1.5
MAX_ROIS = 20  #20


SCALER_CLIP = 8.0


SPD_EPS = 3e-2
EIG_MIN = 1e-6
EIG_MAX = 1e6


GRAD_CLIP_NORM = 5.0
ALLOW_PNG = True

USE_MVCA = True            
USE_MU_IN_CLASSIFIER = True   
USE_REEIG = True
USE_BIMAP = True


MU_AGGREGATION_MODE = "attn"    #mean/attn

AUX_SPD_LOSS_WEIGHT = 0.02    
AUX_MARGIN = 1.0              
AUX_TRIPLETS_PER_ANCHOR = 2   
AUX_MIN_POS_PER_BATCH = 2     
AUX_MIN_NEG_PER_BATCH = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"====================================================")
print(f"【高阶配置】当前多视图特征聚合模式: {MU_AGGREGATION_MODE} (Multi-Head Q-K)")
print(f"【消融配置】当前是否使用 MVCA 模块: {USE_MVCA}")
print(f"使用设备：{DEVICE}")
print(f"====================================================")

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ===================== Labels =====================
def read_labels(base: Path) -> Tuple[Path, List[Tuple[str, int]]]:
    p1 = base / "roi_output" / "labels.txt"
    p2 = base / "labels.txt"
    labels_path = p1 if p1.exists() else p2

    lines: List[Tuple[str, int]] = []
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split(",")
            if len(parts) != 2:
                continue
            name = parts[0].strip()
            label = int(parts[1])
            lines.append((name, label))
    return labels_path, lines


# ===================== IO utils =====================
def _img_exts() -> List[str]:
    exts = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]
    if ALLOW_PNG:
        exts += ["*.png", "*.PNG"]
    return exts


def list_images_in_case_dir(case_dir: Path) -> List[Path]:
    imgs: List[Path] = []
    for pat in _img_exts():
        imgs.extend(case_dir.glob(pat))
    return sorted(imgs)



def rgb_hist_simple(img: np.ndarray, bins: int = 32) -> np.ndarray:
    h = []
    for c in range(3):
        hist, _ = np.histogram(img[..., c].ravel(), bins=bins, range=(0, 255), density=True)
        h.append(hist.astype(np.float32))
    return np.concatenate(h, axis=0)


def sobel_mag_hist_simple(gray: np.ndarray, bins: int = 32) -> np.ndarray:
    sx = sobel(gray, axis=0, mode='reflect')
    sy = sobel(gray, axis=1, mode='reflect')
    mag = np.hypot(sx, sy)
    max_val = np.percentile(mag, 99)
    hist, _ = np.histogram(mag.ravel(), bins=bins, range=(0, max_val + 1e-6), density=True)
    return hist.astype(np.float32)


def lbp_hist_simple(gray: np.ndarray) -> np.ndarray:
    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, density=True)
    return hist.astype(np.float32)


def extract_glcm_features_simple(gray: np.ndarray) -> np.ndarray:
    gray_scaled = (gray / 16).astype(np.uint8)
    try:
        glcm = graycomatrix(
            gray_scaled, distances=[1], angles=[0, np.pi / 2],
            levels=16, symmetric=True, normed=True
        )
        feats = []
        for prop in ['contrast', 'energy', 'correlation', 'homogeneity']:
            feats.extend(graycoprops(glcm, prop).flatten().tolist())
        return np.array(feats, dtype=np.float32)
    except Exception:
        return np.zeros(8, dtype=np.float32)


def _largest_cc(binary: np.ndarray) -> np.ndarray:
    lab = cc_label(binary > 0)
    if lab.max() == 0:
        return binary.astype(np.uint8)
    areas = [(lab == i).sum() for i in range(1, lab.max() + 1)]
    k = int(np.argmax(areas) + 1)
    return (lab == k).astype(np.uint8)


def extract_shape_features_simple(gray: np.ndarray) -> np.ndarray:
    try:
        thresh = threshold_otsu(gray)
        binary = (gray > thresh).astype(np.uint8)
        binary = _largest_cc(binary)
        if binary.sum() == 0:
            raise ValueError("No foreground pixels")
        props = regionprops(binary)[0]
        shape_features = [
            props.area, props.perimeter, props.eccentricity, props.solidity,
            props.extent, props.major_axis_length, props.minor_axis_length,
            props.equivalent_diameter,
            props.moments_hu[0], props.moments_hu[1], props.moments_hu[2],
            props.moments_hu[3], props.moments_hu[4], props.moments_hu[5],
            props.moments_hu[6],
        ]
        return np.array(shape_features, dtype=np.float32)
    except Exception:
        return np.zeros(15, dtype=np.float32)


def extract_gray_statistics_simple(gray: np.ndarray) -> np.ndarray:
    try:
        mean_v = np.mean(gray)
        std_v = np.std(gray)
        var_v = np.var(gray)
        median_v = np.median(gray)
        q25 = np.percentile(gray, 25)
        q75 = np.percentile(gray, 75)
        sk = skew(gray.ravel())
        kt = kurtosis(gray.ravel())
        hist = np.histogram(gray, bins=64)[0].astype(np.float64)
        entr = entropy(hist + 1e-8)
        return np.array([mean_v, std_v, var_v, median_v, q25, q75, sk, kt, entr], dtype=np.float32)
    except Exception:
        return np.zeros(9, dtype=np.float32)


def extract_gabor_features_simple(gray: np.ndarray) -> np.ndarray:
    try:
        feats = []
        frequencies = [0.2, 0.4]
        orientations = [0, np.pi / 2]
        for freq in frequencies:
            for theta in orientations:
                sigma = 2 * np.pi * freq
                kernel = np.real(scipy.signal.gabor_kernel(freq, theta=theta, sigma_x=sigma, sigma_y=sigma))
                filtered = scipy.ndimage.convolve(gray, kernel, mode='reflect')
                feats.extend([np.mean(filtered), np.std(filtered), np.max(filtered)])
        return np.array(feats, dtype=np.float32)
    except Exception:
        return np.zeros(12, dtype=np.float32)


def extract_radiomics_features_simplified(img: Image.Image) -> np.ndarray:
    img = img.resize(ROI_STANDARD_SIZE, Image.Resampling.LANCZOS)
    arr = np.asarray(img.convert("RGB"))
    gray = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.uint8)

    feature_modules = [
        rgb_hist_simple(arr),                
        sobel_mag_hist_simple(gray),          
        lbp_hist_simple(gray),                
        extract_glcm_features_simple(gray),   
        extract_shape_features_simple(gray),  
        extract_gray_statistics_simple(gray), 
        extract_gabor_features_simple(gray),  
    ]
    all_features = np.concatenate(feature_modules, axis=0).astype(np.float32)
    all_features = np.nan_to_num(all_features, nan=0.0, posinf=1e6, neginf=-1e6)
    if all_features.shape[0] != 182:
        if all_features.shape[0] < 182:
            all_features = np.pad(all_features, (0, 182 - all_features.shape[0]), mode="constant")
        else:
            all_features = all_features[:182]
    return all_features


def precompute_case_roi_features(case_dir: Path, max_rois: int = MAX_ROIS) -> np.ndarray:
    imgs = list_images_in_case_dir(case_dir)
    roi_feats: List[np.ndarray] = []
    if imgs:
        for p in imgs[:max_rois]:
            try:
                with Image.open(p) as im:
                    roi_feats.append(extract_radiomics_features_simplified(im))
            except Exception:
                continue
    if not roi_feats:
        roi_feats = [np.zeros(182, dtype=np.float32)]
    return np.stack(roi_feats).astype(np.float32)


def fit_train_scaler(train_case_dirs: List[Path], max_rois: int = MAX_ROIS) -> StandardScaler:
    feats_all: List[np.ndarray] = []
    print("在训练集上拟合 StandardScaler...")
    for d in train_case_dirs:
        X = precompute_case_roi_features(d, max_rois=max_rois)
        feats_all.append(X)
    X_all = np.concatenate(feats_all, axis=0)
    scaler = StandardScaler()
    scaler.fit(X_all)
    return scaler



class ThyroidRadiomicsDataset(Dataset):
    def __init__(self, case_dirs: List[Path], labels: List[int], scaler: StandardScaler, max_rois: int = MAX_ROIS):
        self.case_dirs = case_dirs
        self.labels = labels
        self.scaler = scaler
        self.max_rois = max_rois
        self.data = self._precompute_all()

    def _precompute_all(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for d, y in zip(self.case_dirs, self.labels):
            X = precompute_case_roi_features(d, max_rois=self.max_rois)
            X = self.scaler.transform(X).astype(np.float32)
            X = np.clip(X, -SCALER_CLIP, SCALER_CLIP)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            out.append({
                "roi_features": torch.from_numpy(X),
                "label": torch.tensor(int(y), dtype=torch.long),
            })
        return out

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    labels = torch.stack([item["label"] for item in batch])
    roi_features = [item["roi_features"] for item in batch]
    return {"roi_features": roi_features, "label": labels}


def sanitize_spd_batch(spd: torch.Tensor, eps: float = SPD_EPS) -> torch.Tensor:
    spd = 0.5 * (spd + spd.transpose(-1, -2))
    spd = torch.nan_to_num(spd, nan=0.0, posinf=0.0, neginf=0.0)
    B, d, _ = spd.shape
    I = torch.eye(d, device=spd.device, dtype=spd.dtype).unsqueeze(0)
    tr = torch.diagonal(spd, dim1=-2, dim2=-1).sum(-1)
    scale = torch.clamp(tr / (d + 1e-8), min=1e-6, max=1e6).view(B, 1, 1)
    spd = spd / scale + eps * I
    return 0.5 * (spd + spd.transpose(-1, -2))


def robust_logm_spd_batch(spd: torch.Tensor, mode: str = "cpu") -> torch.Tensor:
    spd = sanitize_spd_batch(spd, eps=SPD_EPS)
    spd_dtype = spd.dtype
    spd_device = spd.device

    def _logm_on(t: torch.Tensor) -> torch.Tensor:
        w, V = torch.linalg.eigh(t)
        w = torch.clamp(w, min=EIG_MIN, max=EIG_MAX)
        out = V @ torch.diag_embed(torch.log(w)) @ V.transpose(-1, -2)
        out = 0.5 * (out + out.transpose(-1, -2))
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    if mode == "auto":
        B, d, _ = spd.shape
        I = torch.eye(d, device=spd_device, dtype=spd_dtype).unsqueeze(0)
        jitter = 1e-6
        for _ in range(8):
            try:
                return _logm_on(spd + jitter * I)
            except Exception:
                jitter *= 10.0

    spd_cpu = spd.detach().to("cpu", dtype=torch.float64)
    jitter = 1e-6
    I_cpu = torch.eye(spd_cpu.shape[-1], device="cpu", dtype=torch.float64).unsqueeze(0)
    for _ in range(14):
        try:
            out_cpu = _logm_on(spd_cpu + jitter * I_cpu)
            return out_cpu.to(device=spd_device, dtype=spd_dtype)
        except Exception:
            jitter *= 10.0
            if jitter > 1.0:
                break

    return torch.zeros_like(spd)


def spd_triu_vector(spd: torch.Tensor) -> torch.Tensor:
    B, d, _ = spd.shape
    idx = torch.triu_indices(d, d, device=spd.device)
    row, col = idx[0], idx[1]
    v = spd[:, row, col]
    off = row != col
    v[:, off] = v[:, off] * np.sqrt(2.0)
    return v


class MultiViewContextAlignment(nn.Module):
    def __init__(self, dim: int, num_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
        T, d = x.shape
        if T <= 1:
            return x

        residual = x
        x_norm = self.norm1(x)
        
        q = self.q_proj(x_norm).view(T, self.num_heads, self.head_dim).transpose(0, 1) 
        k = self.k_proj(x_norm).view(T, self.num_heads, self.head_dim).transpose(0, 1) 
        v = self.v_proj(x_norm).view(T, self.num_heads, self.head_dim).transpose(0, 1) 
        
        scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.head_dim)
        
        if valid_mask is not None:
            mask_expanded = valid_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, 1, T]
            scores = scores.masked_fill(mask_expanded == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        
        context = torch.matmul(attn_weights, v) 
        context = context.transpose(0, 1).contiguous().view(T, d)
        
        x = residual + self.dropout(self.out_proj(context))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class MultiHeadQueryKeyAttention(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_heads: int = 8, temperature: float = ATTN_TEMP):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        
        self.key_proj = nn.Linear(in_dim, hidden_dim)
        self.query_heads = nn.Parameter(torch.randn(num_heads, self.head_dim))
        self.log_temperature = nn.Parameter(torch.tensor(np.log(float(max(temperature, 1e-6)))))
        self.head_weight = nn.Parameter(torch.ones(num_heads))
        
        nn.init.orthogonal_(self.key_proj.weight)
        nn.init.orthogonal_(self.query_heads)

    def forward(self, z: torch.Tensor, valid_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        T, d = z.shape
        k = self.key_proj(z).view(T, self.num_heads, self.head_dim).transpose(0, 1) 
        
        temp = torch.exp(self.log_temperature).clamp(min=1e-3, max=10.0)
        q = self.query_heads.unsqueeze(1) # [num_heads, 1, head_dim]
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
        
        scores = (k * q).sum(dim=-1) / (np.sqrt(self.head_dim) * temp) # [num_heads, T]
        
        if valid_mask is not None:
            scores = scores.masked_fill(valid_mask.unsqueeze(0) == 0, -1e9)
            
        head_alpha = F.softmax(
            scores,
            dim=-1
        )
        head_weight = F.softmax(
            self.head_weight,
            dim=0
        )
        alpha = (
            head_alpha *
            head_weight.unsqueeze(-1)
        ).sum(dim=0)
        
        if valid_mask is not None:
            alpha = alpha * valid_mask
            alpha_sum = alpha.sum() + 1e-8
            alpha = alpha / alpha_sum
            
        pooled = (alpha.unsqueeze(-1) * z).sum(dim=0)
        return alpha, pooled


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.5, label_smoothing: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        with torch.no_grad():
            smooth_targets = torch.zeros_like(logits)
            smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        
        focal_weight = torch.pow(1.0 - probs, self.gamma)
        

        alpha = self.alpha.to(logits.device) if self.alpha is not None else 1.0
        
        loss = - (alpha * smooth_targets * focal_weight * log_probs).sum(dim=-1)
        return loss.mean()


# ===================== Aux loss: SPD alignment =====================
def spd_distance_from_log(logA: torch.Tensor, logB: torch.Tensor) -> torch.Tensor:
    D = logA - logB
    D = 0.5 * (D + D.t())
    v = spd_triu_vector(D.unsqueeze(0)).squeeze(0)
    return torch.linalg.norm(v, ord=2)


def aux_spd_triplet_loss(log_spd_batch: torch.Tensor, labels: torch.Tensor,
                         margin: float = AUX_MARGIN,
                         triplets_per_anchor: int = AUX_TRIPLETS_PER_ANCHOR) -> torch.Tensor:
    B = labels.shape[0]
    if B < 3:
        return torch.tensor(0.0, device=labels.device)

    labels_np = labels.detach().cpu().numpy().tolist()
    idx_by_class: Dict[int, List[int]] = {}
    for i, y in enumerate(labels_np):
        idx_by_class.setdefault(int(y), []).append(i)

    has_pos = any(len(v) >= AUX_MIN_POS_PER_BATCH for v in idx_by_class.values())
    has_neg = (len(idx_by_class.keys()) >= 2)
    if (not has_pos) or (not has_neg):
        return torch.tensor(0.0, device=labels.device)

    losses = []
    rng = np.random.default_rng(RANDOM_STATE + 123)

    for a in range(B):
        ya = labels_np[a]
        pos_pool = idx_by_class.get(int(ya), [])
        neg_pool = [j for c, v in idx_by_class.items() if c != int(ya) for j in v]
        if len(pos_pool) < 2 or len(neg_pool) < AUX_MIN_NEG_PER_BATCH:
            continue

        pos_candidates = [p for p in pos_pool if p != a]
        if not pos_candidates:
            continue

        for _ in range(triplets_per_anchor):
            p = int(rng.choice(pos_candidates))
            n = int(rng.choice(neg_pool))

            d_pos = spd_distance_from_log(log_spd_batch[a], log_spd_batch[p])
            d_neg = spd_distance_from_log(log_spd_batch[a], log_spd_batch[n])
            losses.append(F.relu(d_pos - d_neg + margin))

    if not losses:
        return torch.tensor(0.0, device=labels.device)

    return torch.stack(losses).mean()


# ===================== Model =====================
class End2EndRadiomicsSPD(nn.Module):
    def __init__(self, in_dim: int = 182, z_dim: int = Z_DIM, attn_hidden: int = ATTN_HIDDEN, num_classes: int = 2):
        super().__init__()
        self.z_dim = z_dim
        self.triu_dim = z_dim * (z_dim + 1) // 2
        
        if USE_BIMAP:
            self.encoder = nn.Sequential(
            nn.Linear(in_dim,256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(256,z_dim),
            nn.LayerNorm(z_dim)
        )
        else:
            self.encoder = nn.Sequential(nn.Linear(in_dim,z_dim))

        if USE_MVCA:
            self.cafr = MultiViewContextAlignment(dim=z_dim, num_heads=2, dropout=0.1)
        else:
            self.cafr = None 

        if MU_AGGREGATION_MODE == "attn":
            self.attn = MultiHeadQueryKeyAttention(in_dim=z_dim, hidden_dim=attn_hidden, num_heads=4, temperature=ATTN_TEMP)
        else:
            self.attn = None

        if USE_MU_IN_CLASSIFIER:
            self.mu_project_dim = 128 
            self.mu_projector = nn.Sequential(
                nn.Linear(z_dim, self.mu_project_dim),
                nn.LayerNorm(self.mu_project_dim),
                nn.GELU(),
                nn.Dropout(0.20)
            )
            self.fuse_dim = self.mu_project_dim + self.triu_dim

            # self.fuse_dim = z_dim
        else:
            self.fuse_dim = self.triu_dim
        


        self.head = nn.Sequential(
            nn.Linear(self.fuse_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(256, num_classes),
        )
        

        self._init_weights()

        self.valid_roi_count = 0
        self.total_roi_count = 0

    def _init_weights(self):
        for m in self.encoder:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        if USE_MU_IN_CLASSIFIER:
            for m in self.mu_projector:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def build_spd_second_moment(self, z: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        alpha = alpha / (alpha.sum() + 1e-8)
        M = torch.einsum("t,ti,tj->ij", alpha, z, z)
        M = 0.5 * (M + M.t())

        tr = torch.trace(M)
        scale = tr / (self.z_dim + 1e-8)
        scale = torch.clamp(scale, min=1e-6, max=1e6)
        M = M / (scale + 1e-8)

        if USE_REEIG:
            M = M + SPD_EPS * torch.eye(
            self.z_dim,
            device=z.device,
            dtype=z.dtype
        )
            
        M = 0.5 * (M + M.t())
        M = torch.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
        return M

    def forward(self, roi_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu_list, spd_list = [], []

        for X in roi_list:
            X = X.to(DEVICE)
            
            with torch.no_grad():

                roi_variance = torch.var(X, dim=-1)

                valid_mask = (roi_variance > 1e-5).float()
                self.valid_roi_count += valid_mask.sum().item()
                self.total_roi_count += valid_mask.numel()
                if valid_mask.sum() == 0:
                    valid_mask[0] = 1.0


            z = self.encoder(X)     
            

            z = z * valid_mask.unsqueeze(-1)
            
            if self.cafr is not None and USE_MVCA:
                z_refined = self.cafr(z, valid_mask=valid_mask)
                z_refined = z_refined * valid_mask.unsqueeze(-1)
            else:
                z_refined = z  


            if MU_AGGREGATION_MODE == "attn" and self.attn is not None:
                alpha, mu = self.attn(z_refined, valid_mask=valid_mask)
                # ROI Quality Score
                quality = torch.norm(z_refined, p=2, dim=1)
                quality = quality / (quality.mean().detach() + 1e-8)
                quality = torch.clamp( quality, min=0.3, max=2.0)
                alpha_q = alpha * quality
                alpha_q = alpha_q / (alpha_q.sum() + 1e-8)
                spd = self.build_spd_second_moment(z_refined, alpha_q)
            else:

                v_count = valid_mask.sum().clamp(min=1.0)
                mu = z_refined.sum(dim=0) / v_count
                
                alpha_for_spd = valid_mask / (valid_mask.sum() + 1e-8)
                spd = self.build_spd_second_moment(z_refined, alpha_for_spd)

            mu_list.append(mu)
            spd_list.append(spd)

        mu_batch = torch.stack(mu_list, dim=0)          
        spd_batch = torch.stack(spd_list, dim=0)        

        log_spd = robust_logm_spd_batch(spd_batch, mode="cpu")
        v = spd_triu_vector(log_spd)
        
        if USE_MU_IN_CLASSIFIER:

            mu_projected = self.mu_projector(mu_batch)
            feat = torch.cat([mu_projected, v], dim=1)
        else:
            feat = v

        # feat = mu_batch

        logits = self.head(feat)
        return logits, mu_batch, spd_batch, log_spd


# ===================== Train / Eval =====================
def train_model(fold, model, train_loader, val_loader, ce_criterion, optimizer, epochs: int):
    best_val_score = 0.0
    best_state = None

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    for epoch in range(epochs):
        model.valid_roi_count = 0
        model.total_roi_count = 0
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0

        for batch in train_loader:
            feats = batch["roi_features"]
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits, _, _, log_spd = model(feats)

            loss_ce = ce_criterion(logits, labels)
            loss_aux = aux_spd_triplet_loss(log_spd, labels, margin=AUX_MARGIN,
                                            triplets_per_anchor=AUX_TRIPLETS_PER_ANCHOR)
            loss = loss_ce + AUX_SPD_LOSS_WEIGHT * loss_aux

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            tr_loss += loss.item() * labels.size(0)
            pred = logits.argmax(dim=1)
            tr_total += labels.size(0)
            tr_correct += (pred == labels).sum().item()

        tr_acc = tr_correct / max(tr_total, 1)
        tr_loss = tr_loss / max(tr_total, 1)

        model.eval()
        va_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []
        
        with torch.no_grad():
            for batch in val_loader:
                feats = batch["roi_features"]
                labels = batch["label"].to(DEVICE)
                logits, _, _, log_spd = model(feats)
                
                loss_ce = ce_criterion(logits, labels)
                loss_aux = aux_spd_triplet_loss(log_spd, labels, margin=AUX_MARGIN,
                                                triplets_per_anchor=AUX_TRIPLETS_PER_ANCHOR)
                loss = loss_ce + AUX_SPD_LOSS_WEIGHT * loss_aux

                va_loss += loss.item() * labels.size(0)
                probs = F.softmax(logits, dim=1)
                pred = logits.argmax(dim=1)
                
                all_preds.extend(pred.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
                all_probs.extend(probs[:, 1].cpu().numpy().tolist())

        va_total = max(len(all_labels), 1)
        va_loss = va_loss / va_total
        
        try:
            val_auc = roc_auc_score(all_labels, all_probs)
        except Exception:
            val_auc = 0.5
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        
        val_score = 0.5 * val_auc + 0.5 * val_f1

        if val_score > best_val_score:
            best_val_score = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        scheduler.step()
        roi_ratio = (model.valid_roi_count /(model.total_roi_count + 1e-8))

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:

            print(
                f"ROI保留率: {roi_ratio:.4f}"
            )
            print(
                f"Fold [{fold+1}] Epoch [{epoch+1}/{epochs}] "
                f"| Train Loss: {tr_loss:.4f} "
                f"Acc: {tr_acc:.4f} "
                f"| Val Score: {val_score:.4f} "
                f"(AUC: {val_auc:.4f}, F1: {val_f1:.4f})"
            )


    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_score


def evaluate_model(model, data_loader) -> Dict[str, float]:
    model.eval()
    all_preds, all_labels = [], []
    all_probs = []

    with torch.no_grad():
        for batch in data_loader:
            feats = batch["roi_features"]
            labels = batch["label"].to(DEVICE)
            logits, _, _, _ = model(feats)

            probs = F.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    ap = average_precision_score(all_labels, all_probs)


    prec_binary, rec_binary, f1_binary, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average='binary'
    )



    print(
        f"F1={f1_binary:.4f}",
        f"prec={prec_binary:.4f}",
        f"rec={rec_binary:.4f}"
    )   


    return {
        "accuracy": acc,
        "prec": prec_binary,
        "rec": rec_binary,
        "f1": f1_binary,   
        "auc": auc,
        "ap": ap
    }


# ===================== Main 5-Fold CV =====================
def main():
    print("加载标签数据...")
    labels_path, label_items = read_labels(BASE_ROOT)
    print(f"labels.txt 路径：{labels_path}")

    cases = [c for c, _ in label_items]
    labels = [y for _, y in label_items]

    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    num_classes = len(le.classes_)
    print(f"类别数：{num_classes}，分层映射：{le.classes_}")
    print("类别分布：", np.bincount(labels_encoded))

    case_dirs = np.array([ROI_OUTPUT_DIR / c for c in cases])
    labels_encoded = np.array(labels_encoded)

    assert all([d.exists() for d in case_dirs]), "部分病例目录不存在，请检查路径结构。"

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = []

    print(f"\n==================== 开始 {N_FOLDS} 折嵌套交叉验证 (已升级高阶架构) ====================")
    for fold, (train_src_idx, test_idx) in enumerate(skf.split(case_dirs, labels_encoded)):
        print(f"\n---正在执行第 {fold+1} 折 / 共 {N_FOLDS} 折---")
        
        fold_train_src_dirs = case_dirs[train_src_idx]
        fold_train_src_labels = labels_encoded[train_src_idx]
        
        fold_test_dirs = case_dirs[test_idx].tolist()
        fold_test_labels = labels_encoded[test_idx].tolist()

        train_dirs, val_dirs, train_labels, val_labels = train_test_split(
            fold_train_src_dirs, 
            fold_train_src_labels, 
            test_size=VAL_SIZE, 
            random_state=RANDOM_STATE, 
            stratify=fold_train_src_labels
        )
        
        train_dirs_f, train_labels_f = train_dirs.tolist(), train_labels.tolist()
        val_dirs_f, val_labels_f = val_dirs.tolist(), val_labels.tolist()

        scaler = fit_train_scaler(train_dirs_f, max_rois=MAX_ROIS)
        
        train_ds = ThyroidRadiomicsDataset(train_dirs_f, train_labels_f, scaler=scaler, max_rois=MAX_ROIS)
        val_ds = ThyroidRadiomicsDataset(val_dirs_f, val_labels_f, scaler=scaler, max_rois=MAX_ROIS)
        test_ds = ThyroidRadiomicsDataset(fold_test_dirs, fold_test_labels, scaler=scaler, max_rois=MAX_ROIS)

        train_y = np.array(train_labels_f, dtype=np.int64)
        class_counts = np.bincount(train_y)
        
        p_factor = 0.5
        class_w = 1.0 / np.power(np.maximum(class_counts, 1), p_factor)
        
        sample_w = class_w[train_y]
        sampler = WeightedRandomSampler(weights=torch.tensor(sample_w, dtype=torch.double), num_samples=len(sample_w), replacement=True)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=True, collate_fn=custom_collate_fn)

        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True, collate_fn=custom_collate_fn)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True, collate_fn=custom_collate_fn)

        model = End2EndRadiomicsSPD(in_dim=182, z_dim=Z_DIM, attn_hidden=ATTN_HIDDEN, num_classes=num_classes).to(DEVICE)
        

        normed_class_weights = class_w / np.sum(class_w) * num_classes
        alpha_tensor = torch.tensor([normed_class_weights], dtype=torch.float32) 
        

        ce_criterion = FocalLoss(alpha=alpha_tensor, gamma=2.5, label_smoothing=0.0)
        
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)

        model, _ = train_model(fold, model, train_loader, val_loader, ce_criterion, optimizer, EPOCHS)

        metrics = evaluate_model(model, test_loader)
        fold_metrics.append(metrics)
        
        ROI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ROI_OUTPUT_DIR / f"model_fold_{fold+1}.pth")


    print("\n" + "=" * 25 + " 5折 independent独立测试最终指标汇总 " + "=" * 25)
    print(f"{'折数':<8}{'准确率(Acc)':<12}{'精确率(Prec)':<12}{'召回率(Rec)':<12}{'F1 分数':<12}{'AUC':<12}{'AP':<12}")
    print("-" * 85)
    
    accs, prec, rec, f1, aucs, aps = [], [], [], [], [], []
    for i, m in enumerate(fold_metrics):
        print(f"Fold {i+1:<4}{m['accuracy']:.4f}      {m['prec']:.4f}      {m['rec']:.4f}      {m['f1']:.4f}      {m['auc']:.4f}      {m['ap']:.4f}")
        accs.append(m['accuracy'])
        prec.append(m['prec'])
        rec.append(m['rec'])
        f1.append(m['f1'])
        aucs.append(m['auc'])
        aps.append(m['ap'])
    
    print("-" * 85)
    print(f"Mean    {np.mean(accs):.4f}      {np.mean(prec):.4f}      {np.mean(rec):.4f}      {np.mean(f1):.4f}      {np.mean(aucs):.4f}      {np.mean(aps):.4f}")


if __name__ == "__main__":
    main()