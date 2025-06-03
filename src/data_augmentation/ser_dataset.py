import os
import random
from typing import Optional, Tuple, List

import librosa
import nltk
import numpy as np
import torch
from librosa.effects import pitch_shift, time_stretch
from nltk.corpus import wordnet
from torch.utils.data import Dataset

# 仅首次会下载一次 WordNet：同义词替换用
nltk.download("wordnet", quiet=True)


def synonym_replace(sentence: str, n_swaps: int = 1) -> str:
    """对句子做 *n_swaps* 次同义词替换（仅针对原始字符串）。"""
    words = sentence.split()
    if not words:
        return sentence
    for _ in range(n_swaps):
        idx = random.randrange(len(words))
        syns = wordnet.synsets(words[idx])
        if syns:
            lemma = syns[0].lemmas()[0].name().replace("_", " ")
            words[idx] = lemma
    return " ".join(words)


def time_mask(mel: np.ndarray, T_mask: int = 30) -> np.ndarray:
    """对 Mel-谱做随机时域掩蔽。"""
    n_mels, T = mel.shape
    t = random.randint(0, max(0, T - T_mask))
    mel[:, t : t + T_mask] = 0.0
    return mel


def freq_mask(mel: np.ndarray, F_mask: int = 10) -> np.ndarray:
    """对 Mel-谱做随机频域掩蔽。"""
    n_mels, T = mel.shape
    f = random.randint(0, max(0, n_mels - F_mask))
    mel[f : f + F_mask, :] = 0.0
    return mel


class SERAugmentedDataset(Dataset):
    """Multimodal SER 数据集。

    支持：
    * 预存 Mel 谱 (``spectro_data``) 或从 WAV 现场计算 (``wav_list``)
    * 可选音频增强（时域拉伸、变调、加噪）
    * 对四类 (ang, hap, sad, neu) 做差异化增强
    * 可选文本增强（同义词替换）
    """

    def __init__(
        self,
        wav_list: Optional[str] = None,
        spectro_data: Optional[str] = None,
        labels: Optional[str] = None,
        text_data: Optional[str] = None,
        augment: bool = False,
        sr: int = 16_000,
        pitch_range: float = 2.0,
        stretch_range: float = 0.1,
        noise_std: float = 0.005,
        extra_noises: Optional[List[np.ndarray]] = None,
    ) -> None:
        super().__init__()

        # ────────────────────────────────────────────────────────────────
        # 1. 载入 labels
        # ----------------------------------------------------------------
        if labels is None or not os.path.exists(labels):
            raise FileNotFoundError(f"[Dataset] 找不到 labels 文件：{labels}")

        self.y = self._load_np_generic(labels, prefer_keys=("y",)).astype(np.int64)
        self.N = len(self.y)

        # ────────────────────────────────────────────────────────────────
        # 2. 载入文本（可选）
        # ----------------------------------------------------------------
        self.text: Optional[np.ndarray] = None
        self.raw_text: Optional[List[str]] = None

        if text_data is not None:
            if not os.path.exists(text_data):
                raise FileNotFoundError(f"[Dataset] 找不到 text_data：{text_data}")
            arr = self._load_np_generic(text_data, prefer_keys=("X", "x"))

            if arr.dtype.kind in ("U", "S") or arr.dtype == object:
                self.raw_text = arr.tolist()
            else:
                self.text = arr.astype(np.float32)

            if len(arr) != self.N:
                raise ValueError(
                    f"[Dataset] text_data 大小 {len(arr)} 与 labels 大小 {self.N} 不匹配"
                )

        # ────────────────────────────────────────────────────────────────
        # 3. 载入或检查 Mel 谱
        # ----------------------------------------------------------------
        self.precomputed_spectro: Optional[np.ndarray] = None
        self.n_mels: Optional[int] = None
        self.T_max: Optional[int] = None

        if spectro_data is not None:
            if not os.path.exists(spectro_data):
                raise FileNotFoundError(f"[Dataset] 找不到 spectro_data 文件：{spectro_data}")
            arr_sp = self._load_np_generic(spectro_data, prefer_keys=("X", "x", "y"))

            if arr_sp.ndim != 3:
                raise ValueError("[Dataset] spectro_data 必须是三维 (N, n_mels, T_max)")
            if arr_sp.shape[0] != self.N:
                raise ValueError(
                    f"[Dataset] spectro_data 的 N={arr_sp.shape[0]} 与 labels 的 N={self.N} 不匹配"
                )

            self.precomputed_spectro = arr_sp.astype(np.float32)
            self.n_mels, self.T_max = self.precomputed_spectro.shape[1:]

        # ────────────────────────────────────────────────────────────────
        # 4. 记录 wav_list（如果需要从 WAV 现场计算 Mel）
        # ----------------------------------------------------------------
        self.wav_list: Optional[List[str]] = None
        if wav_list is not None:
            if not os.path.exists(wav_list):
                raise FileNotFoundError(f"[Dataset] 找不到 wav_list 文件：{wav_list}")
            tmp = np.load(wav_list, allow_pickle=True)
            if tmp.dtype.kind not in ("U", "S") and tmp.dtype != object:
                raise ValueError("[Dataset] wav_list 应为字符串数组")
            self.wav_list = tmp.tolist()
            if len(self.wav_list) != self.N:
                raise ValueError(
                    f"[Dataset] wav_list 长度 {len(self.wav_list)} 与 labels 长度 {self.N} 不匹配"
                )

        # 至少要有一个 Mel 来源
        if self.precomputed_spectro is None and self.wav_list is None:
            raise ValueError("[Dataset] 必须提供 wav_list 或 spectro_data（二者其一）")

        # 如果只提供 wav_list，先占位 n_mels；T_max 训练时动态更新
        if self.precomputed_spectro is None:
            self.n_mels, self.T_max = 64, None

        # ────────────────────────────────────────────────────────────────
        # 5. 其余设置
        # ----------------------------------------------------------------
        self.augment = augment
        self.sr = sr
        self.pitch_range = pitch_range
        self.stretch_range = stretch_range
        self.noise_std = noise_std

        # —— 新增：用于叠加真实环境噪声的列表（可选）
        self.extra_noises = extra_noises or []

    # ------------------------------------------------------------------
    # 核心辅助函数
    # ------------------------------------------------------------------
    @staticmethod
    def _load_np_generic(path: str, prefer_keys: Tuple[str, ...] = ("X",)) -> np.ndarray:
        """通用加载器：既支持 .npy 也支持 .npz，并按优先 key 取数组。"""
        obj = np.load(path, allow_pickle=True)
        if isinstance(obj, np.lib.npyio.NpzFile):
            for k in (*prefer_keys, *obj.keys()):
                if k in obj:
                    return obj[k]
            raise ValueError(f"[Dataset] 未找到期望键 {prefer_keys} 于 {path}")
        return obj

    # ------------------------------------------------------------------
    # Dataset 接口实现
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.N

    def _pad_or_truncate(self, mel: np.ndarray) -> np.ndarray:
        """把 (n_mels, T_var) 的 Mel pad / truncate 到统一 T_max。"""
        n_mels, T_var = mel.shape
        if self.T_max is None:
            self.T_max = T_var
        if T_var < self.T_max:
            pad_width = self.T_max - T_var
            return np.pad(mel, ((0, 0), (0, pad_width)), mode="constant")
        return mel[:, : self.T_max]

    def __getitem__(self, idx: int):
        # 1) Mel 部分
        if self.precomputed_spectro is not None and not self.augment:
            mel = self.precomputed_spectro[idx]
        else:
            if self.wav_list is None:
                raise RuntimeError("[Dataset] 缺少 wav_list，无法现场计算 Mel")
            wav_path = self.wav_list[idx]
            if not os.path.exists(wav_path):
                raise FileNotFoundError(f"[Dataset] WAV 不存在：{wav_path}")

            y, sr = librosa.load(wav_path, sr=self.sr)

            if self.augment:
                # ──────────────────────────────────────────────────────
                # 5.1) 基础三步增强（对所有类别都做）
                # 时域拉伸 (± stretch_range)
                y = time_stretch(
                    y, rate=random.uniform(1 - self.stretch_range, 1 + self.stretch_range)
                )
                # 变调 (± pitch_range)
                y = pitch_shift(
                    y, sr=sr, n_steps=random.uniform(-self.pitch_range, self.pitch_range)
                )
                # 加噪
                y = y + np.random.randn(len(y)) * self.noise_std

                # ──────────────────────────────────────────────────────
                # 5.2) 类别定向增强
                label = int(self.y[idx])

                if label == 0:  # ang: 做低频增益 + 高频掩蔽
                    # a) 低频增益：对 STFT 的低于 300Hz 部分乘以 gain_factor
                    D = librosa.stft(y)
                    freqs = librosa.fft_frequencies(sr=sr, n_fft=(D.shape[0] - 1) * 2)
                    gain_factor = 1.5  # 低频放大倍数
                    low_freq_mask = freqs < 300
                    D[low_freq_mask, :] *= gain_factor
                    y = librosa.istft(D, length=len(y))

                elif label == 2:  # sad: 更强的时域拉伸 (rate < 0.8) + 向下 pitch_shift
                    # a) 明显减速 (0.6 ~ 0.8)，表现悲伤拖沓
                    y = time_stretch(y, rate=random.uniform(0.6, 0.8))
                    # b) 向下变调 (-4 ~ -2)
                    y = pitch_shift(y, sr=sr, n_steps=random.uniform(-4.0, -2.0))

                elif label == 3:  # neu: 加入轻混响或轻噪，让 neutral 更“干净”
                    # a) 叠加少量白噪声，模拟轻室内环境
                    y = y + np.random.randn(len(y)) * (self.noise_std * 0.5)
                    # b) 轻混响：单 tap 混响
                    reverb_delay = int(0.02 * sr)  # 20ms 延迟
                    decay = 0.3                    # 衰减系数
                    if len(y) > reverb_delay:
                        y_reverb = np.zeros_like(y)
                        y_reverb[reverb_delay:] = y[:-reverb_delay] * decay
                        y = y + y_reverb

                elif label == 1:  # hap: 更极端的 pitch_shift/time_stretch + 环境噪声
                    # a) 更大范围的 pitch shift (±4)
                    y = pitch_shift(y, sr=sr, n_steps=random.uniform(-4.0, 4.0))
                    # b) 更明显的 time stretch (1.1 ~ 1.3)，模拟兴奋
                    y = time_stretch(y, rate=random.uniform(1.1, 1.3))
                    # c) 叠加真实环境噪声（如果提供了）
                    if len(self.extra_noises) > 0:
                        env = random.choice(self.extra_noises)
                        if len(env) >= len(y):
                            env_cut = env[: len(y)]
                        else:
                            env_cut = np.tile(env, int(np.ceil(len(y) / len(env))))[: len(y)]
                        y = 0.8 * y + 0.2 * env_cut

            # ──────────────────────────────────────────────────────
            # 计算 Mel-谱 & pad/truncate
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
            mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
            mel = self._pad_or_truncate(mel)

            # ──────────────────────────────────────────────────────
            # 5.3) 动态 SpecAugment：根据不同类别掩蔽不同频段/时段
            if self.augment:
                # 对所有样本做一次中等强度的时/频掩蔽
                if random.random() < 0.5:
                    mel = time_mask(mel, T_mask=random.randint(10, 30))
                if random.random() < 0.5:
                    mel = freq_mask(mel, F_mask=random.randint(5, 15))

                # 再根据类别做定向掩蔽
                if label == 0:  # ang: 掩蔽更多高频
                    mel = freq_mask(mel, F_mask=random.randint(10, 20))
                elif label == 2:  # sad: 掩蔽更多低频
                    n_mels, _ = mel.shape
                    F_mask = random.randint(10, 20)
                    mel[0 : F_mask, :] = 0.0
                elif label == 3:  # neu: 轻度掩蔽
                    if random.random() < 0.5:
                        mel = time_mask(mel, T_mask=random.randint(5, 15))
                    if random.random() < 0.5:
                        mel = freq_mask(mel, F_mask=random.randint(5, 10))
                # hap 已在前面 waveform 层面做了极端增强，mel 层可视需要继续加强

        mel_tensor = torch.from_numpy(mel)

        # ────────────────────────────────────────────────────────────────
        # 2) 文本部分
        if self.text is not None:
            txt_tensor = torch.from_numpy(self.text[idx]).float()
        elif self.raw_text is not None:
            sent = self.raw_text[idx]
            if self.augment:
                # 基础同义词替换
                sent = synonym_replace(sent, n_swaps=1)
                # 再根据类别做定向文本增强
                label = int(self.y[idx])
                if label == 0:  # ang: 多做一次同义词替换
                    sent = synonym_replace(sent, n_swaps=1)
                elif label == 2:  # sad: 可做回译以增强悲伤语气 (需自行实现 back_translate)
                    # sent = back_translate(sent, src_lang="en", mid_lang="de")
                    pass
                elif label == 3:  # neu: 保持原文或只做一次替换
                    pass
                elif label == 1:  # hap: 再做一次同义词替换
                    sent = synonym_replace(sent, n_swaps=1)
            txt_tensor = sent
        else:
            txt_tensor = torch.zeros(1, dtype=torch.float32)

        # ────────────────────────────────────────────────────────────────
        # 3) 标签
        label = int(self.y[idx])
        return mel_tensor, txt_tensor, label
