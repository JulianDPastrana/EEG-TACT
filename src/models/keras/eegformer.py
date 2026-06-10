import os

import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from sklearn.metrics import balanced_accuracy_score
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import (
    Activation,
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    Dense,
    DepthwiseConv2D,
    Dropout,
    Input,
    Layer,
    MultiHeadAttention,
    Reshape,
    SeparableConv2D,
    SpatialDropout2D,
)
from tensorflow.keras.models import Model

try:
    import mne
except Exception:  # pragma: no cover - optional runtime dependency
    mne = None


class EEGDataset_ADHD_TF:
    """Subject-level dataset where each .mat file represents one subject."""

    def __init__(
        self,
        adhd_dir,
        control_dir,
        lowcut=0.5,
        highcut=60.0,
        notch=50.0,
        window=2.0,
        overlap=0.5,
        default_fs=128,
    ):
        self.samples = []
        self.lowcut = float(lowcut)
        self.highcut = float(highcut)
        self.notch = float(notch)
        self.window = float(window)
        self.overlap = float(overlap)
        self.default_fs = int(default_fs)

        self._process_folder(adhd_dir, label=1)
        self._process_folder(control_dir, label=0)

        if len(self.samples) == 0:
            raise ValueError("No subjects loaded. Check folders and file contents.")

    def _require_mne(self):
        if mne is None:
            raise ImportError(
                "mne is required to load .mat EEG files. Install it with `pip install mne`."
            )

    def _process_folder(self, folder, label):
        if not os.path.isdir(folder):
            raise ValueError(f"Directory does not exist: {folder}")

        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith(".mat"):
                continue
            mat_path = os.path.join(folder, fname)
            eeg = self._process_mat(mat_path)
            if eeg is not None:
                self.samples.append((fname, eeg.astype(np.float32), int(label)))

    def _process_mat(self, file_path):
        self._require_mne()

        mat = loadmat(file_path)
        key = os.path.splitext(os.path.basename(file_path))[0]
        if key not in mat:
            key = None
            for k, v in mat.items():
                if isinstance(v, np.ndarray) and v.ndim == 2 and v.size > 0:
                    key = k
                    break
            if key is None:
                return None

        data = np.asarray(mat[key], dtype=np.float64)

        fs = None
        for k in mat.keys():
            kl = k.lower()
            if ("fs" in kl) or ("freq" in kl) or ("sampling" in kl) or ("sfreq" in kl):
                try:
                    fs = int(np.squeeze(mat[k]))
                    break
                except Exception:
                    fs = None
        if fs is None or fs <= 0:
            fs = self.default_fs

        if data.ndim != 2:
            return None

        if data.shape[1] <= 256 and data.shape[0] > data.shape[1]:
            data = data.T

        n_ch, n_times = data.shape
        min_samples = int(np.ceil(self.window * fs))
        if n_times < min_samples:
            return None

        ch_names = [f"Ch{i + 1}" for i in range(n_ch)]
        info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=["eeg"] * n_ch)
        raw = mne.io.RawArray(data, info, verbose=False)

        raw.set_eeg_reference("average", verbose=False)
        raw.notch_filter(freqs=[self.notch], method="iir", verbose=False)
        raw.filter(self.lowcut, self.highcut, method="iir", verbose=False)

        step = self.window * (1.0 - self.overlap)
        if step <= 0:
            raise ValueError("overlap must be < 1.0 so that step > 0")

        epochs = mne.make_fixed_length_epochs(
            raw,
            duration=self.window,
            overlap=self.window - step,
            preload=True,
            verbose=False,
        )

        eeg_data = epochs.get_data()
        if eeg_data.size == 0:
            return None

        return eeg_data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def build_epoch_arrays(dataset_obj):
    x_list, y_list, group_list = [], [], []

    for i in range(len(dataset_obj)):
        name, eeg_epochs, label = dataset_obj[i]
        epoch_count = int(eeg_epochs.shape[0])
        x_list.append(eeg_epochs)
        y_list.append(np.full((epoch_count,), int(label), dtype=np.int32))
        group_list.append(np.full((epoch_count,), str(name), dtype=object))

    x = np.concatenate(x_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0).astype(np.int32)
    groups = np.concatenate(group_list, axis=0)
    return x, y, groups


def build_subject_table_from_epochs(groups_epoch, y_epoch):
    subject_names = np.unique(groups_epoch.astype(str))
    subject_labels = []

    for subject_name in subject_names:
        subject_y = np.unique(y_epoch[groups_epoch.astype(str) == subject_name])
        if len(subject_y) != 1:
            raise ValueError(f"Subject {subject_name} has multiple labels across epochs: {subject_y}")
        subject_labels.append(int(subject_y[0]))

    return subject_names, np.array(subject_labels, dtype=np.int32)


def epoch_indices_from_subjects(groups_epoch, subject_names_subset):
    subject_names_subset = np.array(subject_names_subset).astype(str)
    return np.where(np.isin(groups_epoch.astype(str), subject_names_subset))[0]


def make_ds_from_indices(X, y, groups, idxs, training, with_groups, batch_size, seed):
    x = X[idxs]
    yy = y[idxs]

    if with_groups:
        gg = groups[idxs].astype(str)
        ds = tf.data.Dataset.from_tensor_slices((x, yy, gg))
    else:
        ds = tf.data.Dataset.from_tensor_slices((x, yy))

    if training:
        ds = ds.shuffle(len(idxs), seed=seed, reshuffle_each_iteration=True)

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


class ValBalancedAccuracy(tf.keras.callbacks.Callback):
    def __init__(self, val_ds_xy):
        super().__init__()
        self.val_ds = val_ds_xy
        self.best = -np.inf
        self.last = None

    def on_epoch_end(self, epoch, logs=None):
        y_true, y_pred = [], []
        for xb, yb in self.val_ds:
            prob = self.model(xb, training=False).numpy().reshape(-1)
            pred = (prob >= 0.5).astype(int)
            y_true.extend(yb.numpy().tolist())
            y_pred.extend(pred.tolist())

        bacc = balanced_accuracy_score(y_true, y_pred)
        self.last = float(bacc)
        self.best = max(self.best, self.last)
        print(f" - val_balanced_accuracy: {self.last:.4f}", end="")


class RMSNorm(Layer):
    def __init__(self, d_model, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        self.eps = float(eps)

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale",
            shape=(self.d_model,),
            initializer="ones",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        rms = tf.sqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.scale

    def get_config(self):
        return {**super().get_config(), "d_model": self.d_model, "eps": self.eps}


class TransformerEncoderLayer(Layer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout = float(dropout)

        if self.d_model % self.nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead}).")

        key_dim = self.d_model // self.nhead
        self.self_attn = MultiHeadAttention(
            num_heads=self.nhead,
            key_dim=key_dim,
            output_shape=self.d_model,
            name="self_attn",
        )
        self.ffn = tf.keras.Sequential(
            [
                Dense(self.dim_feedforward, activation="gelu", name="ffn_0"),
                Dropout(self.dropout, name="ffn_2"),
                Dense(self.d_model, name="ffn_3"),
            ],
            name="ffn",
        )
        self.norm1 = RMSNorm(self.d_model, name="norm1")
        self.norm2 = RMSNorm(self.d_model, name="norm2")
        self.dropout1 = Dropout(self.dropout, name="dropout1")
        self.dropout2 = Dropout(self.dropout, name="dropout2")

    def call(self, x, training=None):
        h = self.norm1(x)
        x = x + self.dropout1(self.self_attn(h, h, h, training=training), training=training)
        h = self.norm2(x)
        x = x + self.dropout2(self.ffn(h, training=training), training=training)
        return x

    def get_config(self):
        return {
            **super().get_config(),
            "d_model": self.d_model,
            "nhead": self.nhead,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
        }


class TransformerEncoder(Layer):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward=2048, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = int(num_layers)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout = float(dropout)

        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}.")

        self.layers_ = [
            TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                name=f"layers_{i}",
            )
            for i in range(self.num_layers)
        ]

    def call(self, x, training=None):
        for layer in self.layers_:
            x = layer(x, training=training)
        return x

    def get_config(self):
        return {
            **super().get_config(),
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
        }


def build_model(
    n_channels=19,
    n_samples=256,
    F1=8,
    D=3,
    F2=16,
    kern_length=64,
    pool1=4,
    pool2=8,
    eeg_activation="elu",
    d_model=None,
    nhead=2,
    dim_feedforward=128,
    num_layers=2,
    do_rate_transf=0.3,
    do_rate_eeg=0.1,
    do_rate_cls=0.3,
):
    reg = tf.keras.regularizers.l2(1e-4)

    inputs = Input(shape=(n_channels, n_samples), name="input")
    x = Reshape((n_channels, n_samples, 1), name="expand_dims")(inputs)

    x = Conv2D(
        F1,
        (1, kern_length),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name="eeg_temporal",
    )(x)
    x = BatchNormalization(name="bn_t")(x)

    x = DepthwiseConv2D(
        (n_channels, 1),
        depth_multiplier=D,
        padding="valid",
        use_bias=False,
        depthwise_initializer="he_normal",
        depthwise_constraint=max_norm(1.0),
        name="eeg_depthwise",
    )(x)
    x = BatchNormalization(name="bn_dw")(x)
    x = Activation(eeg_activation, name="act_dw")(x)
    x = AveragePooling2D((1, pool1), name="pool_dw")(x)
    x = SpatialDropout2D(do_rate_eeg, name="drop_dw")(x)

    x = SeparableConv2D(
        F2,
        (1, 16),
        padding="same",
        use_bias=False,
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
        name="eeg_separable",
    )(x)
    x = BatchNormalization(name="bn_sep")(x)
    x = Activation(eeg_activation, name="act_sep")(x)
    x = AveragePooling2D((1, pool2), name="pool_sep")(x)
    x = SpatialDropout2D(do_rate_eeg, name="drop_sep")(x)

    x = Reshape((-1, F2), name="to_tokens")(x)

    if d_model is not None and int(d_model) != int(F2):
        x = Dense(
            int(d_model),
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=reg,
            name="proj_dmodel",
        )(x)
        d_used = int(d_model)
    else:
        d_used = int(F2)

    x = TransformerEncoder(
        num_layers=num_layers,
        d_model=d_used,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=do_rate_transf,
        name="encoder",
    )(x)

    x = AttentionPooling(d_used, name="attn_pool")(x)
    x = Dropout(do_rate_cls, name="final_dropout")(x)

    outputs = Dense(
        1,
        activation="sigmoid",
        kernel_constraint=max_norm(0.25),
        name="classifier",
    )(x)

    return Model(inputs, outputs, name="EEGNet_Transformer_SoftmaxFixed")


class AttentionPooling(Layer):
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        self.attn = Dense(1)

    def call(self, x, training=None):
        scores = self.attn(x)
        attn_weights = tf.nn.softmax(scores, axis=1)
        return tf.reduce_sum(x * attn_weights, axis=1)

    def get_config(self):
        return {**super().get_config(), "d_model": self.d_model}


__all__ = [
    "AttentionPooling",
    "EEGDataset_ADHD_TF",
    "RMSNorm",
    "TransformerEncoder",
    "TransformerEncoderLayer",
    "ValBalancedAccuracy",
    "build_epoch_arrays",
    "build_model",
    "build_subject_table_from_epochs",
    "epoch_indices_from_subjects",
    "make_ds_from_indices",
]


if __name__ == "__main__":
    model = build_model()
    print(model)
    model.summary()