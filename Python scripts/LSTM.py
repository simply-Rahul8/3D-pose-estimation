import os
# silence TF C++ and Python warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import json, numpy as np, pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D,
    Bidirectional, LSTM, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import BinaryFocalCrossentropy
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, mean_absolute_error,
    confusion_matrix, classification_report
)
from sklearn.utils.class_weight import compute_class_weight
from collections import defaultdict, Counter

def load_sample(sample_id, data_path):
    kp = json.load(open(os.path.join(data_path, f"{sample_id}_result_time_series_3d.json")))['all_pose_3d']
    km = json.load(open(os.path.join(data_path, f"{sample_id}_result_kinematics.json")))['frames']
    A = np.array([f['keypoints'] for f in kp]).reshape(len(kp), -1)
    B = np.array([np.concatenate([
            np.array(fr['joint_velocities']).flatten(),
            np.array(fr['joint_accelerations']).flatten(),
            np.array(list(fr['joint_angles'].values())).flatten()
        ]) for fr in km])
    L = min(len(A), len(B))
    return np.concatenate([A[:L], B[:L]], axis=1)

def make_windows(X, y_class, y_reg, ids, win=20, step=5):
    Xw, yc, yr, iw = [], [], [], []
    for x, c, r, sid in zip(X, y_class, y_reg, ids):
        for start in range(0, x.shape[0] - win + 1, step):
            Xw.append(x[start:start+win])
            yc.append(c); yr.append(r); iw.append(sid)
    return np.array(Xw), np.array(yc), np.array(yr), np.array(iw)

def build_model(T, F):
    inp = Input((T, F))
    x = Conv1D(64, 3, activation='relu', padding='same')(inp)
    x = MaxPooling1D(2)(x)
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2)(x)
    x = Bidirectional(LSTM(64))(x)
    x = BatchNormalization()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    out_c = Dense(1, activation='sigmoid', name='class')(x)
    out_r = Dense(1, activation='linear',  name='reg')(x)
    m = Model(inp, [out_c, out_r])
    m.compile(
      'adam',
      loss={
        'class': BinaryFocalCrossentropy(gamma=2.0),
        'reg':   'mse'
      },
      metrics={
        'class': ['accuracy',
                  tf.keras.metrics.Recall(name='recall'),
                  tf.keras.metrics.Precision(name='precision')],
        'reg':   ['mae']
      },
      weighted_metrics={'class': [], 'reg': []}
    )
    return m

def main():
    sample_ids = [f"sample{i}" for i in range(1,13)]
    labels_class = {s:(1 if i<6 else 0) for i,s in enumerate(sample_ids)}
    labels_reg   = {s:(5.0 if i<6 else 0.0) for i,s in enumerate(sample_ids)}
    data_path = "C:/Users/MYSEL/Desktop/mmpose-updated/mmpose/vis_results/dataset_3d/data"

    X_full = [load_sample(s, data_path) for s in sample_ids]
    y_class_full = np.array([labels_class[s] for s in sample_ids])
    y_reg_full   = np.array([labels_reg[s]   for s in sample_ids])

    Xw, ywc, ywr, iw = make_windows(X_full, y_class_full, y_reg_full, sample_ids, win=20, step=5)
    print("Window dataset:", Xw.shape)

    ns, T, F = Xw.shape
    Xw_flat = Xw.reshape(-1, F)
    scaler = StandardScaler().fit(Xw_flat)
    Xw = scaler.transform(Xw_flat).reshape(ns, T, F)

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
    win_probs, win_trues, win_ids = [], [], []
    win_pr, win_reg_trues = [], []

    for fold, (tr, te) in enumerate(skf.split(Xw, ywc),1):
        print(f" Fold {fold}/4", end='', flush=True)
        cw = compute_class_weight('balanced', classes=[0,1], y=ywc[tr])
        sample_weight = np.array([cw[val] for val in ywc[tr]])
        model = build_model(T, F)
        es = EarlyStopping('val_loss', patience=5, restore_best_weights=True)
        rp = ReduceLROnPlateau('val_loss', factor=0.5, patience=3)
        model.fit(
          Xw[tr],
          {'class': ywc[tr], 'reg': ywr[tr]},
          sample_weight={'class': sample_weight, 'reg': np.ones_like(sample_weight)},
          validation_split=0.2, epochs=50, batch_size=8,
          callbacks=[es,rp], verbose=0
        )
        pc, pr = model.predict(Xw[te])
        probs = pc.flatten()
        win_probs.extend(probs)
        win_pr.extend(pr.flatten())
        win_trues.extend(ywc[te])
        win_reg_trues.extend(ywr[te])
        win_ids.extend(iw[te])
        print(" done")

    grouped = defaultdict(list)
    for pid, tc, p in zip(win_ids, win_trues, win_probs):
        grouped[pid].append((tc,p))
    best_acc, best_thr = 0, 0.5
    for thr in np.linspace(0.2,0.8,31):
        y_t, y_p = [], []
        for sid, lst in grouped.items():
            y_t.append(lst[0][0])
            y_p.append(1 if np.mean([p for _,p in lst])>thr else 0)
        acc = accuracy_score(y_t, y_p)
        if acc>best_acc:
            best_acc, best_thr = acc, thr

    print(f"\nBest threshold = {best_thr:.2f} → Sample Acc: {best_acc*100:.2f}%")

    y_t, y_p = [], []
    for sid in sample_ids:
        lst = grouped[sid]
        y_t.append(lst[0][0])
        y_p.append(1 if np.mean([p for _,p in lst])>best_thr else 0)
    cm = confusion_matrix(y_t, y_p)
    cr = classification_report(y_t, y_p, target_names=['Normal','Fatigue'])
    print("\nConfusion matrix at best threshold:")
    print(cm)
    print("\nClassification report:")
    print(cr)

    y_rp, y_rt = [], []
    for sid in sample_ids:
        preds = [pr for pid,pr in zip(win_ids, win_pr) if pid==sid]
        y_rp.append(np.mean(preds)); y_rt.append(labels_reg[sid])
    print(f"\nSample-level Regression MAE: {mean_absolute_error(y_rt,y_rp):.2f}")

if __name__=="__main__":
    main()
