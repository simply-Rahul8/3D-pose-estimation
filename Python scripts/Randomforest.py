import os, json
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor,
    VotingClassifier, VotingRegressor
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error, confusion_matrix

def load_sample(sample_id, data_path):
    kp = json.load(open(os.path.join(data_path, f"{sample_id}_result_time_series_3d.json")))['all_pose_3d']
    km = json.load(open(os.path.join(data_path, f"{sample_id}_result_kinematics.json")))['frames']
    A = np.array([f['keypoints'] for f in kp]).reshape(len(kp), -1)
    B = np.array([
        np.concatenate([
            np.array(fr['joint_velocities']).flatten(),
            np.array(fr['joint_accelerations']).flatten(),
            np.array(list(fr['joint_angles'].values())).flatten()
        ])
        for fr in km
    ])
    L = min(len(A), len(B))
    return np.concatenate([A[:L], B[:L]], axis=1)

def make_windows(X, y_class, y_level, ids, window=30, step=15):
    Xw, yc, yr, iw = [], [], [], []
    for x, c, r, sid in zip(X, y_class, y_level, ids):
        for start in range(0, x.shape[0] - window + 1, step):
            w = x[start:start+window]
            stats = [w.mean(0), w.std(0), w.min(0), w.max(0)]
            Xw.append(np.hstack(stats))
            yc.append(c)
            yr.append(r)
            iw.append(sid)
    return np.vstack(Xw), np.array(yc), np.array(yr), np.array(iw)

def main():
    # — 1) sample IDs, classes & true fatigue levels —
    sample_ids = [f"sample{i}" for i in range(1,13)]
    labels = {}
    # first 6 fatigued level=5, last 6 normal level=0
    for i,s in enumerate(sample_ids):
        labels[s] = {
            'class': 1 if i<6 else 0,
            'level': 5.0 if i<6 else 0.0
        }

    data_path = "C:/Users/MYSEL/Desktop/mmpose-updated/mmpose/vis_results/dataset_3d/data"

    # — 2) load full sequences —
    X_full = [load_sample(s, data_path) for s in sample_ids]
    y_class = np.array([labels[s]['class'] for s in sample_ids])
    y_level = np.array([labels[s]['level'] for s in sample_ids])

    # — 3) make overlapping windows —
    Xw, ywc, ywr, iw = make_windows(X_full, y_class, y_level, sample_ids,
                                    window=30, step=15)
    print("Window dataset:", Xw.shape)

    # — 4) normalize window features —
    scaler = StandardScaler().fit(Xw)
    Xw = scaler.transform(Xw)

    # — 5) prepare ensembles —
    clf = VotingClassifier([
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=5, random_state=0)),
        ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=0))
    ], voting='soft', weights=[2,1])
    reg = VotingRegressor([
        ('rf', RandomForestRegressor(n_estimators=200, max_depth=5, random_state=0)),
        ('gb', GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=0))
    ])

    # — 6) stratified 4-fold on windows, aggregate to samples —
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
    win_pc, win_tc, win_pr, win_tr = [], [], [], []
    samp_pc, samp_tc, samp_pr, samp_tr, samp_ids = [], [], [], [], []

    for fold, (tr, te) in enumerate(skf.split(Xw, ywc), 1):
        # train on windows
        clf.fit(Xw[tr], ywc[tr])
        reg.fit(Xw[tr], ywr[tr])

        # window-level predict
        pc = clf.predict(Xw[te])
        pr = reg.predict(Xw[te])
        win_pc.extend(pc); win_tc.extend(ywc[te])
        win_pr.extend(pr); win_tr.extend(ywr[te])

        # sample-level aggregate
        test_ids = iw[te]
        for sid in np.unique(test_ids):
            mask = (test_ids == sid)
            # classification by majority vote
            pred_c = Counter(pc[mask]).most_common(1)[0][0]
            # regression by average
            pred_r = pr[mask].mean()
            samp_ids.append(sid)
            samp_tc.append(labels[sid]['class'])
            samp_pc.append(pred_c)
            samp_tr.append(labels[sid]['level'])
            samp_pr.append(pred_r)

        print(f" Fold {fold}/4 done")

    # — 7) report metrics —
    print(f"\nWindow-level —  Acc: {accuracy_score(win_tc,win_pc)*100:.2f}%, "
          f"MAE: {mean_absolute_error(win_tr,win_pr):.2f}")
    # window-level confusion matrix
    cm_win = confusion_matrix(win_tc, win_pc)
    print("Window-level Confusion Matrix:")
    print(cm_win)

    print(f"\nSample-level —  Acc: {accuracy_score(samp_tc,samp_pc)*100:.2f}%, "
          f"MAE: {mean_absolute_error(samp_tr,samp_pr):.2f}")
    # sample-level confusion matrix
    cm_samp = confusion_matrix(samp_tc, samp_pc)
    print("Sample-level Confusion Matrix:")
    print(cm_samp)

    # — 8) save sample-level results —
    df = pd.DataFrame({
        'Sample': samp_ids,
        'TrueClass': samp_tc, 'PredClass': samp_pc,
        'TrueLevel': samp_tr, 'PredLevel': samp_pr
    })
    df.to_csv("rf_gb_ensemble_with_level.csv", index=False)
    print("Saved rf_gb_ensemble_with_level.csv")

if __name__ == "__main__":
    main()
