"""Rotation-noise experiment on real data (in-sample setting, no train/test split).

Fit a clean MD-LP on all the data, then evaluate it on the same data after adding
noise, which isolates the noise effect.
Sites: VGG_L3, VGG_L9, RES_L3, RES_L8; theta in 5-degree steps; c in {4,8,12,16}.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .cfg import EXP3, SITES, features_path
from ..common.mdlp import compute_mdlp, save_json
from ..common.metrics import linear_cka, svcca_score

SITES_V = ["VGG_L3", "VGG_L9", "RES_L3", "RES_L8"]
CS = [4.0, 8.0, 12.0, 16.0]
THETAS = list(range(0, 91, 5))
SEED = 20261108
N_CKA = 250
N_TSNE = 120
PCA_DIM = 256


def _tsne(P):
    kw = dict(n_components=2, perplexity=30.0, init="pca", random_state=0)
    try:
        return TSNE(max_iter=500, **kw).fit_transform(P)
    except TypeError:
        return TSNE(n_iter=500, **kw).fit_transform(P)


def _overlap(a, b, labels):
    dpair = np.linalg.norm(a - b, axis=1)
    dmat = np.linalg.norm(a[:, None, :] - a[None, :, :], axis=2)
    same = labels[:, None] == labels[None, :]
    dmat = np.where(same, np.inf, dmat)
    return float(np.mean(dpair < dmat.min(axis=1)))


def _acc_bin(X, y, w, b):
    return float(np.mean(((X @ w + b) >= 0).astype(np.int32) == y))


def main():
    out = {"cs": CS, "thetas": THETAS, "split": "none", "sites": {}}
    for site in SITES_V:
        arr = np.load(features_path(SITES[site][0]))
        X = np.asarray(arr[f"train_{site}"], dtype=np.float64)
        y = arr["y_train"].astype(np.int32)
        yb = (y == 0).astype(np.int32)
        mu, sd = X.mean(0), X.std(0) + 1e-9
        X = (X - mu) / sd
        d = X.shape[1]

        delta = X[yb == 1].mean(0) - X[yb == 0].mean(0)
        e1 = delta / np.linalg.norm(delta)
        mu4 = {c: X[y == c].mean(0) for c in range(4)}
        Xc = X - np.stack([mu4[v] for v in y])
        rng0 = np.random.RandomState(SEED + 9)
        v = rng0.randn(d)
        v /= np.linalg.norm(v)
        for _ in range(25):
            z = Xc.T @ (Xc @ v) / len(Xc)
            v = z / (np.linalg.norm(z) + 1e-30)
        e2 = v - (v @ e1) * e1
        e2 /= np.linalg.norm(e2)
        sig1 = float(np.std(Xc @ e1))
        sig2 = float(np.std(Xc @ e2))

        probe = compute_mdlp(X, yb, mode="exact", positive=1)
        clean_acc = _acc_bin(X, yb, probe["w"], probe["b"])
        rng = np.random.RandomState(SEED)
        sub = rng.choice(len(X), size=N_CKA, replace=False)
        tsub = sub[:N_TSNE]
        pca = PCA(n_components=min(PCA_DIM, min(X.shape) - 1),
                  svd_solver="randomized", random_state=0).fit(X[::4])
        V = pca.components_.T
        A_proj = X[sub] @ V
        e1p, e2p = e1 @ V, e2 @ V
        print(f"[{site}] d={d} sig1={sig1:.2f} sig2={sig2:.2f} "
              f"clean(in-sample)={clean_acc:.4f}", flush=True)

        res = {}
        for c in CS:
            res[f"{c:g}"] = {}
            for th in THETAS:
                t = np.deg2rad(th)
                u = np.cos(t) * e1 + np.sin(t) * e2
                up = np.cos(t) * e1p + np.sin(t) * e2p
                amp = c * sig1
                xi = rng.randn(len(X))
                Xp = X + amp * xi[:, None] * u
                probe_acc = _acc_bin(Xp, yb, probe["w"], probe["b"])
                B = A_proj + amp * xi[sub][:, None] * up
                cka = linear_cka(A_proj, B)
                svcca = svcca_score(A_proj[:256], B[:256])
                emb = _tsne(np.vstack([A_proj[:N_TSNE], B[:N_TSNE]]))
                tsne = _overlap(emb[:N_TSNE], emb[N_TSNE:], yb[tsub])
                res[f"{c:g}"][f"{th}"] = dict(probe=probe_acc, cka=cka,
                                              svcca=svcca, tsne=tsne)
            print(f"    {site} c={c:g} done", flush=True)
        out["sites"][site] = dict(d=int(d), sig1=sig1, sig2=sig2,
                                  clean=clean_acc, res=res)
        save_json(EXP3 / "real_rot_insample.json", out)

    print("[done]")


if __name__ == "__main__":
    main()
