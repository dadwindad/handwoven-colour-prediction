"""Colour prediction models for a two-colour plain weave.

Every model follows the same interface so the experiment can treat them alike:
    fit(ds, idx) -> self          ds: data.Dataset, idx: training row indices
    predict(ds, idx) -> (n, 3)    predicted fabric L*a*b*

Families
  * baselines           mean of the two yarn colours
  * physics             colour-mixing laws from the literature (Chae et al., 2014 and refs)
  * ML                  learners on the six yarn L*a*b* values, tuned by inner CV
  * hybrid              physics model + ML learner trained on its residuals
The CNN models that use photos live in cnn.py.

Approximation: the published mixing laws act on spectral reflectance R(λ).
Table 4.7 has only L*a*b*, so the laws are applied to the tristimulus
channels X/Xn, Y/Yn, Z/Zn as pseudo-reflectances. This is stated in the results.
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor

from .colorlib import DEFAULT_WHITE, WHITE_POINTS, delta_e_2000, lab_to_xyz, xyz_to_lab

R_MIN = 1e-3


# ---------------------------------------------------------------- helpers
def features(ds, idx):
    """Six inputs: warp L*a*b* and weft L*a*b*, scaled to roughly unit range."""
    return np.hstack([ds.warp_lab[idx], ds.weft_lab[idx]]) / 50.0


def _fit_params(loss, theta0):
    res = minimize(loss, np.asarray(theta0, float), method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-5, "maxiter": 4000})
    return res.x


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ---------------------------------------------------------------- baselines
class MeanLab:
    name = "เฉลี่ย L*a*b*"
    family = "baseline"

    def fit(self, ds, idx, cv=None):
        return self

    def predict(self, ds, idx):
        return (ds.warp_lab[idx] + ds.weft_lab[idx]) / 2


class DimitrovskiGabrijelcic:
    """D-G: weighted average in L*a*b* (Eq. 10-12 of Chae 2014), warp share w fitted.

    The background share is taken as zero because the open area is unknown;
    w absorbs it."""

    name = "D-G (L*a*b* ถ่วงน้ำหนัก)"
    family = "physics"

    def __init__(self, white=DEFAULT_WHITE):
        self.white = white  # unused (works in L*a*b*), kept so every physics model shares one signature

    def fit(self, ds, idx, cv=None):
        wl, fl, y = ds.warp_lab[idx], ds.weft_lab[idx], ds.lab[idx]
        loss = lambda t: delta_e_2000(_sigmoid(t[0]) * wl + (1 - _sigmoid(t[0])) * fl, y).mean()
        self.w = float(_sigmoid(_fit_params(loss, [0.0])[0]))
        return self

    def predict(self, ds, idx):
        return self.w * ds.warp_lab[idx] + (1 - self.w) * ds.weft_lab[idx]


# ---------------------------------------------------------------- physics on pseudo-reflectance
class ReflectanceMixModel:
    """Generic additive mixing law: f(R_mix) = w f(R_warp) + (1 - w) f(R_weft), per channel.

    Subclasses define f / f_inv and optional extra parameters. w (warp share)
    is always fitted, plus any extra parameters, by minimising mean CIEDE2000.
    """

    family = "physics"
    extra0 = ()           # initial values of extra (unconstrained) parameters

    def __init__(self, white=DEFAULT_WHITE):
        self.white = white

    def _R(self, lab):
        return np.clip(lab_to_xyz(lab, self.white) / WHITE_POINTS[self.white], R_MIN, 1.0)

    def _lab(self, R):
        return xyz_to_lab(np.clip(R, R_MIN, 1.5) * WHITE_POINTS[self.white], self.white)

    def mix(self, Rw, Rf, w, extra):
        return self.f_inv(w * self.f(Rw, extra) + (1 - w) * self.f(Rf, extra), extra)

    def fit(self, ds, idx, cv=None):
        Rw, Rf, y = self._R(ds.warp_lab[idx]), self._R(ds.weft_lab[idx]), ds.lab[idx]

        def loss(t):
            pred = self._lab(self.mix(Rw, Rf, _sigmoid(t[0]), t[1:]))
            return delta_e_2000(pred, y).mean()

        t = _fit_params(loss, [0.0, *self.extra0])
        self.w, self.extra = float(_sigmoid(t[0])), t[1:]
        return self

    def predict(self, ds, idx):
        return self._lab(self.mix(self._R(ds.warp_lab[idx]), self._R(ds.weft_lab[idx]), self.w, self.extra))


def _ks(R):
    return (1 - R) ** 2 / (2 * R)


def _ks_inv(ks):
    ks = np.maximum(ks, 0)
    return 1 + ks - np.sqrt(ks**2 + 2 * ks)


class SimpleKS(ReflectanceMixModel):
    """Simple Kubelka-Munk: (K/S)_mix = Σ c_i (K/S)_i  (Eq. 4)."""
    name = "Simple K/S"
    f = staticmethod(lambda R, e: _ks(R))
    f_inv = staticmethod(lambda v, e: _ks_inv(v))


class LogKS(ReflectanceMixModel):
    """Reed et al. (2004): log (K/S)_mix = Σ c_i log (K/S)_i  (Eq. 9)."""
    name = "log K/S"
    f = staticmethod(lambda R, e: np.log(np.maximum(_ks(R), 1e-9)))
    f_inv = staticmethod(lambda v, e: _ks_inv(np.exp(v)))


class StearnsNoechel(ReflectanceMixModel):
    """Stearns & Noechel (1944): f(R) = (1-R) / (b (R-0.01) + 0.01), b fitted  (Eq. 5-6)."""
    name = "S-N (Stearns-Noechel)"
    extra0 = (np.log(0.15),)

    @staticmethod
    def f(R, e):
        b = np.exp(e[0])
        return (1 - R) / (b * (R - 0.01) + 0.01)

    @staticmethod
    def f_inv(v, e):
        b = np.exp(e[0])
        return (1 - 0.01 * v * (1 - b)) / (1 + b * v)


class ModifiedSN(ReflectanceMixModel):
    """Philips-Invernizzi et al. (2002): b(λ) = (0.12 λ + 42.75) / 1000  (Eq. 7-8).

    No spectrum, so each tristimulus channel uses the wavelength near its peak
    (X ≈ 600 nm, Y ≈ 555 nm, Z ≈ 450 nm). No free parameter besides w."""
    name = "modified S-N (ประมาณ λ)"
    B = (0.12 * np.array([600.0, 555.0, 450.0]) + 42.75) / 1000

    def f(self, R, e):
        return (1 - R) / (self.B * (R - 0.01) + 0.01)

    def f_inv(self, v, e):
        return (1 - 0.01 * v * (1 - self.B)) / (1 + self.B * v)


class WarburtonOliver(ReflectanceMixModel):
    """Warburton & Oliver (1956): R = R_w^x R_f^(1-x)  (Eq. 13)."""
    name = "W-O (Warburton-Oliver)"
    f = staticmethod(lambda R, e: np.log(R))
    f_inv = staticmethod(lambda v, e: np.exp(v))


class Friele(ReflectanceMixModel):
    """Friele (1952): F(R) = exp(-σ (1-R)^2 / (2R)), σ fitted."""
    name = "Friele"
    extra0 = (np.log(0.1),)

    @staticmethod
    def f(R, e):
        return np.exp(-np.exp(e[0]) * _ks(R))

    @staticmethod
    def f_inv(v, e):
        return _ks_inv(-np.log(np.clip(v, 1e-12, 1.0)) / np.exp(e[0]))


class YuleNielsen(ReflectanceMixModel):
    """Yule-Nielsen: R^(1/n) additive, plus gain k for shading between threads.

        XYZ = k (w XYZ_w^(1/n) + (1-w) XYZ_f^(1/n))^n
    """
    name = "Yule-Nielsen (n, w, k)"
    extra0 = (0.0, 0.0)   # log(n-1), log k

    @staticmethod
    def f(R, e):
        return R ** (1 / (1 + np.exp(e[0])))

    @staticmethod
    def f_inv(v, e):
        return np.exp(e[1]) * np.maximum(v, 0) ** (1 + np.exp(e[0]))

    @property
    def params(self):
        return {"n": float(1 + np.exp(self.extra[0])), "w": self.w, "k": float(np.exp(self.extra[1]))}


PHYSICS = [DimitrovskiGabrijelcic, SimpleKS, LogKS, StearnsNoechel, ModifiedSN, WarburtonOliver,
           Friele, YuleNielsen]


# ---------------------------------------------------------------- ML learners
def _gp(dims=6):
    kernel = (ConstantKernel(1.0, (1e-2, 1e2))
              * RBF(length_scale=np.ones(dims), length_scale_bounds=(1e-2, 1e2))
              + WhiteKernel(1e-2, (1e-5, 1e1)))
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=0)


# name -> (estimator, hyper-parameter grid). Grids are searched with inner 3-fold CV
# on the training fold only (nested CV), so no test data leaks into tuning.
LEARNERS = {
    "GP": (_gp(), None),   # kernel hyper-parameters are fitted by marginal likelihood
    "Polynomial Ridge": (
        make_pipeline(StandardScaler(), PolynomialFeatures(), Ridge()),
        {"polynomialfeatures__degree": [2, 3], "ridge__alpha": [0.01, 0.1, 1, 10]},
    ),
    "KNN": (
        make_pipeline(StandardScaler(), KNeighborsRegressor()),
        {"kneighborsregressor__n_neighbors": [3, 5, 8, 12], "kneighborsregressor__weights": ["uniform", "distance"]},
    ),
    "SVR": (
        make_pipeline(StandardScaler(), MultiOutputRegressor(SVR())),
        {"multioutputregressor__estimator__C": [1, 10, 100],
         "multioutputregressor__estimator__epsilon": [0.1, 0.5],
         "multioutputregressor__estimator__gamma": ["scale", 0.1]},
    ),
    "Random Forest": (
        RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1),
        {"max_features": [0.5, 1.0], "min_samples_leaf": [1, 3]},
    ),
    "XGBoost": (
        XGBRegressor(n_estimators=400, tree_method="hist", random_state=0, n_jobs=4, verbosity=0),
        {"max_depth": [3, 5], "learning_rate": [0.03, 0.1], "subsample": [0.8, 1.0]},
    ),
    "MLP (BPNN)": (
        make_pipeline(StandardScaler(), MLPRegressor(max_iter=3000, early_stopping=True, n_iter_no_change=50,
                                                     random_state=0)),
        {"mlpregressor__hidden_layer_sizes": [(32,), (64, 32), (128, 64)], "mlpregressor__alpha": [1e-4, 1e-2, 1e-1]},
    ),
}


def _tuned(learner_name, cv=None):
    """Grid search on the training fold. cv=None: shuffled 3-fold inner CV; otherwise a list of
    (train, validation) position arrays that mimic the outer split scheme (grouped inner CV)."""
    est, grid = LEARNERS[learner_name]
    if grid is None:
        return clone(est)
    inner = KFold(3, shuffle=True, random_state=0) if cv is None else cv
    return GridSearchCV(clone(est), grid, cv=inner, scoring="neg_mean_squared_error", n_jobs=-1)


class MLModel:
    """Pure ML: learner maps the six yarn L*a*b* values straight to fabric L*a*b*."""

    family = "ml"

    def __init__(self, learner):
        self.learner = learner
        self.name = f"ML ล้วน: {learner}"

    def fit(self, ds, idx, cv=None):
        self.est = _tuned(self.learner, cv).fit(features(ds, idx), ds.lab[idx])
        return self

    def predict(self, ds, idx):
        return self.est.predict(features(ds, idx))


class HybridModel:
    """Hybrid: physics model, then an ML learner fitted on the residual (measured - physics)."""

    family = "hybrid"

    def __init__(self, physics_cls, learner, white=DEFAULT_WHITE):
        self.physics = physics_cls(white)
        self.learner = learner
        self.name = f"Hybrid: {physics_cls.name.split(' (')[0]} + {learner}"

    def fit(self, ds, idx, cv=None):
        self.physics.fit(ds, idx)
        resid = ds.lab[idx] - self.physics.predict(ds, idx)
        self.est = _tuned(self.learner, cv).fit(features(ds, idx), resid)
        return self

    def predict(self, ds, idx, return_std=False):
        base = self.physics.predict(ds, idx)
        if return_std:
            corr, std = self.est.predict(features(ds, idx), return_std=True)
            return base + corr, std
        return base + self.est.predict(features(ds, idx))


# ---------------------------------------------------------------- CNN-A (photo features)
class CNNYarnPhotos:
    """CNN-A: ResNet-18 (ImageNet) features of the warp and weft yarn photos + a tuned head.

    Features come from data/images/yarn_embeddings_resnet18.npz, written by
    `python -m weavecolor.cnn`, so this experiment never imports PyTorch."""

    family = "cnn"

    def __init__(self, head="Ridge"):
        self.head = head
        self.name = f"CNN-A: ResNet-18 ภาพด้าย + {head}"

    @staticmethod
    def available():
        from .data import ROOT
        return (ROOT / "data" / "images" / "yarn_embeddings_resnet18.npz").exists()

    def _x(self, ds, idx):
        from .data import ROOT
        z = np.load(ROOT / "data" / "images" / "yarn_embeddings_resnet18.npz")
        emb = dict(zip(z["codes"], z["emb"]))
        return np.hstack([np.stack([emb[c] for c in ds.warp[idx]]), np.stack([emb[c] for c in ds.weft[idx]])])

    def fit(self, ds, idx, cv=None):
        if self.head == "Ridge":
            from sklearn.linear_model import RidgeCV
            # cv=None: efficient leave-one-out; otherwise the grouped inner splits
            self.est = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 12), cv=cv))
        elif self.head == "GP":
            # 1024 photo features: compress to 8 whitened PCA scores before the ARD Gaussian
            # process (unwhitened scores have SD ~9 and push length scales to their bounds)
            from sklearn.decomposition import PCA
            self.est = make_pipeline(StandardScaler(), PCA(n_components=8, whiten=True, random_state=0), _gp(8))
        else:
            self.est = _tuned(self.head, cv)
        self.est.fit(self._x(ds, idx), ds.lab[idx])
        return self

    def predict(self, ds, idx):
        return self.est.predict(self._x(ds, idx))


# ---------------------------------------------------------------- hybrid with uncertainty
class HybridEnsemble:
    """Hybrid with an uncertainty estimate: physics model + a bagged ensemble of tuned learners.

    Hyper-parameters are tuned once on the training fold (inner CV), then n_members
    copies are refitted on bootstrap resamples with different seeds. The spread of the
    members' predictions is the uncertainty: predict(..., return_std=True) returns the
    per-channel standard deviation in L*a*b* units.
    """

    family = "hybrid"

    def __init__(self, physics_cls=StearnsNoechel, learner="XGBoost", n_members=10, white=DEFAULT_WHITE):
        self.physics = physics_cls(white)
        self.learner = learner
        self.n_members = n_members
        self.name = f"Hybrid: {physics_cls.name.split(' (')[0]} + {learner} ensemble×{n_members}"

    def fit(self, ds, idx, cv=None):
        self.physics.fit(ds, idx)
        X, resid = features(ds, idx), ds.lab[idx] - self.physics.predict(ds, idx)
        tuned = _tuned(self.learner, cv).fit(X, resid)
        best = tuned.best_estimator_ if hasattr(tuned, "best_estimator_") else tuned
        rng = np.random.default_rng(0)
        self.members = []
        for m in range(self.n_members):
            boot = rng.integers(0, len(idx), len(idx))
            est = clone(best)
            if "random_state" in est.get_params():
                est.set_params(random_state=m)
            self.members.append(est.fit(X[boot], resid[boot]))
        return self

    def predict(self, ds, idx, return_std=False):
        base = self.physics.predict(ds, idx)
        X = features(ds, idx)
        preds = np.stack([m.predict(X) for m in self.members])
        mean = base + preds.mean(0)
        return (mean, preds.std(0)) if return_std else mean
