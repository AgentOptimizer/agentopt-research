"""Discrete qNEHVI over categorical agent configurations.

The live Bayesian selector in :mod:`agentopt.model_selection.bayesian_optimization`
fits a MixedSingleTaskGP and scores single-objective Expected Improvement.
This module is the two-objective counterpart used by the offline Pareto
baselines: a ModelListGP of two MixedSingleTaskGPs and BoTorch's noisy
expected hypervolume improvement, evaluated on a finite candidate set.

qNEHVI has no identification stopping time. Callers run it until the
observation budget is exhausted.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np


def _require_botorch() -> None:
    try:
        import torch  # noqa: F401
        from botorch.acquisition.multi_objective.monte_carlo import (  # noqa: F401
            qNoisyExpectedHypervolumeImprovement,
        )
        from botorch.models.gp_regression_mixed import MixedSingleTaskGP  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "qNEHVI requires optional dependencies. "
            'Install with `pip install -e ".[bayesian]"`'
        ) from exc


def encode_configuration_features(
    model_names: Sequence[str],
) -> Tuple[np.ndarray, List[int]]:
    """Map configuration names to integer categorical features.

    Two-role names ``planner=X + solver=Y`` become a length-2 code.  A flat
    list of model names becomes a single categorical column.  Mixed naming
    styles fall back to one categorical id per configuration.
    """
    names = [str(name) for name in model_names]
    if not names:
        raise ValueError("model_names must be nonempty")
    parsed: List[Optional[List[Tuple[str, str]]]] = []
    for name in names:
        if " + " not in name:
            parsed.append(None)
            continue
        roles: List[Tuple[str, str]] = []
        ok = True
        for part in name.split(" + "):
            if "=" not in part:
                ok = False
                break
            role, value = part.split("=", 1)
            roles.append((role.strip(), value.strip()))
        parsed.append(roles if ok else None)

    if all(item is None for item in parsed) or any(item is None for item in parsed):
        unique = {name: index for index, name in enumerate(sorted(set(names)))}
        features = np.asarray([[unique[name]] for name in names], dtype=np.float64)
        return features, [0]

    n_roles = len(parsed[0] or ())
    if any(item is None or len(item) != n_roles for item in parsed):
        unique = {name: index for index, name in enumerate(sorted(set(names)))}
        features = np.asarray([[unique[name]] for name in names], dtype=np.float64)
        return features, [0]

    value_levels = [
        sorted({item[role][1] for item in parsed if item is not None})
        for role in range(n_roles)
    ]
    features = np.asarray(
        [
            [value_levels[role].index(item[role][1]) for role in range(n_roles)]
            for item in parsed
            if item is not None
        ],
        dtype=np.float64,
    )
    return features, list(range(n_roles))


def select_qnehvi_index(
    train_features: np.ndarray,
    train_objectives: np.ndarray,
    candidate_features: np.ndarray,
    *,
    categorical_dims: Sequence[int],
    reference_point: Sequence[float] = (0.0, 0.0),
    mc_samples: int = 64,
    seed: int = 0,
    candidate_batch_size: Optional[int] = 64,
) -> int:
    """Return the candidate index with largest qNEHVI.

    ``train_objectives`` and the acquisition are maximization-valued.
    ``candidate_features`` may repeat previously observed configurations;
    qNEHVI treats those as additional noisy evaluations of the same point.
    """
    _require_botorch()
    import torch
    from botorch.acquisition.multi_objective.monte_carlo import (
        qNoisyExpectedHypervolumeImprovement,
    )
    from botorch.fit import fit_gpytorch_mll
    from botorch.models.gp_regression_mixed import MixedSingleTaskGP
    from botorch.models.model_list_gp_regression import ModelListGP
    from botorch.models.transforms.outcome import Standardize
    try:
        from botorch.sampling.normal import SobolQMCNormalSampler
    except ImportError:  # BoTorch < 0.10
        from botorch.sampling.samplers import SobolQMCNormalSampler
    from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

    train_x = _finite_2d(train_features, "train_features")
    train_y = _finite_2d(train_objectives, "train_objectives")
    cand_x = _finite_2d(candidate_features, "candidate_features")
    if train_x.shape[0] != train_y.shape[0]:
        raise ValueError("train_features and train_objectives must have the same rows")
    if train_y.shape[1] != 2:
        raise ValueError("train_objectives must have two columns")
    if cand_x.shape[1] != train_x.shape[1]:
        raise ValueError("candidate_features must match train_features columns")
    if cand_x.shape[0] < 1:
        raise ValueError("candidate_features must be nonempty")
    ref = np.asarray(reference_point, dtype=np.float64)
    if ref.shape != (2,) or not np.all(np.isfinite(ref)):
        raise ValueError("reference_point must be a finite length-2 vector")
    cat_dims = [int(dim) for dim in categorical_dims]
    if not cat_dims:
        raise ValueError("categorical_dims must be nonempty")
    if candidate_batch_size is not None and int(candidate_batch_size) < 1:
        raise ValueError("candidate_batch_size must be positive or None")

    torch_seed = int(seed) % (2**31)
    torch.manual_seed(torch_seed)
    dtype = torch.float64
    x_train = torch.as_tensor(train_x, dtype=dtype)
    y_train = torch.as_tensor(train_y, dtype=dtype)
    x_cand = torch.as_tensor(cand_x, dtype=dtype)
    ref_t = torch.as_tensor(ref, dtype=dtype)

    outcomes: List[Any] = []
    for obj in range(2):
        outcomes.append(
            MixedSingleTaskGP(
                train_X=x_train,
                train_Y=y_train[:, obj : obj + 1],
                cat_dims=cat_dims,
                outcome_transform=Standardize(m=1),
            )
        )
    model = ModelListGP(*outcomes)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([int(mc_samples)]), seed=torch_seed)
    acq = qNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_t,
        X_baseline=x_train,
        sampler=sampler,
        prune_baseline=True,
    )
    batch_size = (
        cand_x.shape[0]
        if candidate_batch_size is None
        else min(int(candidate_batch_size), cand_x.shape[0])
    )
    score_chunks = []
    with torch.no_grad():
        for start in range(0, cand_x.shape[0], batch_size):
            values = acq(x_cand[start : start + batch_size].unsqueeze(1))
            score_chunks.append(values.detach().cpu().reshape(-1))
    scores = torch.cat(score_chunks).numpy()
    if scores.size != cand_x.shape[0]:
        raise RuntimeError("qNEHVI returned a score vector of unexpected length")
    return int(np.argmax(scores))


def _finite_2d(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{name} must have shape (n, d) with n, d ≥ 1")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array
