"""m5.stats - the statistics behind every M5 number (plan §6), in numpy with scipy optional.

Pure functions and small classes; numpy arrays in and out; no torch. scipy is used when it is
importable and replaced by tested numpy/Python implementations when it is not (`have_scipy()`
says which), so a report computed on a pod without scipy carries the same numbers.

Arguments are explicit: nothing here reads `m5.config`. Callers pass the config values by exact
key (CONFIDENCE, DELTA, Q, FRONTIER_Q, PRIOR_*, SPRT_*, LAYER_AR1, LAYER_JUMP_DF, ...), so a
statistic can never run under a defaulted threshold. Every function takes valid data only: the
caller excludes missing trials first (missing is not zero, implementation rule 5).

    intervals     Wilson, Clopper-Pearson, Newcombe difference, TOST equivalence, rate_with_null
    clustering    ICC (ANOVA), design effect, effective n, cluster bootstrap
    correction    Rogan-Gladen with propagated uncertainty, PPI++ for a mean, label budgets
    sdt           d-prime and criterion (log-linear correction), TPR at fixed FPR
    paired        exact McNemar; Tango's score interval for a paired difference
    psychometric  the per-layer grid posterior (floor, lapse, overdispersion), expected
                  information gain, predictive checks, and the heavy-tailed cross-layer chain
    effect        isotonic regression, the four-parameter (Hill) curve, prompt bootstrap
    sequential    truncated SPRT with exact operating characteristics; betting confidence sequence
    halving       successive-halving keep rule; APT thresholding-bandit index
    summaries     R*(q), the frontier, W*, P_win and A_win from aligned posterior draws
    selection     Learn-then-Test fixed-sequence certification, EB shrinkage, winner's curse
    rank          Spearman (ties), within-strata rank correlation, AUC (DeLong), Cohen's kappa
    beta, normal  the special functions, scipy optional
"""

from __future__ import annotations

from m5.stats.beta import betainc, betaincinv, binom_cdf, binom_sf, have_scipy, lgamma
from m5.stats.clustering import (
    BootstrapResult,
    cluster_bootstrap,
    design_effect,
    effective_n,
    icc_binary,
    icc_from_counts,
    mean_cluster_size,
)
from m5.stats.correction import (
    CorrectedRate,
    PPIResult,
    label_budget,
    ppi_mean,
    rogan_gladen,
    rogan_gladen_interval,
)
from m5.stats.effect import (
    EffectBootstrap,
    HillFit,
    bootstrap_hill,
    fit_hill,
    half_plateau_dose,
    isotonic,
    isotonic_by_dose,
    isotonic_half_dose,
)
from m5.stats.halving import apt_index, apt_next, keep_layers
from m5.stats.intervals import (
    Equivalence,
    clopper_pearson,
    clopper_pearson_lower,
    clopper_pearson_upper,
    newcombe_diff,
    rate_with_null,
    tost_rate_difference,
    wilson,
    wilson_lower,
    wilson_upper,
)
from m5.stats.normal import norm_cdf, norm_ppf, z_one_sided, z_two_sided
from m5.stats.paired import mcnemar_exact, paired_rate_difference
from m5.stats.psychometric import (
    GridPrior,
    LayerChain,
    LayerPosterior,
    ThresholdSummary,
    beta_quantile_grid,
    detection_curve,
    detection_threshold,
    make_mu_grid,
    mixture_mu_logprior,
    psychometric_prior,
)
from m5.stats.rank import (
    auc,
    bootstrap_spearman,
    cohen_kappa,
    rankdata,
    spearman,
    within_strata_spearman,
)
from m5.stats.sdt import SDT, dprime, tpr_at_fpr
from m5.stats.selection import (
    FixedSequenceResult,
    Shrinkage,
    WinnersCurse,
    binomial_tail_pvalue,
    eb_shrink,
    learn_then_test_fixed_sequence,
    winners_curse_sim,
)
from m5.stats.sequential import ACCEPT_H0, ACCEPT_H1, CONTINUE, SPRT, BettingCS, OperatingCharacteristic
from m5.stats.summaries import LayerDraws, combine_draws, frontier, layer_terms, p_win, r_star, w_star

__all__ = [
    "ACCEPT_H0", "ACCEPT_H1", "CONTINUE", "SPRT", "SDT", "BettingCS", "BootstrapResult",
    "CorrectedRate", "EffectBootstrap", "Equivalence", "FixedSequenceResult", "GridPrior",
    "HillFit", "LayerChain", "LayerDraws", "LayerPosterior", "OperatingCharacteristic", "PPIResult",
    "Shrinkage", "ThresholdSummary", "WinnersCurse", "apt_index", "apt_next", "auc",
    "beta_quantile_grid", "betainc", "betaincinv", "binom_cdf", "binom_sf", "binomial_tail_pvalue",
    "bootstrap_hill", "bootstrap_spearman", "clopper_pearson", "clopper_pearson_lower",
    "clopper_pearson_upper", "cluster_bootstrap", "cohen_kappa", "combine_draws", "design_effect",
    "detection_curve", "detection_threshold", "dprime", "eb_shrink", "effective_n", "fit_hill",
    "frontier", "half_plateau_dose", "have_scipy", "icc_binary", "icc_from_counts", "isotonic",
    "isotonic_by_dose", "isotonic_half_dose", "keep_layers", "label_budget", "layer_terms",
    "learn_then_test_fixed_sequence", "lgamma", "make_mu_grid", "mcnemar_exact",
    "mean_cluster_size", "mixture_mu_logprior", "newcombe_diff", "norm_cdf", "norm_ppf", "p_win",
    "paired_rate_difference", "ppi_mean", "psychometric_prior", "r_star", "rankdata",
    "rate_with_null", "rogan_gladen", "rogan_gladen_interval", "spearman", "tost_rate_difference",
    "tpr_at_fpr", "w_star", "wilson", "wilson_lower", "wilson_upper", "winners_curse_sim",
    "within_strata_spearman", "z_one_sided", "z_two_sided",
]
