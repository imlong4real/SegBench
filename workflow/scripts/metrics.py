#----------------------·•●  🧽  ●•·-------------------------
#                   TRACER Metrics Module
#----------------------·•●──────●•·-------------------------
# Author: Long Yuan
# Affiliation: Johns Hopkins University
# Email: lyuan13@jhmi.edu
#-----------------------------------------------------------

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import geopandas as gpd
import scipy.sparse as sp

from ._kernels import pair_aggregate_dense

#
def get_confident_nuclei_transcripts(
    sdata,
    *,
    qv_min: float = 30,
    low_pct: float = 20,
    high_pct: float = 80,
    save_qv_filtered: bool = False,
    parquet_path: str = "qv_filtered_transcripts.parquet",
    exclude_ids: set | None = None,
):
    """
    From a SpatialData object, extract high-quality nucleus transcripts and
    return a confident nucleus DataFrame (nuc_df_confident).

    Parameters
    ----------
    sdata : SpatialData
        The loaded SpatialData object.
    qv_min : float
        Minimum qv to keep.
    low_pct : float
        Lower percentile threshold for nucleus transcript count.
    high_pct : float
        Upper percentile threshold for nucleus transcript count.
    save_qv_filtered : bool
        If True, save the QV-filtered transcripts to a Parquet file.
    parquet_path : str
        Path to save QV-filtered transcripts if requested.
    exclude_ids : set | None, optional (default=None)
        Set of cell IDs to exclude, e.g. {"-1", "DROP", "nan", "UNASSIGNED"}.
        If None, defaults to {"UNASSIGNED"}.

    Returns
    -------
    nuc_df_confident : DataFrame
        Transcripts belonging to confident nuclei.
    fitlered_df : DataFrame
        Transcripts passing the qv threshold.
    """

    # Load transcripts
    transcripts = sdata.points["transcripts"].compute()

    # Apply QV filter
    df = transcripts[transcripts["qv"] >= qv_min].copy()

    # Filter to valid gene list
    # Ensures we only keep transcripts whose gene exists in AnnData table
    valid_genes = set(sdata.tables["table"].var.index)
    df = df[df["feature_name"].isin(valid_genes)].copy()

    # Optionally save the qv-filtered transcripts
    if save_qv_filtered:
        df.to_parquet(parquet_path, index=False)
        qv_out = parquet_path
        print("Saved parquet:", qv_out)
    else:
        qv_out = None

    # Extract nucleus-overlapping transcripts with a valid cell_id
    if exclude_ids is None:
        exclude_ids = {"UNASSIGNED"}
    
    if exclude_ids:
        nuc_df = df[
            (~df["cell_id"].isin(exclude_ids)) &
            (df["overlaps_nucleus"] == 1)
        ].copy()
    else:
        nuc_df = df[df["overlaps_nucleus"] == 1].copy()

    # Compute transcript-count thresholds per nucleus
    nuc_counts = nuc_df.groupby("cell_id").size()

    low_thres = np.percentile(nuc_counts, low_pct)
    high_thres = np.percentile(nuc_counts, high_pct)
    print("Transcript count thresholds:", low_thres, high_thres)

    # Identify confident nuclei
    good_ids = nuc_counts[(nuc_counts >= low_thres) & (nuc_counts <= high_thres)].index
    
    print("Number of confident nuclei:", len(good_ids))
    nuc_df_confident = nuc_df[nuc_df["cell_id"].isin(good_ids)].copy()

    return nuc_df_confident, df

#
def compute_npmi(
    df_subset,
    group_key="cell_id",
    min_occurrences_per_context=2,
    count_col=None,
    set_neg_one=False,
    thr=0.05
):
    """
    Compute PMI/NPMI using presence/absence of genes at the cell or nucleus level,
    with robustness control by requiring each gene to occur at least N times
    within a context (cell or nucleus) before being considered "present".
    Optional:
    set_neg_one : bool
        If True, assigns NPMI = -1 for gene pairs with zero observed
        co-occurrence (P_ij == 0) when both marginal probabilities
        exceed thr.
    thr : float
        Marginal probability threshold used for the optional -1
        assignment (default 0.05).
    -------
    long_df : DataFrame
        Columns:
            gene_i, gene_j, P_i, P_j, P_ij,
            P_i_given_j, P_j_given_i, PMI, NPMI
    """

    # 0. Minimal column projection (no .copy() of the entire 100M-row frame)
    if count_col is None:
        df = df_subset[[group_key, "feature_name"]]
    else:
        df = df_subset[[group_key, "feature_name", count_col]]
    group_series = df[group_key].astype(str)

    # ----------------------------------------------------------------------
    # Filter by minimum occurrences per context
    # ----------------------------------------------------------------------
    if count_col is None:
        counts = (
            df.assign(_grp=group_series)
              .groupby(["_grp", "feature_name"])
              .size()
              .rename("gene_count")
              .reset_index()
              .rename(columns={"_grp": group_key})
        )
    else:
        counts = (
            df.assign(_grp=group_series)
              .groupby(["_grp", "feature_name"])[count_col]
              .sum()
              .rename("gene_count")
              .reset_index()
              .rename(columns={"_grp": group_key})
        )

    df_filtered = counts[counts["gene_count"] >= min_occurrences_per_context]
    if df_filtered.empty:
        raise ValueError(
            f"No genes pass min_occurrences_per_context={min_occurrences_per_context}."
        )

    # ----------------------------------------------------------------------
    # Build sparse contexts × genes presence matrix via categorical codes.
    # Previously used df.pivot_table(values=1, aggfunc="max", fill_value=0)
    # which densifies to C×G ints in pandas — for C=200K, G=500 that's
    # 800 MB of pandas overhead, independent of the actual sparsity.
    # ----------------------------------------------------------------------
    ctx_cat = pd.Categorical(df_filtered[group_key].astype(str))
    gene_cat = pd.Categorical(df_filtered["feature_name"].astype(str))

    rows_i = ctx_cat.codes.astype(np.int32)
    cols_i = gene_cat.codes.astype(np.int32)
    vals = np.ones(len(rows_i), dtype=np.int32)

    contexts = ctx_cat.categories.to_numpy()
    genes = gene_cat.categories.to_numpy()
    C = len(contexts)
    G_gene = len(genes)
    M = sp.coo_matrix(
        (vals, (rows_i, cols_i)), shape=(C, G_gene)
    ).tocsr()
    M.data = np.ones_like(M.data, dtype=np.int32)  # binarise

    # ----------------------------------------------------------------------
    # Probabilities P(i), P(i,j) — sparse co-occurrence matmul.
    # ----------------------------------------------------------------------
    counts_i = np.asarray(M.sum(axis=0)).ravel()
    P_i = counts_i / C

    # Sparse × sparse; returns sparse. Dense-ify for the elementwise ops
    # below — at G ≈ 500 the G×G matrix is 2 MB float64, trivial.
    co_matrix_sp = (M.T @ M)
    P_ij = np.asarray(co_matrix_sp.todense(), dtype=np.float64) / C

    # ----------------------------------------------------------------------
    # Conditional probabilities
    # ----------------------------------------------------------------------
    P_i_col = P_i[:, None]
    P_j_row = P_i[None, :]

    with np.errstate(divide="ignore", invalid="ignore"):
        P_i_given_j = np.where(P_j_row > 0, P_ij / P_j_row, np.nan)
        P_j_given_i = np.where(P_i_col > 0, P_ij / P_i_col, np.nan)

    # ----------------------------------------------------------------------
    # PMI & NPMI
    # ----------------------------------------------------------------------
    PMI = np.full_like(P_ij, np.nan)
    NPMI = np.full_like(P_ij, np.nan)

    denom = P_i_col * P_j_row
    valid = (P_ij > 0) & (denom > 0)

    with np.errstate(divide="ignore", invalid="ignore"):
        PMI[valid] = np.log(P_ij[valid] / denom[valid])
        NPMI[valid] = PMI[valid] / (-np.log(P_ij[valid]))

    # ----------------------------------------------------------------------
    # Optional: assign -1 if P_i > thr and P_j > thr and _P_ij = 0 (i.e. strong individual presence but no co-occurrence)
    # ----------------------------------------------------------------------
    if set_neg_one:
        zero_coocc = (P_ij == 0) & (P_i_col > thr) & (P_j_row > thr)
        NPMI[zero_coocc] = -1.0

    # ----------------------------------------------------------------------
    # Convert to long format
    # ----------------------------------------------------------------------
    G = len(genes)
    long_df = pd.DataFrame({
        "gene_i": np.repeat(genes, G),
        "gene_j": np.tile(genes, G),
        "P_i": np.repeat(P_i, G),
        "P_j": np.tile(P_i, G),
        "P_ij": P_ij.ravel(),
        "P_i_given_j": P_i_given_j.ravel(),
        "P_j_given_i": P_j_given_i.ravel(),
        "PMI": PMI.ravel(),
        "NPMI": NPMI.ravel(),
    })

    return long_df


# ---------------------------------------------------------------------
# Bootstrap NPMI with active sampling and sparse output.
# ---------------------------------------------------------------------

@dataclass
class NpmiBootstrapResult:
    """Output of :func:`compute_npmi_bootstrap`.

    ``W_sparse`` is a G×G upper-triangle CSR float32 with explicit
    nonzero entries only for pairs that the active-sampler classified as
    significantly above +tau (positive-settled), significantly below
    -tau (negative-settled), or assigned -1 by the dropout rule.

    Pairs that settled inside the dead zone, ran out of bootstrap budget
    while still unsettled, or had no observed cooccurrence and failed
    the dropout rule (indeterminate) are absent in CSR — i.e. encoded as
    zero. Per the design choice in the consolidation plan, downstream
    coherence code treats absent ≡ zero.
    """

    W_sparse: sp.csr_matrix
    genes: np.ndarray
    diagnostics: dict = field(default_factory=dict)
    pair_ci: pd.DataFrame | None = None


def _build_presence_matrix(
    df_subset: pd.DataFrame,
    *,
    group_key: str,
    feature_col: str,
    min_occurrences_per_context: int,
    count_col: str | None,
):
    """Build the contexts × genes binary CSR presence matrix.

    Mirrors the construction in :func:`compute_npmi` so the bootstrap
    function and the legacy single-pass function share the same gene /
    context vocabulary semantics.
    """
    # Skip redundant astype(str) when input is already categorical or object
    # — those are the two dtypes the caller will hand us (categorical from
    # the memory_optimize path; object from raw load). Both work directly
    # with groupby + pd.Categorical without an intermediate string cast.
    # int/float-valued group keys would still get auto-stringified by
    # pd.Categorical's category-building, so functionality is preserved.
    if count_col is None:
        df = df_subset[[group_key, feature_col]]
        counts = (
            df.groupby([group_key, feature_col], observed=True, sort=False)
              .size()
              .rename("gene_count")
              .reset_index()
        )
    else:
        df = df_subset[[group_key, feature_col, count_col]]
        counts = (
            df.groupby([group_key, feature_col], observed=True, sort=False)[count_col]
              .sum()
              .rename("gene_count")
              .reset_index()
        )

    df_filtered = counts[counts["gene_count"] >= min_occurrences_per_context]
    if df_filtered.empty:
        raise ValueError(
            f"No genes pass min_occurrences_per_context={min_occurrences_per_context}."
        )

    # Use pd.factorize (not pd.Categorical) so the codes/categories are
    # derived strictly from VALUES PRESENT in `df_filtered`. If the input
    # column was already categorical (memory_optimize=True path), the
    # categorical's category dictionary may carry stale labels for cells
    # whose (cell, gene) rows were just dropped by the
    # `>= min_occurrences_per_context` filter — `pd.Categorical(cat_series)`
    # would preserve those, inflating C and breaking population statistics
    # like `expected_full = k_i*k_j / C` (→ false `indeterminate` vs
    # `neg_one` classifications). `pd.factorize` always reflects only
    # observed values, so memON and memOFF agree on M and downstream stats.
    rows_codes, contexts = pd.factorize(df_filtered[group_key], sort=True)
    cols_codes, genes = pd.factorize(df_filtered[feature_col], sort=True)
    rows = rows_codes.astype(np.int32)
    cols = cols_codes.astype(np.int32)
    contexts = np.asarray(contexts)
    genes = np.asarray(genes)
    # int32 for the binary presence values: M.T @ M sums these along the
    # dot product, so a pair cooccurring in >127 cells would overflow int8
    # and end up encoded as 0 (excluded from the sparse cooccurrence
    # matrix). int32 is safe for any realistic cell count.
    vals = np.ones(len(rows), dtype=np.int32)
    M = sp.coo_matrix(
        (vals, (rows, cols)), shape=(len(contexts), len(genes))
    ).tocsr()
    M.data = np.ones_like(M.data, dtype=np.int32)  # binarise
    return M, genes, contexts


def _bootstrap_npmi_for_pairs(
    M_sample: sp.csr_matrix,
    pairs_i: np.ndarray,
    pairs_j: np.ndarray,
    alpha: float = 0.0,
    metric: str = "npmi",
):
    """Vectorized PMI or NPMI for the given upper-triangle pair list on
    a bootstrap-sampled presence matrix.

    Returns a 1D float64 array of length ``len(pairs_i)`` in the
    requested ``metric`` (``"pmi"`` = log p_ij/(p_i p_j); ``"npmi"`` =
    PMI / -log(p_ij), bounded ±1).

    Parameters
    ----------
    alpha : float
        Beta(alpha, alpha) Jeffreys-style additive smoothing. ``alpha=0``
        is the unsmoothed estimator: pairs with ``k_ij=0`` in this
        bootstrap iteration return NaN (the caller filters these). With
        ``alpha>0`` (e.g. 0.5 for Jeffreys), smoothed probabilities are
        ``(k + alpha) / (N + 2 * alpha)``; every pair returns a finite
        value (no filtering needed). Smoothing eliminates the "log of
        zero" iter dropping, which biases the bootstrap median upward
        (less negative) for pairs with k_ij_full=1 by removing the most-
        negative iters from the sample.
    metric : {"npmi", "pmi"}
        Which scale to return. Affects per-iteration return value AND
        therefore the downstream bootstrap median/CI. tau thresholds in
        the parent ``compute_npmi_bootstrap`` are interpreted in this
        same metric.
    """
    if metric not in ("npmi", "pmi"):
        raise ValueError(f"metric must be 'npmi' or 'pmi' (got {metric!r})")
    N_b = M_sample.shape[0]
    marg = np.asarray(M_sample.sum(axis=0)).ravel().astype(np.int64)
    Mi = M_sample[:, pairs_i]
    Mj = M_sample[:, pairs_j]
    co = np.asarray(Mi.multiply(Mj).sum(axis=0)).ravel().astype(np.int64)

    if alpha > 0:
        # Jeffreys-style additive smoothing — all probabilities > 0, no
        # filtering needed.
        N_b_a = N_b + 2.0 * alpha
        Pij = (co + alpha) / N_b_a
        Pi  = (marg[pairs_i] + alpha) / N_b_a
        Pj  = (marg[pairs_j] + alpha) / N_b_a
        with np.errstate(divide="ignore", invalid="ignore"):
            pmi = np.log(Pij / (Pi * Pj))
            if metric == "pmi":
                out = pmi
            else:
                out = pmi / (-np.log(Pij))
        return out

    Pij = co / N_b
    Pi = marg[pairs_i] / N_b
    Pj = marg[pairs_j] / N_b
    out = np.full(co.shape[0], np.nan, dtype=np.float64)
    valid = (co > 0) & (Pi > 0) & (Pj > 0)
    if valid.any():
        with np.errstate(divide="ignore", invalid="ignore"):
            pmi = np.log(Pij[valid] / (Pi[valid] * Pj[valid]))
            if metric == "pmi":
                out[valid] = pmi
            else:
                out[valid] = pmi / (-np.log(Pij[valid]))
    return out


def compute_npmi_bootstrap(
    df_subset: pd.DataFrame,
    *,
    group_key: str = "cell_id",
    feature_col: str = "feature_name",
    min_occurrences_per_context: int = 2,
    count_col: str | None = None,
    tau: float | tuple[float, float] | list = 0.05,
    ci_level: float = 0.95,
    max_bootstraps: int = 10_000,
    coarse_block: int = 200,
    refine_block: int = 500,
    min_expected_cooccur_for_evidence: float = 10.0,
    min_samples_for_ci: int = 30,
    seed: int | None = None,
    show_progress: bool = False,
    persist_ci: bool = False,
    subsample_size: int | None = None,
    metric: str = "npmi",
    expected_cooccur_for_neg_one: float | None = None,  # deprecated alias
    alpha: float = 0.1,
    min_expected_cooccur_for_bootstrap: float | None = None,
    set_neg_one: bool = True,
    nuclear_only: bool = False,
    nucleus_col: str = "overlaps_nucleus",
    percentile_filter: tuple[float, float] | None = None,
    per_gene_percentile_filter: tuple[float, float] | None = None,
    exclude_contexts: set | list | None = None,
    memory_optimize: bool = True,
) -> NpmiBootstrapResult:
    """Bootstrap NPMI over contexts (cells) with active sampling.

    Returns sparse output: only pairs whose 95% bootstrap CI excludes
    [-tau, +tau] (on either side) get explicit entries, plus pairs
    assigned -1 by the dropout rule. Dead-zone, indeterminate, and
    unsettled-at-budget pairs are encoded as absent (= zero) in the
    output CSR.

    Early-stopping policy: only "strong" classifications (CI clearly
    outside ±tau_high) and "tight_null" (CI clearly inside ±tau_low)
    trigger early-stop. Weak / dead_zone classifications keep iterating
    to budget so their CI / median are not locked in by a CI that just
    barely crosses the lower threshold. Final classification of any
    still-unsettled pair is done at budget exhaustion using the final CI.

    Downstream ranking note: because settled-pair CI bounds cluster near
    the threshold by construction (a pair settles the moment CI crosses
    tau), CI bounds are biased estimates for ranking purposes. Use the
    per-pair `median` (point estimate from the bootstrap distribution)
    or `legacy_pmi`/`legacy_npmi` (full-data point estimates) for
    ranking, weighting, or threshold-based discrimination. CIs are for
    classification (is this pair above/below the threshold?), not for
    magnitude estimation.

    Parameters
    ----------
    df_subset : DataFrame
        Long-format transcripts with at least ``group_key`` and
        ``feature_col`` columns. Same contract as :func:`compute_npmi`.
    tau : float or 2-element sequence
        Dead-zone threshold(s). If scalar, single threshold (legacy
        behavior): pairs classified as ``pos`` (CI_lo > tau), ``neg``
        (CI_hi < -tau), ``dead_zone`` (CI inside ±tau), or ``unsettled``.
        If a 2-element sequence (tau_low, tau_high), the function emits
        7 kinds: ``pos_strong`` (CI_lo > tau_high), ``pos_weak`` (tau_low
        < CI_lo ≤ tau_high), ``neg_strong``, ``neg_weak``, ``tight_null``
        (CI inside ±tau_low), ``dead_zone`` (CI inside ±tau_high but
        straddles ±tau_low), ``unsettled``. The vector form lets the
        caller separate "is this real" (tau_low) from "is this strong"
        (tau_high) decisions.
    ci_level : float
        Confidence level for the bootstrap CI (0.95 → percentiles 2.5%
        and 97.5%).
    max_bootstraps : int
        Hard upper bound on bootstrap iterations.
    coarse_block, refine_block : int
        Block size for batched bootstrap iterations. The first
        ``coarse_block`` iterations sample every observed-cooccur pair;
        subsequent ``refine_block`` blocks only re-sample pairs still
        unsettled.
    expected_cooccur_for_neg_one : float
        Threshold on ``E[cooccur] = p_i * p_j * N`` above which a pair
        with zero observed cooccurrence is assigned NPMI = -1 (true
        mutual exclusivity). Pairs below this threshold are
        indeterminate and dropped to zero.
    min_samples_for_ci : int
        Minimum bootstrap samples required before a pair's CI is
        evaluated. Pairs with fewer samples (e.g. very rare cooccur)
        stay unsettled until they have enough or the budget runs out.
    seed : int or None
        rng seed for reproducibility.
    show_progress : bool
        Print iteration / settled-pair counts to stdout.
    persist_ci : bool
        When True, populate ``result.pair_ci`` with a per-pair DataFrame
        carrying ``(gene_i, gene_j, kind, median, ci_lo, ci_hi,
        n_bootstraps)``. Useful for sanity-checking the active sampler;
        off by default to keep the result lean.
    subsample_size : int or None
        When set, each bootstrap iteration samples this many cells with
        replacement instead of the full ``C``. Subsample bootstrap is
        much faster on large datasets (1M+ cells) at the cost of wider
        CIs — which makes the active sampler more conservative (more
        pairs land in the dead-zone or unsettled buckets).
    """
    # Backward-compat: `expected_cooccur_for_neg_one` was the old kwarg name
    # for the dropout-rule threshold. The new policy uses one threshold
    # (`min_expected_cooccur_for_evidence`) for all evidence claims, so we
    # accept the old name as an alias.
    if expected_cooccur_for_neg_one is not None:
        min_expected_cooccur_for_evidence = float(expected_cooccur_for_neg_one)

    if metric not in ("npmi", "pmi"):
        raise ValueError(f"metric must be 'npmi' or 'pmi' (got {metric!r})")

    # ------------------------------------------------------------------
    # Pre-filter pipeline + memory optimization.
    # Order matters:
    #   1. Drop excluded contexts (sentinels: "UNASSIGNED", "-1", ...) —
    #      always applied (default set), to prevent the unassigned-pool
    #      mega-context from poisoning the population statistics.
    #   2. nuclear_only: restrict to nucleus-overlapping tx (Long's QC).
    #   3. percentile_filter: drop contexts whose tx-count is outside
    #      [percentile(low), percentile(high)] of the per-context tx
    #      distribution. Long's defaults are (20, 80) → keep middle 60%.
    #   4. memory_optimize: trim columns + cast group_key/feature_col to
    #      categorical to cut peak DataFrame RSS by ~5x.
    # The math is invariant to memory_optimize because categorical labels
    # produce identical groupby/cooccurrence semantics as object strings.
    # ------------------------------------------------------------------
    _df = df_subset
    n_in = len(_df)
    n_dropped = {"excluded": 0, "nuclear": 0, "percentile": 0}

    # Step 1: exclude sentinel contexts. Default applies even if user passes
    # exclude_contexts=None (defensive). Pass exclude_contexts=set() to
    # bypass the default (rare; not recommended).
    if exclude_contexts is None:
        excl = {"UNASSIGNED", "-1", "DROP", "nan", "None", ""}
    else:
        excl = set(map(str, exclude_contexts))
    if excl:
        _grp_str = _df[group_key].astype(str)
        keep = ~_grp_str.isin(excl)
        n_dropped["excluded"] = int((~keep).sum())
        if not keep.all():
            _df = _df.loc[keep]

    # Step 2: nuclear-only filter
    if nuclear_only:
        if nucleus_col not in _df.columns:
            raise ValueError(
                f"nuclear_only=True requires column {nucleus_col!r} "
                f"(found columns: {list(_df.columns)})"
            )
        _nuc = _df[nucleus_col]
        if _nuc.dtype == bool:
            keep = _nuc
        else:
            # 1/0 int, "True"/"False" str, "1.0"/"0.0" float — coerce.
            keep = _nuc.astype(str).isin({"True", "true", "1", "1.0"}) | (
                pd.to_numeric(_nuc, errors="coerce") == 1
            )
        keep = keep.fillna(False).astype(bool)
        n_dropped["nuclear"] = int((~keep).sum())
        if not keep.all():
            _df = _df.loc[keep]

    # Step 3: percentile filter on per-context tx-count
    if percentile_filter is not None:
        low_pct, high_pct = float(percentile_filter[0]), float(percentile_filter[1])
        if not (0.0 <= low_pct < high_pct <= 100.0):
            raise ValueError(
                f"percentile_filter must satisfy 0 <= low < high <= 100 "
                f"(got {percentile_filter!r})"
            )
        # Per-context tx-count distribution AFTER prior filters.
        ctx_counts = _df.groupby(group_key, observed=True).size()
        low_thr = float(np.percentile(ctx_counts.values, low_pct))
        high_thr = float(np.percentile(ctx_counts.values, high_pct))
        good_ctx = set(ctx_counts[(ctx_counts >= low_thr) &
                                    (ctx_counts <= high_thr)].index.astype(str))
        keep = _df[group_key].astype(str).isin(good_ctx)
        n_dropped["percentile"] = int((~keep).sum())
        if not keep.all():
            _df = _df.loc[keep]

    # Step 3b: per-gene percentile filter — INFER per-gene size bands ONLY.
    # For each gene g, compute the per-cell total-tx distribution
    # restricted to cells where g passes `min_occurrences_per_context`,
    # then derive (n_min_g, n_max_g) percentile bounds from THAT
    # distribution. Apply later (after _build_presence_matrix) as a
    # SIZE-BASED admittance matrix `A` over the post-presence-filter
    # cell set: A[c, g] = 1 iff cell c.total_tx ∈ [n_min_g, n_max_g],
    # regardless of whether c expresses g.
    #
    # This is the (B″) framework: per-gene admittance is a property of
    # cell SIZE (where g is biologically expected), independent of the
    # observed presence of g. PMI for pair (i, j) is then computed under
    # per-pair scoping: universe = cells admitted for both i AND j,
    # marginals + cooccurrence all defined over the same pair-specific
    # universe. This avoids the asymmetric-scoping bias of the simpler
    # tx-level inner-join approach (which conflated admittance with
    # presence and biased PMI for cross-cell-type pairs).
    n_dropped["per_gene_percentile"] = 0
    pgp_bands: dict | None = None  # gene_name -> (n_min, n_max) when active
    pgp_cell_total: pd.Series | None = None
    if per_gene_percentile_filter is not None:
        pgp_lo, pgp_hi = float(per_gene_percentile_filter[0]), float(per_gene_percentile_filter[1])
        if not (0.0 <= pgp_lo < pgp_hi <= 100.0):
            raise ValueError(
                f"per_gene_percentile_filter must satisfy 0 <= low < high <= 100 "
                f"(got {per_gene_percentile_filter!r})"
            )
        # 1. Per-cell total tx (count of tx per cell, all genes; will be
        #    indexed by cell-id and re-aligned to M's row order later).
        pgp_cell_total = (_df.groupby(group_key, observed=True).size()
                              .rename("_total_tx"))
        # 2. Per-(cell, gene) count.
        cg = (_df.groupby([group_key, feature_col], observed=True).size()
                  .rename("_cg_count").reset_index())
        # 3. Restrict to g+ cells (count >= min_occurrences_per_context).
        cg = cg[cg["_cg_count"] >= min_occurrences_per_context]
        if cg.empty:
            pgp_bands = {}
        else:
            # 4. Join total-tx onto each g+ (cell, gene) row.
            cg = cg.join(pgp_cell_total, on=group_key)
            # 5. Per-gene percentile bands from g+ cells' total-tx.
            grp = cg.groupby(feature_col, observed=True)["_total_tx"]
            lo_per_gene = grp.quantile(pgp_lo / 100.0)
            hi_per_gene = grp.quantile(pgp_hi / 100.0)
            pgp_bands = {
                str(g): (float(lo_per_gene[g]), float(hi_per_gene[g]))
                for g in lo_per_gene.index
            }
        # NOTE: We do NOT inner-join _df. The size-band admittance A is
        # constructed AFTER _build_presence_matrix so it can align with
        # M's cell row order. n_dropped["per_gene_percentile"] stays 0
        # at the tx level — the filter's effect is at the matmul level.

    # Step 4: memory_optimize — trim to needed columns + categorical cast.
    if memory_optimize:
        wanted = [group_key, feature_col]
        if count_col is not None and count_col in _df.columns:
            wanted.append(count_col)
        # Drop unused columns; reset_index drops the legacy positional index.
        _df = _df[wanted].reset_index(drop=True)
        # Categorical cast of object/string columns. Categorical-from-categorical
        # is a no-op; categorical-from-object dedupes labels to int32 codes
        # plus a small categories dictionary, dropping per-row overhead from
        # ~70 bytes (object str) to 4 bytes.
        if not isinstance(_df[group_key].dtype, pd.CategoricalDtype):
            _df = _df.assign(**{group_key: _df[group_key].astype("category")})
        if not isinstance(_df[feature_col].dtype, pd.CategoricalDtype):
            _df = _df.assign(**{feature_col: _df[feature_col].astype("category")})

    df_subset = _df  # downstream code uses this name
    n_out = len(df_subset)
    pre_filter_diag = {
        "n_input_rows": int(n_in),
        "n_dropped_excluded": n_dropped["excluded"],
        "n_dropped_nuclear": n_dropped["nuclear"],
        "n_dropped_percentile": n_dropped["percentile"],
        "n_dropped_per_gene_percentile": n_dropped.get("per_gene_percentile", 0),
        "n_kept_rows": int(n_out),
        "exclude_contexts_applied": sorted(excl) if excl else [],
        "nuclear_only": bool(nuclear_only),
        "percentile_filter": list(percentile_filter) if percentile_filter is not None else None,
        "per_gene_percentile_filter": (list(per_gene_percentile_filter)
                                         if per_gene_percentile_filter is not None else None),
        "memory_optimize": bool(memory_optimize),
    }
    if show_progress:
        print(
            f"[bootstrap_npmi] pre-filter: {n_in:,} → {n_out:,} tx "
            f"(excluded={n_dropped['excluded']:,}, "
            f"nuclear={n_dropped['nuclear']:,}, "
            f"percentile={n_dropped['percentile']:,})",
            flush=True,
        )
        if memory_optimize:
            print(
                f"[bootstrap_npmi] memory_optimize: kept {len(df_subset.columns)} cols, "
                f"casted {group_key!r}/{feature_col!r} to categorical",
                flush=True,
            )

    # Parse tau: accept scalar (legacy) or 2-element vector (dual threshold).
    tau_arr = np.atleast_1d(np.asarray(tau, dtype=float))
    if tau_arr.size == 1:
        tau_low = tau_high = float(tau_arr[0])
        _is_dual_tau = False
    elif tau_arr.size == 2:
        tau_low = float(tau_arr.min())
        tau_high = float(tau_arr.max())
        _is_dual_tau = (tau_low < tau_high)
    else:
        raise ValueError(
            f"tau must be scalar or 2-element sequence (got shape {tau_arr.shape})"
        )

    rng = np.random.default_rng(seed)

    M, genes, contexts = _build_presence_matrix(
        df_subset,
        group_key=group_key,
        feature_col=feature_col,
        min_occurrences_per_context=min_occurrences_per_context,
        count_col=count_col,
    )
    C, G = M.shape

    # Build size-band admittance matrix A when per_gene_percentile_filter
    # is active. A has the same shape as M; A[c, g] = 1 iff cell c (i.e.,
    # contexts[c]) has total_tx in gene g's [n_min_g, n_max_g] band,
    # REGARDLESS of whether c expresses g. When A is None, downstream code
    # treats every (cell, gene) as admitted (legacy behavior).
    #
    # CRITICAL: M from _build_presence_matrix is "presence only". For
    # per-pair scoping the cooccurrence and marginal matmuls require
    # M_admit = M_presence · A (elementwise) — i.e., admitted+present —
    # because k_i_ij = (M_admit.T @ A)[i,j] needs cells admitted for BOTH
    # i and j AND with i present. We replace M with M_admit below.
    A: sp.csr_matrix | None = None
    if pgp_bands is not None and pgp_cell_total is not None and len(pgp_bands) > 0:
        ctx_total_tx = pgp_cell_total.reindex(contexts).to_numpy(dtype=np.int64)
        n_min_arr = np.array([pgp_bands.get(str(g), (np.nan, np.nan))[0]
                                for g in genes], dtype=np.float64)
        n_max_arr = np.array([pgp_bands.get(str(g), (np.nan, np.nan))[1]
                                for g in genes], dtype=np.float64)
        A_dense = (
            (ctx_total_tx[:, None] >= n_min_arr[None, :]) &
            (ctx_total_tx[:, None] <= n_max_arr[None, :])
        )
        A_dense &= np.isfinite(n_min_arr[None, :])
        A = sp.csr_matrix(A_dense.astype(np.int32))
        # M_admit = M_presence · A (elementwise sparse multiply).
        # Both M and A are CSR — sp.csr.multiply broadcasts elementwise.
        M = M.multiply(A).tocsr()
        # Re-binarize after multiply (multiply may yield non-{0,1} ints
        # for matrices that already had non-binary entries — defensive).
        M.data = np.ones_like(M.data, dtype=np.int32)
        if show_progress:
            a_density = float(A.nnz) / max(A.shape[0] * A.shape[1], 1)
            m_density = float(M.nnz) / max(M.shape[0] * M.shape[1], 1)
            print(
                f"[bootstrap_npmi] per_gene size-band admittance A: "
                f"shape={A.shape}, A_density={a_density:.3f}, "
                f"M_admit_density={m_density:.3f}, "
                f"bands_for_n_genes={len(pgp_bands)}", flush=True,
            )

    # Global marginals (computed from M, which IS M_admit when per-gene
    # filter is active — denoised presence). Standard PMI scoping: single
    # universe of size C for all pairs. The per-gene admittance affects
    # WHICH cells count as "g+" (via M_admit's elementwise mask), not
    # which cells are in the universe.
    marg_global = np.asarray(M.sum(axis=0)).ravel().astype(np.int64)
    p_global = marg_global / C

    iter_size = int(subsample_size) if subsample_size is not None else C
    if iter_size <= 0:
        raise ValueError(f"subsample_size must be positive (got {subsample_size!r})")

    # Observed cooccurrence (upper triangle). M.T @ M counts cells with
    # both genes admitted+present.
    co_full = (M.T @ M).tocoo()
    upper = co_full.row < co_full.col
    obs_i = co_full.row[upper].astype(np.int32)
    obs_j = co_full.col[upper].astype(np.int32)
    obs_k = co_full.data[upper].astype(np.int64)

    p_i_obs = p_global[obs_i]
    p_j_obs = p_global[obs_j]
    expected_full_obs = p_i_obs * p_j_obs * C
    expected_sub_obs  = p_i_obs * p_j_obs * iter_size

    # Pre-output sparse builder.
    out_rows: list[int] = []
    out_cols: list[int] = []
    out_vals: list[float] = []

    # CI records (only built when persist_ci=True).
    ci_records: list[tuple] | None = [] if persist_ci else None

    thr = float(min_expected_cooccur_for_evidence)

    # --------------------------------------------------------------------
    # Stage 1: handle pairs with observed k=0 separately. These are
    # absent from `obs_*` arrays. Iterate over the dense p_outer (G×G)
    # to enumerate them and apply the existing neg_one / indeterminate
    # / low_evidence classification.
    # --------------------------------------------------------------------
    n_neg_one = 0
    n_indeterminate = 0
    n_low_evidence_zero = 0  # k=0 pairs flagged low_evidence (E_full < thr)
    if G > 1:
        observed_set = set(zip(obs_i.tolist(), obs_j.tolist()))
        p_outer = np.outer(p_global, p_global)
        np.fill_diagonal(p_outer, 0.0)
        # Vectorised iteration over upper triangle:
        ti, tj = np.triu_indices(G, k=1)
        E_full_all = p_outer[ti, tj] * C  # (G*(G-1)/2,)
        for idx in range(ti.size):
            i = int(ti[idx]); j = int(tj[idx])
            if (i, j) in observed_set:
                continue
            E_full = float(E_full_all[idx])
            if E_full >= thr and set_neg_one:
                # k=0 with high expected count → mutual exclusion.
                # NPMI sentinel = -1 (perfect avoidance limit).
                # PMI sentinel  = -log(E_full)  (the PMI you'd see at k=1, the
                # smallest possible non-zero cooccur — a finite proxy for -∞).
                # When set_neg_one=False, this branch is skipped and the
                # pair falls through to the "indeterminate" classification
                # below — matches the legacy compute_npmi default.
                npmi_sentinel = -1.0
                pmi_sentinel = float(-np.log(E_full)) if E_full > 0 else np.nan
                w_value = pmi_sentinel if metric == "pmi" else npmi_sentinel
                out_rows.append(i); out_cols.append(j); out_vals.append(w_value)
                n_neg_one += 1
                if ci_records is not None:
                    # `median`/`ci_lo`/`ci_hi` are RESERVED for bootstrap output.
                    # neg_one didn't run bootstrap → NaN. The W-matrix sentinel
                    # value (npmi_sentinel/pmi_sentinel) is preserved on
                    # `legacy_npmi`/`legacy_pmi` columns.
                    ci_records.append((
                        i, j, str(genes[i]), str(genes[j]),
                        "neg_one",
                        npmi_sentinel, pmi_sentinel,
                        np.nan, np.nan, np.nan, 0,
                        E_full, float(p_outer[i, j] * iter_size),
                    ))
            else:
                # k=0 with low expected count → indeterminate / low_evidence
                # (we keep "indeterminate" name for back-compat: same outcome,
                # absent from W).
                n_indeterminate += 1
                if ci_records is not None:
                    ci_records.append((
                        i, j, str(genes[i]), str(genes[j]),
                        "indeterminate",
                        np.nan, np.nan,
                        np.nan, np.nan, np.nan, 0,
                        E_full, float(p_outer[i, j] * iter_size),
                    ))
        del p_outer, observed_set, ti, tj, E_full_all

    # --------------------------------------------------------------------
    # Stage 2: vectorised legacy NPMI and PMI for all observed-cooccur pairs.
    # Both are the canonical population estimates; which one is stored in
    # `W_sparse` depends on `metric`. `pair_ci` carries both.
    # --------------------------------------------------------------------
    p_ij_full = obs_k.astype(np.float64) / C
    legacy_npmi = np.full(obs_i.size, np.nan, dtype=np.float64)
    legacy_pmi = np.full(obs_i.size, np.nan, dtype=np.float64)
    valid = (p_ij_full > 0) & (p_i_obs > 0) & (p_j_obs > 0)
    if valid.any():
        with np.errstate(divide="ignore", invalid="ignore"):
            pmi_v = np.log(p_ij_full[valid] / (p_i_obs[valid] * p_j_obs[valid]))
            legacy_pmi[valid] = pmi_v
            legacy_npmi[valid] = pmi_v / (-np.log(p_ij_full[valid]))

    # Which of the two is the canonical W value depends on `metric`.
    legacy_for_W = legacy_pmi if metric == "pmi" else legacy_npmi

    # --------------------------------------------------------------------
    # Stage 3: classify each observed pair by evidence tier.
    #
    # Eligibility uses max(k_observed, E[k_ij]) rather than E[k_ij] alone.
    # Using only E[k] under H0 misses rare-cell-type-marker coexpression where
    # the marginals are tiny (so E[k]≪thr) but observed cooccur is large
    # (e.g., macrophage scavenger receptors at k=37 with E[k]=1.0). Those are
    # real biology, not rare-event PMI artifacts — the data IS sufficient.
    # --------------------------------------------------------------------
    obs_k_f = obs_k.astype(np.float64)
    observed_sub_obs = obs_k_f * iter_size / C
    high_evidence = np.maximum(obs_k_f, expected_full_obs) >= thr
    # Bootstrap-eligibility threshold. By default the same as evidence
    # threshold (preserves legacy behavior — pairs whose expected cooccurrence
    # in a bootstrap subsample is below thr are routed to "legacy_only" and
    # skipped to save compute). Setting `min_expected_cooccur_for_bootstrap`
    # lower (e.g., 0.0) lets ALL high-evidence pairs run the bootstrap, at
    # the cost of wider CIs for sparse pairs (and ~3-5x more wall time).
    thr_boot = (float(min_expected_cooccur_for_bootstrap)
                  if min_expected_cooccur_for_bootstrap is not None
                  else thr)
    can_bootstrap = high_evidence & (np.maximum(observed_sub_obs, expected_sub_obs) >= thr_boot)
    legacy_only_mask = high_evidence & ~can_bootstrap  # robust legacy, but bootstrap can't refine
    low_evidence_mask = ~high_evidence

    # low_evidence pairs (k_full > 0 but E_full < thr): value = 0 in W,
    # bootstrap not run → median/ci columns NaN (reserved for bootstrap output).
    n_low_evidence_obs = int(low_evidence_mask.sum())
    if ci_records is not None and n_low_evidence_obs:
        for k_idx in np.flatnonzero(low_evidence_mask):
            i = int(obs_i[k_idx]); j = int(obs_j[k_idx])
            ci_records.append((
                i, j, str(genes[i]), str(genes[j]),
                "low_evidence",
                float(legacy_npmi[k_idx]) if np.isfinite(legacy_npmi[k_idx]) else np.nan,
                float(legacy_pmi[k_idx]) if np.isfinite(legacy_pmi[k_idx]) else np.nan,
                np.nan, np.nan, np.nan, 0,
                float(expected_full_obs[k_idx]), float(expected_sub_obs[k_idx]),
            ))

    # legacy_only pairs: store legacy value (NPMI or PMI per `metric`) in W.
    # `median`/`ci_lo`/`ci_hi` are reserved for bootstrap output → NaN here.
    # The legacy point estimate lives on `legacy_npmi`/`legacy_pmi` columns.
    if legacy_only_mask.any():
        sel = np.flatnonzero(legacy_only_mask)
        for k_idx in sel:
            i = int(obs_i[k_idx]); j = int(obs_j[k_idx])
            v = legacy_for_W[k_idx]
            if np.isfinite(v):
                out_rows.append(i); out_cols.append(j); out_vals.append(float(v))
            if ci_records is not None:
                ci_records.append((
                    i, j, str(genes[i]), str(genes[j]),
                    "legacy_only",
                    float(legacy_npmi[k_idx]) if np.isfinite(legacy_npmi[k_idx]) else np.nan,
                    float(legacy_pmi[k_idx]) if np.isfinite(legacy_pmi[k_idx]) else np.nan,
                    np.nan, np.nan, np.nan, 0,
                    float(expected_full_obs[k_idx]), float(expected_sub_obs[k_idx]),
                ))

    n_legacy_only = int(legacy_only_mask.sum())
    # n_low_evidence: pairs with k>0 AND E_full < thr (kind="low_evidence")
    # Distinct from n_indeterminate (k=0 AND E_full < thr).
    n_low_evidence = n_low_evidence_obs
    n_can_bootstrap = int(can_bootstrap.sum())

    if n_can_bootstrap == 0:
        # No pairs eligible for the bootstrap loop; build output and return.
        diagnostics = {
            "n_neg_one": n_neg_one,
            "n_indeterminate": n_indeterminate,
            "n_low_evidence": n_low_evidence,
            "n_legacy_only": n_legacy_only,
            "n_dead_zone": 0,
            "n_pos": 0,
            "n_neg": 0,
            "n_unsettled": 0,
            "n_bootstraps_per_pair": np.zeros(0, dtype=np.int32),
            "subsample_size": iter_size,
            "min_expected_cooccur_for_evidence": thr,
            "pre_filter": pre_filter_diag,
        }
        W_sparse = sp.coo_matrix(
            (out_vals, (out_rows, out_cols)),
            shape=(G, G),
            dtype=np.float32,
        ).tocsr()
        pair_ci_df = _ci_records_to_df(ci_records) if ci_records is not None else None
        return NpmiBootstrapResult(
            W_sparse=W_sparse, genes=genes,
            diagnostics=diagnostics, pair_ci=pair_ci_df,
        )

    # --------------------------------------------------------------------
    # Stage 4: active sampling on bootstrap-eligible pairs.
    # --------------------------------------------------------------------
    boot_idx = np.flatnonzero(can_bootstrap)  # indices into obs_*
    pairs_i = obs_i[boot_idx]
    pairs_j = obs_j[boot_idx]
    legacy_npmi_boot = legacy_npmi[boot_idx]
    legacy_pmi_boot = legacy_pmi[boot_idx]
    legacy_for_W_boot = legacy_for_W[boot_idx]

    n_pairs = pairs_i.size
    unsettled = np.ones(n_pairs, dtype=bool)
    n_samples = np.zeros(n_pairs, dtype=np.int32)
    sample_lists: list[list[float]] = [[] for _ in range(n_pairs)]
    # settled_kind values:
    #   0  unsettled
    #   1  pos_strong  (CI_lo > tau_high)
    #   2  pos_weak    (tau_low < CI_lo ≤ tau_high) — only fires when _is_dual_tau
    #  -1  neg_strong  (CI_hi < -tau_high)
    #  -2  neg_weak    (-tau_high ≤ CI_hi < -tau_low) — only fires when _is_dual_tau
    #   3  tight_null  (CI inside ±tau_low) — only fires when _is_dual_tau
    #   4  dead_zone   (CI inside ±tau_high but straddles ±tau_low; collapses to "dead_zone"
    #                   in scalar mode where it's the same as kind=3)
    settled_kind = np.zeros(n_pairs, dtype=np.int8)
    # Track at which n_done each pair settled. -1 = never settled.
    settled_at_n_done = np.full(n_pairs, -1, dtype=np.int32)

    ci_lo_q = (1.0 - ci_level) / 2.0
    ci_hi_q = 1.0 - ci_lo_q

    if persist_ci:
        per_pair_ci_lo = np.full(n_pairs, np.nan, dtype=np.float64)
        per_pair_ci_hi = np.full(n_pairs, np.nan, dtype=np.float64)
        per_pair_median = np.full(n_pairs, np.nan, dtype=np.float64)
    else:
        per_pair_ci_lo = per_pair_ci_hi = per_pair_median = None

    n_done = 0
    while n_done < max_bootstraps and unsettled.any():
        block = coarse_block if n_done == 0 else refine_block
        block = min(block, max_bootstraps - n_done)
        un_idx = np.flatnonzero(unsettled)
        i_un = pairs_i[un_idx]
        j_un = pairs_j[un_idx]
        if show_progress:
            print(f"[bootstrap_npmi] block of {block}, unsettled={un_idx.size}")
        for _ in range(block):
            sample_idx = rng.integers(0, C, size=iter_size)
            M_b = M[sample_idx]
            npmi_block = _bootstrap_npmi_for_pairs(
                M_b, i_un, j_un, alpha=alpha, metric=metric,
            )
            for kk, gk in enumerate(un_idx):
                v = npmi_block[kk]
                if np.isfinite(v):
                    sample_lists[gk].append(float(v))
                    n_samples[gk] += 1
        n_done += block

        for gk in un_idx:
            if n_samples[gk] < min_samples_for_ci:
                continue
            arr = sample_lists[gk]
            lo, hi = np.quantile(arr, [ci_lo_q, ci_hi_q])
            median = float(np.median(arr))
            # IN-LOOP early-stop: only on CONFIDENT classifications.
            #   - pos_strong / neg_strong  : CI clearly outside ±tau_high
            #   - tight_null               : CI clearly inside ±tau_low
            # Weak (CI between ±tau_low and ±tau_high) and dead_zone (CI inside
            # ±tau_high but straddles ±tau_low) intentionally do NOT early-stop:
            # those pairs should keep iterating so they can either firm up to
            # strong or have a precise post-budget median, not get locked in
            # by a CI that just crossed the lower threshold. With scalar tau
            # (tau_low == tau_high), the tight_null branch IS the legacy
            # dead_zone, so behavior matches the original 3-kind early-stop.
            if lo > tau_high:
                unsettled[gk] = False
                settled_kind[gk] = 1   # pos_strong
            elif hi < -tau_high:
                unsettled[gk] = False
                settled_kind[gk] = -1  # neg_strong
            elif lo > -tau_low and hi < tau_low:
                unsettled[gk] = False
                settled_kind[gk] = 3   # tight_null (= legacy "dead_zone" when scalar)
            if not unsettled[gk] and persist_ci:
                per_pair_ci_lo[gk] = lo
                per_pair_ci_hi[gk] = hi
                per_pair_median[gk] = median
            if not unsettled[gk]:
                settled_at_n_done[gk] = n_done
                sample_lists[gk] = []  # release memory

    # Post-budget classification: pairs that didn't early-stop as
    # strong/strong-/tight_null get classified now based on their final CI.
    # In-loop early-stop excluded weak/dead_zone branches, so this pass
    # assigns those kinds (plus catches any pair that drifted into strong
    # territory at the very end).
    for gk in np.flatnonzero(unsettled):
        arr = sample_lists[gk]
        if len(arr) < min_samples_for_ci:
            continue   # leave kind=0 (unsettled), no CI info captured below
        lo, hi = np.quantile(arr, [ci_lo_q, ci_hi_q])
        median = float(np.median(arr))
        # Apply full 6-condition cascade to the final CI:
        if lo > tau_high:
            unsettled[gk] = False
            settled_kind[gk] = 1   # pos_strong (drifted up at end)
        elif lo > tau_low:
            unsettled[gk] = False
            settled_kind[gk] = 2   # pos_weak
        elif hi < -tau_high:
            unsettled[gk] = False
            settled_kind[gk] = -1  # neg_strong (drifted down at end)
        elif hi < -tau_low:
            unsettled[gk] = False
            settled_kind[gk] = -2  # neg_weak
        elif lo > -tau_low and hi < tau_low:
            unsettled[gk] = False
            settled_kind[gk] = 3   # tight_null
        elif lo > -tau_high and hi < tau_high:
            unsettled[gk] = False
            settled_kind[gk] = 4   # dead_zone
        # else: stays kind=0 (genuinely unsettled — CI extends beyond ±tau_high
        #       and doesn't clear ±tau_low)
        if persist_ci:
            per_pair_ci_lo[gk] = lo
            per_pair_ci_hi[gk] = hi
            per_pair_median[gk] = median

    # Capture CI for pairs that have <min_samples and didn't go through cascade above.
    if persist_ci:
        for gk in np.flatnonzero(unsettled):
            arr = sample_lists[gk]
            if len(arr) >= 2:
                # Only happens if min_samples_for_ci wasn't met above.
                lo, hi = np.quantile(arr, [ci_lo_q, ci_hi_q])
                per_pair_ci_lo[gk] = lo
                per_pair_ci_hi[gk] = hi
                per_pair_median[gk] = float(np.median(arr))
            elif len(arr) == 1:
                per_pair_median[gk] = float(arr[0])

    n_pos_strong = int((settled_kind == 1).sum())
    n_pos_weak   = int((settled_kind == 2).sum())
    n_neg_strong = int((settled_kind == -1).sum())
    n_neg_weak   = int((settled_kind == -2).sum())
    n_tight_null = int((settled_kind == 3).sum())
    n_dead_zone  = int((settled_kind == 4).sum())
    n_unsettled  = int(unsettled.sum())
    # Legacy aggregate counts (scalar-tau back-compat: weak bins are empty,
    # tight_null collapses into "dead_zone" reporting).
    n_pos  = n_pos_strong + n_pos_weak
    n_neg  = n_neg_strong + n_neg_weak
    n_dead = n_tight_null + n_dead_zone

    # For settled pos / neg pairs (strong AND weak), store the canonical legacy
    # value in W (the bootstrap CI confirms direction; the value is legacy NPMI
    # or PMI depending on `metric`). tight_null and dead_zone do NOT enter W.
    settled_mask = (
        (settled_kind == 1) | (settled_kind == -1) |
        (settled_kind == 2) | (settled_kind == -2)
    )
    if settled_mask.any():
        for k_idx in np.flatnonzero(settled_mask):
            i = int(pairs_i[k_idx]); j = int(pairs_j[k_idx])
            v = legacy_for_W_boot[k_idx]
            if np.isfinite(v):
                out_rows.append(i); out_cols.append(j); out_vals.append(float(v))

    W_sparse = sp.coo_matrix(
        (out_vals, (out_rows, out_cols)),
        shape=(G, G),
        dtype=np.float32,
    ).tocsr()

    # Append per-pair CI rows for everything that went through the
    # active sampler (settled pos/neg/dead-zone, plus unsettled).
    if ci_records is not None:
        if _is_dual_tau:
            kind_for = {1: "pos_strong", 2: "pos_weak",
                          -1: "neg_strong", -2: "neg_weak",
                          3: "tight_null", 4: "dead_zone",
                          0: "unsettled"}
        else:
            # Scalar tau: kind=2/-2/4 are unreachable; kind=3 (CI inside ±tau)
            # is the legacy "dead_zone" so map it to that name for back-compat.
            kind_for = {1: "pos", 2: "pos",
                          -1: "neg", -2: "neg",
                          3: "dead_zone", 4: "dead_zone",
                          0: "unsettled"}
        for k_idx in range(n_pairs):
            kind = kind_for[int(settled_kind[k_idx])]
            i = int(pairs_i[k_idx])
            j = int(pairs_j[k_idx])
            obs_idx = int(boot_idx[k_idx])
            ci_records.append((
                i, j, str(genes[i]), str(genes[j]),
                kind,
                float(legacy_npmi_boot[k_idx]) if np.isfinite(legacy_npmi_boot[k_idx]) else np.nan,
                float(legacy_pmi_boot[k_idx]) if np.isfinite(legacy_pmi_boot[k_idx]) else np.nan,
                float(per_pair_median[k_idx]) if (per_pair_median is not None and not np.isnan(per_pair_median[k_idx])) else np.nan,
                float(per_pair_ci_lo[k_idx]) if (per_pair_ci_lo is not None and not np.isnan(per_pair_ci_lo[k_idx])) else np.nan,
                float(per_pair_ci_hi[k_idx]) if (per_pair_ci_hi is not None and not np.isnan(per_pair_ci_hi[k_idx])) else np.nan,
                int(n_samples[k_idx]),
                float(expected_full_obs[obs_idx]),
                float(expected_sub_obs[obs_idx]),
            ))

    # Build a histogram of when pairs settled (in iterations of n_done).
    # Bucket boundaries at every refine_block multiple.
    settled_at_hist = {}
    settled_mask_arr = (settled_kind != 0)
    if settled_mask_arr.any():
        vals = settled_at_n_done[settled_mask_arr]
        edges = list(range(0, int(n_done) + max(refine_block, 1), max(refine_block, 1)))
        hist, _ = np.histogram(vals, bins=edges)
        for k_idx, n_in_bucket in enumerate(hist):
            if n_in_bucket > 0:
                settled_at_hist[f"≤{edges[k_idx + 1]}"] = int(n_in_bucket)

    diagnostics = {
        "n_neg_one": n_neg_one,
        "n_indeterminate": n_indeterminate,
        "n_low_evidence": n_low_evidence,
        "n_legacy_only": n_legacy_only,
        "n_dead_zone": n_dead,            # aggregate (tight_null + dead_zone)
        "n_pos": n_pos,                    # aggregate (pos_strong + pos_weak)
        "n_neg": n_neg,                    # aggregate (neg_strong + neg_weak)
        "n_pos_strong": n_pos_strong,
        "n_pos_weak":   n_pos_weak,
        "n_neg_strong": n_neg_strong,
        "n_neg_weak":   n_neg_weak,
        "n_tight_null": n_tight_null,
        "n_dead_zone_only": n_dead_zone,   # CI inside ±tau_high but straddles ±tau_low
        "n_unsettled": n_unsettled,
        "n_bootstraps_per_pair": n_samples,
        "settled_at_n_done": settled_at_n_done,
        "settled_at_hist": settled_at_hist,
        "total_bootstraps_run": n_done,
        "subsample_size": iter_size,
        "min_expected_cooccur_for_evidence": thr,
        "metric": metric,
        "tau_low": tau_low,
        "tau_high": tau_high,
        "is_dual_tau": _is_dual_tau,
        "pre_filter": pre_filter_diag,
    }
    pair_ci_df = _ci_records_to_df(ci_records) if ci_records is not None else None
    return NpmiBootstrapResult(
        W_sparse=W_sparse, genes=genes,
        diagnostics=diagnostics, pair_ci=pair_ci_df,
    )


def _ci_records_to_df(records: list[tuple]) -> pd.DataFrame:
    """Build the pair_ci DataFrame from a flat record list.

    Columns:
      gene_i_idx, gene_j_idx, gene_i, gene_j     — pair identity
      kind                                       — classification (pos/neg/...)
      legacy_npmi, legacy_pmi                    — population point estimates
      median, ci_lo, ci_hi, n_bootstraps         — bootstrap CI on the active metric
      expected_full, expected_sub                — N·p_i·p_j and N_b·p_i·p_j
    """
    return pd.DataFrame.from_records(
        records,
        columns=[
            "gene_i_idx", "gene_j_idx",
            "gene_i", "gene_j",
            "kind",
            "legacy_npmi", "legacy_pmi",
            "median", "ci_lo", "ci_hi", "n_bootstraps",
            "expected_full", "expected_sub",
        ],
    )


#
def build_cell_gene_matrix(filtered_df, min_transcripts=10, genes_npm=None, cell_col="cell_id", exclude_ids=None):
    """
    Construct a binary (presence/absence) cell × gene matrix from a filtered
    transcript-level DataFrame and align it to the NPMI gene universe.

    This function takes a transcript df (already filtered for QV, removes 
    low-quality cells, builds a binary indicator matrix of gene presence within 
    each cell, and then compute purity/conflict scores.

    Parameters
    ----------
    filtered_df : pandas.DataFrame
        A transcript-level table containing at least:
        cell_col and "feature_name"

    min_transcripts : int, optional (default=10)
        Minimum number of transcripts required for a cell to be retained.

    genes_npm : pandas.DataFrame
        The long-format NPMI table containing columns "gene_i", "gene_j", "NPMI".
        
    cell_col : str, optional (default="cell_id")
        The column name containing cell identifiers.
        
    exclude_ids : set | None, optional (default=None)
        Set of cell IDs to exclude, e.g. {"-1", "DROP", "nan", "UNASSIGNED"}.
        If None, defaults to {"UNASSIGNED"}.

    Returns
    -------
    cell_ids : numpy.ndarray, shape (n_cells,)
        List of cell IDs (strings) corresponding to the rows of the matrix.

    genes_cell : numpy.ndarray, shape (n_genes_filtered,)
        Gene names (strings) corresponding to the columns of the filtered 
        presence/absence matrix. Only genes appearing in the NPMI dataset
        are retained.

    M : numpy.ndarray, dtype int8, shape (n_cells, n_genes_filtered)
        Binary presence/absence matrix:
            M[i, j] = 1 if cell i expresses gene j (≥1 transcript)
                      0 otherwise.

    col_idx : numpy.ndarray, dtype int32, shape (n_genes_filtered,)
        For each retained gene column, the corresponding index into the 
        global NPMI gene universe. Used to index into the full NPMI matrix
        when computing purity/conflict for each cell.

    Notes
    -----
    - Presence/absence is used instead of transcript counts because the NPMI
      scoring relies on pairwise co-occurrence patterns rather than expression
      magnitude.
    - Filtering to the NPMI gene universe ensures that the rows of `M` and the
      NPMI matrix use consistent gene indexing.
    """
    
    # Convert cell IDs to string for consistency with AnnData
    df = filtered_df
    # Avoid copying 100M-row df up front; use boolean views where possible.
    cell_col_series = df[cell_col].astype(str)

    # Remove excluded cell IDs
    if exclude_ids is None:
        exclude_ids = {"UNASSIGNED"}
    if exclude_ids:
        keep_mask = ~cell_col_series.isin(exclude_ids)
        cell_col_series = cell_col_series[keep_mask]
        df = df.loc[keep_mask.index[keep_mask]]

    # Filter by minimum transcript count per cell
    cell_counts = cell_col_series.groupby(cell_col_series).size()
    good_ids = cell_counts[cell_counts >= min_transcripts].index
    mask_good = cell_col_series.isin(good_ids)
    df = df.loc[mask_good.index[mask_good]]
    cell_col_series = cell_col_series[mask_good]

    # Restrict gene universe to NPMI vocabulary *before* building the matrix,
    # so sparse construction skips transcripts whose gene never shows up in
    # NPMI pairs at all.
    all_genes = np.union1d(
        genes_npm["gene_i"].unique(),
        genes_npm["gene_j"].unique()
    )

    gene_series = df["feature_name"].astype(str)
    in_vocab = gene_series.isin(all_genes)
    df = df.loc[in_vocab.index[in_vocab]]
    cell_col_series = cell_col_series[in_vocab]
    gene_series = gene_series[in_vocab]

    # Build presence/absence matrix via categorical codes + scipy.sparse.
    # Previous implementation used pivot_table(aggfunc=lambda x: 1), which
    # forces a Python call per group — catastrophic on 100M+ rows. Here we
    # let scipy coalesce duplicates at CSR-build time.
    cell_cat = pd.Categorical(cell_col_series)
    gene_cat = pd.Categorical(gene_series, categories=all_genes)

    rows_i = cell_cat.codes.astype(np.int32)
    cols_i = gene_cat.codes.astype(np.int32)
    # Any gene not in `all_genes` got code -1; defensive filter.
    valid = cols_i >= 0
    if not valid.all():
        rows_i = rows_i[valid]
        cols_i = cols_i[valid]

    n_cells = len(cell_cat.categories)
    n_genes = len(all_genes)

    # COO → CSR de-duplicates automatically (sum_duplicates → binarise).
    coo = sp.coo_matrix(
        (np.ones(len(rows_i), dtype=np.int8), (rows_i, cols_i)),
        shape=(n_cells, n_genes),
    )
    csr = coo.tocsr()
    csr.data = np.ones_like(csr.data, dtype=np.int8)  # binarise

    cell_ids = cell_cat.categories.to_numpy().astype(str)

    # Drop columns (genes) that never appeared in any retained cell — keeps
    # M's width the same as before: only genes actually present.
    col_mass = np.asarray(csr.sum(axis=0)).ravel() > 0
    csr = csr[:, col_mass]
    genes_cell = all_genes[col_mass]
    col_idx = np.flatnonzero(col_mass).astype(np.int32)

    # Densify to int8 for backward-compat (callers expect np.ndarray). At
    # ~200K cells × ~500 genes this is ~100 MiB — negligible next to the
    # 100M-row source df and orders of magnitude smaller than what the
    # pivot_table was allocating.
    M = np.asarray(csr.todense(), dtype=np.int8)

    return cell_ids, genes_cell, M, col_idx

#
def build_npmi_matrix(nucleus_npmi_long):
    """
    Construct a dense NPMI (Normalized Pointwise Mutual Information) matrix
    from a long-format NPMI dataframe.

    Parameters
    ----------
    nucleus_npmi_long : pandas.DataFrame
        Long-format NPMI table where each row represents a gene–gene pair.
        The dataframe must contain at least the following columns:
            - "gene_i" : str
                The first gene in the pair.
            - "gene_j" : str
                The second gene in the pair.
            - "NPMI" : float
                The normalized PMI score between gene_i and gene_j.
                
    Returns
    -------
    npmi_mat : np.ndarray, shape (G, G)
        A dense symmetric matrix where entry (i, j) contains the NPMI value
        between gene_i and gene_j. 
        Missing gene pairs implicitly receive a value of 0.

    gene_to_idx : dict
        A dictionary mapping each gene name to its corresponding row/column
        index in `npmi_mat`. This mapping is required to align the NPMI
        matrix with the columns of the cell × gene presence/absence matrix
        before computing cell purity and conflict scores.

    Notes
    -----
    - The function ensures symmetry of the NPMI matrix by populating both
      (i, j) and (j, i).
    """

    genes = np.union1d(
        nucleus_npmi_long["gene_i"].unique(),
        nucleus_npmi_long["gene_j"].unique(),
    )
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    G = len(genes)

    # Vectorized: no more Python per-row itertuples loop. At G=500 this
    # went from ~2 s to ~10 ms; at G=5000 the old loop would take minutes.
    i_idx = nucleus_npmi_long["gene_i"].map(gene_to_idx).to_numpy()
    j_idx = nucleus_npmi_long["gene_j"].map(gene_to_idx).to_numpy()
    vals = nucleus_npmi_long["NPMI"].to_numpy(dtype=float)

    npmi_mat = np.zeros((G, G), dtype=float)
    npmi_mat[i_idx, j_idx] = vals
    npmi_mat[j_idx, i_idx] = vals

    return npmi_mat, gene_to_idx

#
def attach_metrics_to_adata(adata, purity_df, conflict_df):
    """
    Attach NPMI-derived cell quality metrics (purity and conflict) to an AnnData object.

    This function takes an AnnData object and two DataFrames containing per-cell
    purity and conflict metrics derived from NPMI analysis. It then maps these 
    scores onto `adata.obs` using each cell's unique cell ID from `adata.obs_names`. 
    Four new columns are added to the AnnData object:

        - `cell_purity`        : continuous purity score (float)
        - `cell_purity_bool`   : boolean flag indicating whether the cell meets 
                                 the "pure" criterion based on purity threshold
        - `conflict_score`     : continuous conflict score (float)
        - `is_conflict`        : boolean flag indicating whether the cell meets 
                                 the "high-conflict" criterion based on conflict threshold

    Parameters
    ----------
    adata : AnnData
        The AnnData object whose `.obs` dataframe will be updated. Cell IDs 
        are taken from `adata.obs_names`.

    purity_df : pandas.DataFrame
        DataFrame with columns: cell_id, cell_purity, is_pure

    conflict_df : pandas.DataFrame
        DataFrame with columns: cell_id, conflict_score, is_conflict

    Returns
    -------
    None
        The function modifies `adata` in place by adding the new columns to 
        `adata.obs`. Nothing is explicitly returned.
    """
    # Create mapping dictionaries
    purity_map = dict(zip(purity_df["cell_id"], purity_df["cell_purity"]))
    purity_bool_map = dict(zip(purity_df["cell_id"], purity_df["is_pure"]))
    conflict_map = dict(zip(conflict_df["cell_id"], conflict_df["conflict_score"]))
    conflict_bool_map = dict(zip(conflict_df["cell_id"], conflict_df["is_conflict"]))
    
    # Map using obs_names (cell IDs as index)
    adata.obs["cell_purity"] = adata.obs_names.map(purity_map)
    adata.obs["cell_purity_bool"] = adata.obs_names.map(purity_bool_map)
    adata.obs["conflict_score"] = adata.obs_names.map(conflict_map)
    adata.obs["is_conflict"] = adata.obs_names.map(conflict_bool_map)

#
def compute_cell_purity(
    M,
    col_idx,
    npmi_mat,
    npmi_threshold=0.05,        # NPMI > this = "positive" co-occurrence
    cell_ids=None,
    purity_percentile=80.0,     # top X% are considered "pure"
    purity_threshold=None       # OR set an explicit numeric threshold (overrides percentile)
):
    """
    Compute cell purity score for each cell based on NPMI matrix.

    Purity = fraction of gene-gene NPMI values greater than npmi_threshold.

    Also returns a boolean "is_pure" mask using either:
      - purity_threshold (if given), or
      - the purity_percentile (default 80% → bottom 20% are suspect).
    """

    # Single parallel kernel pass → all primitives we need for every
    # per-row metric. Replaces 200K × O(k^2) Python loop.
    k_arr, n_pos, _sum_neg, _pos_relu, _neg_relu = pair_aggregate_dense(
        M, col_idx, npmi_mat, threshold=npmi_threshold, tau=0.0,
    )
    n_pairs_total = k_arr * (k_arr - 1) // 2

    purity_scores = np.full(M.shape[0], np.nan, dtype=float)
    has_pairs = n_pairs_total > 0
    purity_scores[has_pairs] = n_pos[has_pairs] / n_pairs_total[has_pairs]

    # determine threshold for boolean purity
    valid = ~np.isnan(purity_scores)
    if purity_threshold is None:
        purity_threshold = np.nanpercentile(purity_scores[valid], purity_percentile)

    is_pure = np.zeros_like(purity_scores, dtype=bool)
    is_pure[valid] = purity_scores[valid] >= purity_threshold

    purity_df = None
    if cell_ids is not None:
        purity_df = pd.DataFrame({
            "cell_id": cell_ids,
            "cell_purity": purity_scores,
            "is_pure": is_pure
        })

    return purity_scores, is_pure, purity_threshold, purity_df

#
def compute_cell_conflict(
    M,
    col_idx,
    npmi_mat,
    cell_ids=None,
    conflict_percentile=80.0,   # top X% most conflicting
    conflict_threshold=None     # optional explicit threshold for conflict_score
):
    """
    Conflict score = normalized weighted magnitude of negative NPMI pairs.
    Higher = more contaminated / merged.
    """

    # Kernel returns `sum_neg` per row; conflict = sum_neg / total_pairs.
    # `threshold` arg below doesn't affect sum_neg, only n_pos_above.
    k_arr, _n_pos, sum_neg, _pos_relu, _neg_relu = pair_aggregate_dense(
        M, col_idx, npmi_mat, threshold=0.0, tau=0.0,
    )
    n_pairs_total = k_arr * (k_arr - 1) // 2

    conflict_scores = np.full(M.shape[0], np.nan, dtype=float)
    has_pairs = n_pairs_total > 0
    conflict_scores[has_pairs] = sum_neg[has_pairs] / n_pairs_total[has_pairs]

    valid = ~np.isnan(conflict_scores)
    if conflict_threshold is None:
        conflict_threshold = np.nanpercentile(
            conflict_scores[valid],
            conflict_percentile
        )

    is_conflict = np.zeros_like(conflict_scores, dtype=bool)
    is_conflict[valid] = conflict_scores[valid] >= conflict_threshold

    # optional DF
    if cell_ids is not None:
        conflict_df = pd.DataFrame({
            "cell_id": cell_ids,
            "conflict_score": conflict_scores,
            "is_conflict": is_conflict
        })
    else:
        conflict_df = pd.DataFrame({
            "conflict_score": conflict_scores,
            "is_conflict": is_conflict
        })

    return conflict_scores, is_conflict, conflict_threshold, conflict_df
#
def compute_purity_and_conflict(
    filtered_df,
    nucleus_npmi_long,
    adata,
    *,
    cell_col="cell_id",
    min_transcripts_per_cell=10,
    exclude_ids=None,
    npmi_threshold=0.05,
    purity_percentile=80.0,
    conflict_percentile=80.0,
):
    """
    Starting from filtered_df (already QV- and gene-filtered),
    compute:
      - cell purity score
      - cell conflict score
    and attach them to adata.obs

    Parameters
    ----------
    filtered_df : DataFrame
        Transcript-level data
    nucleus_npmi_long : DataFrame
        Pre-computed NPMI matrix in long format
    adata : AnnData
        AnnData object to attach metrics to
    cell_col : str
        Column name containing cell IDs in filtered_df
    min_transcripts_per_cell : int
        Minimum transcripts required per cell
    npmi_threshold : float
        NPMI threshold for purity calculation
    purity_percentile : float
        Percentile for purity threshold
    conflict_percentile : float
        Percentile for conflict threshold
    exclude_ids : set | None
        Set of cell IDs to exclude, e.g. {"-1", "DROP", "nan", "UNASSIGNED"}

    Returns:
        purity_df, conflict_df
    """
    # -------- Build cell × gene matrix --------
    cell_ids, genes_cell, M, col_idx = build_cell_gene_matrix(
        filtered_df,
        min_transcripts=min_transcripts_per_cell,
        genes_npm=nucleus_npmi_long,
        cell_col=cell_col,
        exclude_ids=exclude_ids,
    )

    # -------- Build NPMI matrix --------
    npmi_mat, gene_to_idx_all = build_npmi_matrix(nucleus_npmi_long)

    # -------- Purity --------
    purity_scores, is_pure, purity_thr, purity_df = compute_cell_purity(
        M=M,
        col_idx=col_idx,
        npmi_mat=npmi_mat,
        npmi_threshold=npmi_threshold,
        cell_ids=cell_ids,
        purity_percentile=purity_percentile,
    )

    print("Purity threshold used:", purity_thr)

    # -------- Conflict --------
    conflict_scores, is_conflict, conflict_thr, conflict_df = compute_cell_conflict(
        M=M,
        col_idx=col_idx,
        npmi_mat=npmi_mat,
        cell_ids=cell_ids,
        conflict_percentile=conflict_percentile,
    )

    print("Conflict threshold used:", conflict_thr)

    # -------- Attach results to adata.obs --------
    attach_metrics_to_adata(adata, purity_df, conflict_df)

    return purity_df, conflict_df

#
from ._utils import relu_symmetric  # noqa: E402 — re-exported for back-compat

#
def compute_cell_purity_relu(
    M,
    col_idx,
    npmi_mat,
    tau=0.05,                  # dead-zone threshold
    cell_ids=None,
    purity_percentile=80.0,
    purity_threshold=None,
    eps=1e-8                   # minimum signal for normalization
):
    """
    ReLU-based cell purity score with relative metrics.

    Uses a symmetric ReLU on NPMI to:
      - zero out weak associations within [-tau, tau]
      - weight stronger positive/negative evidence more
      
    Computes:
      - Absolute purity: sum of positive ReLU values normalized by number of pairs
      - Relative purity: fraction of total signal that is positive
      - Relative conflict: fraction of total signal that is negative
      - Signal strength: total magnitude of non-zero ReLU values

    Parameters
    ----------
    M : np.ndarray, shape (n_cells, n_genes)
        Binary presence/absence matrix
    col_idx : np.ndarray
        Gene indices mapping to NPMI matrix columns
    npmi_mat : np.ndarray
        Full NPMI matrix
    tau : float
        Dead-zone threshold for symmetric ReLU
    cell_ids : array-like, optional
        Cell identifiers for output DataFrame
    purity_percentile : float
        Percentile for purity threshold (if threshold not provided)
    purity_threshold : float, optional
        Explicit threshold for binary purity classification
    eps : float
        Minimum signal strength for computing relative metrics

    Returns
    -------
    purity_scores : np.ndarray
        Absolute purity scores per cell
    is_pure : np.ndarray
        Boolean array indicating pure cells
    purity_threshold : float
        Threshold used for classification
    purity_df : pd.DataFrame or None
        DataFrame with all purity metrics if cell_ids provided
    """
    k_arr, _n_pos, _sum_neg, pos_relu, neg_relu = pair_aggregate_dense(
        M, col_idx, npmi_mat, threshold=0.0, tau=tau,
    )
    n_pairs_total = k_arr * (k_arr - 1) // 2
    has_pairs = n_pairs_total > 0
    total_abs = pos_relu + neg_relu

    n_cells = M.shape[0]
    purity_scores = np.full(n_cells, np.nan, dtype=float)
    signal_strength = np.full(n_cells, np.nan, dtype=float)
    relative_purity = np.full(n_cells, np.nan, dtype=float)
    relative_conflict = np.full(n_cells, np.nan, dtype=float)

    purity_scores[has_pairs] = pos_relu[has_pairs] / n_pairs_total[has_pairs]
    signal_strength[has_pairs] = total_abs[has_pairs]

    has_signal = has_pairs & (total_abs > eps)
    relative_purity[has_signal] = pos_relu[has_signal] / total_abs[has_signal]
    relative_conflict[has_signal] = neg_relu[has_signal] / total_abs[has_signal]

    valid = ~np.isnan(purity_scores)

    if purity_threshold is None:
        purity_threshold = np.nanpercentile(
            purity_scores[valid], purity_percentile
        )

    is_pure = np.zeros_like(purity_scores, dtype=bool)
    is_pure[valid] = purity_scores[valid] >= purity_threshold

    purity_df = None
    if cell_ids is not None:
        purity_df = pd.DataFrame({
            "cell_id": cell_ids,
            "cell_purity_relu": purity_scores,
            "signal_strength": signal_strength,
            "relative_purity": relative_purity,
            "relative_conflict": relative_conflict,
            "is_pure": is_pure
        })

    return purity_scores, is_pure, purity_threshold, purity_df

#
def compute_cell_conflict_relu(
    M,
    col_idx,
    npmi_mat,
    tau=0.05,
    cell_ids=None,
    conflict_percentile=80.0,
    conflict_threshold=None,
    eps=1e-8
):
    """
    ReLU-based conflict score with relative metrics.

    Measures magnitude-weighted negative evidence
    after suppressing weak NPMI values within [-tau, tau].
    
    Computes:
      - Absolute conflict: sum of negative ReLU values normalized by number of pairs
      - Relative conflict: fraction of total signal that is negative
      - Relative purity: fraction of total signal that is positive
      - Signal strength: total magnitude of non-zero ReLU values

    Parameters
    ----------
    M : np.ndarray, shape (n_cells, n_genes)
        Binary presence/absence matrix
    col_idx : np.ndarray
        Gene indices mapping to NPMI matrix columns
    npmi_mat : np.ndarray
        Full NPMI matrix
    tau : float
        Dead-zone threshold for symmetric ReLU
    cell_ids : array-like, optional
        Cell identifiers for output DataFrame
    conflict_percentile : float
        Percentile for conflict threshold (if threshold not provided)
    conflict_threshold : float, optional
        Explicit threshold for binary conflict classification
    eps : float
        Minimum signal strength for computing relative metrics

    Returns
    -------
    conflict_scores : np.ndarray
        Absolute conflict scores per cell
    is_conflict : np.ndarray
        Boolean array indicating high-conflict cells
    conflict_threshold : float
        Threshold used for classification
    conflict_df : pd.DataFrame or None
        DataFrame with all conflict metrics if cell_ids provided
    """
    k_arr, _n_pos, _sum_neg, pos_relu, neg_relu = pair_aggregate_dense(
        M, col_idx, npmi_mat, threshold=0.0, tau=tau,
    )
    n_pairs_total = k_arr * (k_arr - 1) // 2
    has_pairs = n_pairs_total > 0
    total_abs = pos_relu + neg_relu

    n_cells = M.shape[0]
    conflict_scores = np.full(n_cells, np.nan, dtype=float)
    signal_strength = np.full(n_cells, np.nan, dtype=float)
    relative_purity = np.full(n_cells, np.nan, dtype=float)
    relative_conflict = np.full(n_cells, np.nan, dtype=float)

    conflict_scores[has_pairs] = neg_relu[has_pairs] / n_pairs_total[has_pairs]
    signal_strength[has_pairs] = total_abs[has_pairs]

    has_signal = has_pairs & (total_abs > eps)
    relative_purity[has_signal] = pos_relu[has_signal] / total_abs[has_signal]
    relative_conflict[has_signal] = neg_relu[has_signal] / total_abs[has_signal]

    valid = ~np.isnan(conflict_scores)

    if conflict_threshold is None:
        conflict_threshold = np.nanpercentile(
            conflict_scores[valid], conflict_percentile
        )

    is_conflict = np.zeros_like(conflict_scores, dtype=bool)
    is_conflict[valid] = conflict_scores[valid] >= conflict_threshold

    if cell_ids is not None:
        conflict_df = pd.DataFrame({
            "cell_id": cell_ids,
            "cell_conflict_relu": conflict_scores,
            "signal_strength": signal_strength,
            "relative_purity": relative_purity,
            "relative_conflict": relative_conflict,
            "is_conflict": is_conflict
        })
    else:
        conflict_df = pd.DataFrame({
            "cell_conflict_relu": conflict_scores,
            "signal_strength": signal_strength,
            "relative_purity": relative_purity,
            "relative_conflict": relative_conflict,
            "is_conflict": is_conflict
        })

    return conflict_scores, is_conflict, conflict_threshold, conflict_df

#
def attach_metrics_to_adata_relu(adata, purity_df, conflict_df):
    """
    Attach ReLU-based NPMI metrics to AnnData object.
    
    This function adds the following columns to adata.obs:
        - cell_purity_relu: absolute purity score
        - relative_purity: fraction of signal that is positive
        - relative_conflict: fraction of signal that is negative  
        - signal_strength: total magnitude of non-zero ReLU values
        - is_pure: boolean flag for pure cells
        - cell_conflict_relu: absolute conflict score
        - is_conflict: boolean flag for high-conflict cells
        
    Parameters
    ----------
    adata : AnnData
        The AnnData object to update
    purity_df : pd.DataFrame
        DataFrame with purity metrics from compute_cell_purity_relu
    conflict_df : pd.DataFrame
        DataFrame with conflict metrics from compute_cell_conflict_relu
        
    Returns
    -------
    None
        Modifies adata.obs in place
    """
    # Map purity metrics
    purity_map = dict(zip(purity_df["cell_id"], purity_df["cell_purity_relu"]))
    rel_purity_map = dict(zip(purity_df["cell_id"], purity_df["relative_purity"]))
    signal_map_p = dict(zip(purity_df["cell_id"], purity_df["signal_strength"]))
    purity_bool_map = dict(zip(purity_df["cell_id"], purity_df["is_pure"]))
    
    # Map conflict metrics
    conflict_map = dict(zip(conflict_df["cell_id"], conflict_df["cell_conflict_relu"]))
    rel_conflict_map = dict(zip(conflict_df["cell_id"], conflict_df["relative_conflict"]))
    conflict_bool_map = dict(zip(conflict_df["cell_id"], conflict_df["is_conflict"]))
    
    # Attach to adata.obs
    adata.obs["cell_purity_relu"] = adata.obs_names.map(purity_map)
    adata.obs["relative_purity"] = adata.obs_names.map(rel_purity_map)
    adata.obs["relative_conflict"] = adata.obs_names.map(rel_conflict_map)
    adata.obs["signal_strength"] = adata.obs_names.map(signal_map_p)
    adata.obs["is_pure_relu"] = adata.obs_names.map(purity_bool_map)
    adata.obs["cell_conflict_relu"] = adata.obs_names.map(conflict_map)
    adata.obs["is_conflict_relu"] = adata.obs_names.map(conflict_bool_map)

#
def compute_purity_and_conflict_relu(
    filtered_df,
    nucleus_npmi_long,
    adata,
    *,
    cell_col="cell_id",
    min_transcripts_per_cell=10,
    exclude_ids=None,
    tau=0.05,
    purity_percentile=80.0,
    conflict_percentile=80.0,
    eps=1e-8
):
    """
    Compute ReLU-based cell purity and conflict scores and attach to adata.
    
    This function uses a symmetric ReLU transformation to:
      - Suppress weak NPMI associations (within [-tau, tau])
      - Weight stronger positive and negative evidence more heavily
      - Compute both absolute and relative metrics
    
    The following metrics are computed and attached to adata.obs:
      - cell_purity_relu: absolute purity (positive evidence / total pairs)
      - cell_conflict_relu: absolute conflict (negative evidence / total pairs)
      - relative_purity: positive signal / total signal
      - relative_conflict: negative signal / total signal
      - signal_strength: total magnitude of non-zero ReLU values
      - is_pure_relu: boolean flag for pure cells
      - is_conflict_relu: boolean flag for high-conflict cells

    Parameters
    ----------
    filtered_df : pd.DataFrame
        Transcript-level data (already QV- and gene-filtered)
    nucleus_npmi_long : pd.DataFrame
        Pre-computed NPMI matrix in long format
    adata : AnnData
        AnnData object to attach metrics to
    cell_col : str
        Column name containing cell IDs in filtered_df
    min_transcripts_per_cell : int
        Minimum transcripts required per cell
    exclude_ids : set | None
        Set of cell IDs to exclude (e.g., {"UNASSIGNED", "DROP"})
    tau : float
        Dead-zone threshold for symmetric ReLU
    purity_percentile : float
        Percentile for purity threshold
    conflict_percentile : float
        Percentile for conflict threshold
    eps : float
        Minimum signal strength for computing relative metrics

    Returns
    -------
    purity_df : pd.DataFrame
        DataFrame with purity metrics per cell
    conflict_df : pd.DataFrame
        DataFrame with conflict metrics per cell
    """
    # -------- Build cell × gene matrix --------
    cell_ids, genes_cell, M, col_idx = build_cell_gene_matrix(
        filtered_df,
        min_transcripts=min_transcripts_per_cell,
        genes_npm=nucleus_npmi_long,
        cell_col=cell_col,
        exclude_ids=exclude_ids,
    )

    # -------- Build NPMI matrix --------
    npmi_mat, gene_to_idx_all = build_npmi_matrix(nucleus_npmi_long)

    # -------- ReLU-based Purity --------
    purity_scores, is_pure, purity_thr, purity_df = compute_cell_purity_relu(
        M=M,
        col_idx=col_idx,
        npmi_mat=npmi_mat,
        tau=tau,
        cell_ids=cell_ids,
        purity_percentile=purity_percentile,
        eps=eps,
    )

    print("ReLU Purity threshold used:", purity_thr)

    # -------- ReLU-based Conflict --------
    conflict_scores, is_conflict, conflict_thr, conflict_df = compute_cell_conflict_relu(
        M=M,
        col_idx=col_idx,
        npmi_mat=npmi_mat,
        tau=tau,
        cell_ids=cell_ids,
        conflict_percentile=conflict_percentile,
        eps=eps,
    )

    print("ReLU Conflict threshold used:", conflict_thr)

    # -------- Attach results to adata.obs --------
    attach_metrics_to_adata_relu(adata, purity_df, conflict_df)

    return purity_df, conflict_df
