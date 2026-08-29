#!/usr/bin/env python3
"""Audited canonical marker-panel selection for the ROI benchmark.

The panel builder starts from lineage-interpretable candidates, checks platform
presence, quantifies scRNA specificity, and selects 2-5 markers per lineage
(target 3) while avoiding broad stress/interferon genes when alternatives exist.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

import get_metric as gm


TARGET_N = 3
MIN_N = 2
MAX_N = 5
MIN_SPECIFICITY = 0.10

BROAD_OR_STRESS = {
    "ACTB", "ACTG1", "B2M", "FOS", "FOSB", "JUN", "JUNB", "JUND", "EGR1",
    "HSPA1A", "HSPA1B", "HSP90AA1", "HSP90AB1", "MALAT1", "MT-CO1", "MT-CO2",
    "MT-CO3", "MT-CYB", "MT-ND1", "MT-ND2", "MT-ND3", "MT-ND4", "MT-ND5",
    "MT-ND6", "MT-ATP6", "MT-ATP8", "IFI6", "IFI27", "IFI44", "IFI44L",
    "IFIT1", "IFIT2", "IFIT3", "ISG15", "MX1", "OAS1", "OAS2", "OAS3",
    "STAT1",
}


MARKER_CANDIDATES = {
    "cervical": {
        "T_NK": ["CD3D", "CD3E", "TRAC", "NKG7", "GNLY", "KLRD1", "GZMB", "PRF1"],
        "Tumor Epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "KRT17", "KRT5", "MUC1"],
        "Neutrophil": ["FCGR3B", "CSF3R", "CXCR2", "MMP8", "CEACAM8", "S100A8", "S100A9"],
        "Myeloid": ["C1QA", "C1QB", "LST1", "MS4A7", "CD68", "FCGR3A", "LYZ"],
        "Plasma": ["MZB1", "JCHAIN", "XBP1", "SDC1", "IGHG1", "IGKC"],
        "Fibroblast": ["COL1A1", "COL1A2", "DCN", "LUM", "COL3A1", "PDGFRA"],
        "Endothelial": ["PECAM1", "VWF", "KDR", "CLDN5", "ENG", "ESAM"],
        "B_cell": ["MS4A1", "CD79A", "CD79B", "BANK1", "CD74", "PAX5"],
        "Smooth_muscle": ["ACTA2", "MYH11", "TAGLN", "MYL9", "RGS5", "MCAM"],
        "Mast": ["TPSAB1", "TPSB2", "CPA3", "KIT", "MS4A2", "HPGDS"],
    },
    "cosmx_nsclc": {
        "T": ["CD3D", "CD3E", "TRAC", "NKG7", "CD8A", "IL7R", "GZMK", "GZMB"],
        "Myeloid": ["C1QA", "C1QB", "LST1", "MS4A7", "CD68", "FCGR3A", "LYZ"],
        "Cancer": ["EPCAM", "KRT8", "KRT18", "KRT19", "MUC1", "KRT7", "CEACAM6"],
        "B": ["MS4A1", "CD79A", "CD79B", "BANK1", "CD74", "PAX5"],
        "Fibroblasts": ["COL1A1", "COL1A2", "DCN", "LUM", "COL3A1", "PDGFRA"],
        "Plasma": ["MZB1", "JCHAIN", "XBP1", "SDC1", "IGHG1", "IGKC"],
        "Mast": ["TPSAB1", "TPSB2", "CPA3", "KIT", "MS4A2", "HPGDS"],
        "Endothelial": ["PECAM1", "VWF", "KDR", "CLDN5", "ENG", "ESAM"],
        "Ciliated": ["FOXJ1", "PIFO", "TPPP3", "CAPS", "DNAH5", "DNAH11"],
    },
    "merfish_mouse_ileum": {
        "Stem_TA": ["Lgr5", "Olfm4", "Ascl2", "Mki67", "Top2a", "Axin2"],
        "Enterocyte": ["Alpi", "Apoa1", "Apoa4", "Fabp1", "Slc26a3", "Krt20"],
        "Goblet": ["Muc2", "Tff3", "Clca1", "Fcgbp", "Agr2", "Spdef"],
        "Paneth": ["Lyz1", "Defa24", "Mmp7", "Ang4", "Defa17", "Reg3g"],
        "Enteroendocrine": ["Chga", "Chgb", "Neurod1", "Tph1", "Pax6", "Isl1"],
        "Tuft": ["Dclk1", "Trpm5", "Pou2f3", "Alox5ap", "Gfi1b", "Avil"],
    },
}


def dataset_family(dataset: str) -> str:
    if dataset in {"atera_cervical", "xenium5k_cervical"}:
        return "cervical"
    if dataset == "cosmx_nsclc":
        return "cosmx_nsclc"
    if dataset == "merfish_mouse_ileum":
        return "merfish_mouse_ileum"
    raise KeyError(dataset)


def _drop_celltypes(ref: gm.ReferenceData, exclude: list[str] | tuple[str, ...]):
    if not exclude:
        return ref
    labels = ref.obs[ref.celltype_col].astype(str).to_numpy()
    keep = ~np.isin(labels, list(exclude))
    return gm.ReferenceData(
        counts_csr=ref.counts_csr[keep].tocsr(),
        var_names=ref.var_names,
        obs=ref.obs.loc[keep].copy(),
        celltype_col=ref.celltype_col,
    )


def restrict_reference(ref: gm.ReferenceData, genes: set[str]) -> gm.ReferenceData:
    keep = [i for i, g in enumerate(ref.var_names) if str(g) in genes]
    return gm.ReferenceData(
        counts_csr=ref.counts_csr[:, keep].tocsr(),
        var_names=np.asarray(ref.var_names, dtype=str)[keep],
        obs=ref.obs.copy(),
        celltype_col=ref.celltype_col,
    )


def _reference_specificity(ref_norm: sp.csr_matrix, ref: gm.ReferenceData,
                           var_pos: dict[str, int], cell_type: str, gene: str) -> float:
    j = var_pos.get(gene)
    if j is None:
        return np.nan
    labels = ref.obs[ref.celltype_col].astype(str).to_numpy()
    in_mask = labels == cell_type
    out_mask = labels != cell_type
    if in_mask.sum() == 0 or out_mask.sum() == 0:
        return np.nan
    vals = np.asarray(ref_norm[:, j].todense()).ravel()
    return float((vals[in_mask].mean() - vals[out_mask].mean()) / np.log(2.0))


def _top_marker_candidates(ref_u: gm.ReferenceData, n_top: int = 80) -> dict[str, list[str]]:
    logger = logging.getLogger("marker_panels")
    auto = gm.compute_reference_markers(ref_u, n_top=n_top, log=logger)
    out: dict[str, list[str]] = {}
    if auto.empty:
        return out
    for ct, g in auto.groupby("cell_type", sort=False):
        out[str(ct)] = list(g.sort_values("rank")["gene"].astype(str))
    return out


def build_marker_panel(
    dataset: str,
    ref: gm.ReferenceData,
    platform_genes: set[str],
    *,
    outdir: Path,
    exclude_celltypes: list[str] | tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (final_marker_table, audit_table) and save both under ``outdir``."""
    outdir.mkdir(parents=True, exist_ok=True)
    ref = _drop_celltypes(ref, exclude_celltypes)
    platform_genes = set(map(str, platform_genes))
    ref_genes = set(map(str, ref.var_names))
    ref_u = restrict_reference(ref, platform_genes & ref_genes)
    auto_by_ct = _top_marker_candidates(ref_u)
    candidates_by_ct = MARKER_CANDIDATES[dataset_family(dataset)]
    ref_norm = gm._log_normalize(ref.counts_csr).tocsr()
    ref_var_pos = {g: i for i, g in enumerate(ref.var_names)}

    audit_rows: list[dict] = []
    final_rows: list[dict] = []
    for cell_type, canonical in candidates_by_ct.items():
        seen: set[str] = set()
        candidates: list[tuple[str, str]] = []
        for gene in canonical:
            if gene not in seen:
                seen.add(gene)
                candidates.append((gene, "canonical"))
        for gene in auto_by_ct.get(cell_type, []):
            if gene not in seen:
                seen.add(gene)
                candidates.append((gene, "scrna_fallback"))
            if len(candidates) >= 30:
                break

        scored = []
        for order, (gene, source) in enumerate(candidates, start=1):
            upper = gene.upper()
            in_platform = gene in platform_genes
            in_ref = gene in ref_genes
            specificity = _reference_specificity(ref_norm, ref, ref_var_pos, cell_type, gene) if in_ref else np.nan
            broad = upper in BROAD_OR_STRESS
            eligible = bool(in_platform and in_ref and np.isfinite(specificity) and specificity >= MIN_SPECIFICITY and not broad)
            scored.append({
                "dataset": dataset,
                "candidate_marker": gene,
                "expected_lineage": cell_type,
                "candidate_source": source,
                "candidate_priority": order,
                "detection_in_platform_panel": "yes" if in_platform else "no",
                "specificity_in_scrna_reference": specificity,
                "broad_stress_interferon_flag": "yes" if broad else "no",
                "eligible": eligible,
            })

        eligible = [r for r in scored if r["eligible"]]
        canonical_eligible = [r for r in eligible if r["candidate_source"] == "canonical"]
        selected = canonical_eligible[:TARGET_N]
        if len(selected) < MIN_N:
            selected_genes = {r["candidate_marker"] for r in selected}
            for row in eligible:
                if row["candidate_marker"] not in selected_genes:
                    selected.append(row)
                    selected_genes.add(row["candidate_marker"])
                if len(selected) >= TARGET_N:
                    break
        selected = selected[:MAX_N]
        selected_genes = {r["candidate_marker"] for r in selected}

        for row in scored:
            reason = ""
            if row["candidate_marker"] in selected_genes:
                row["selected_for_final_panel"] = "yes"
                row["reason_for_exclusion_if_not_selected"] = ""
            else:
                if row["detection_in_platform_panel"] == "no":
                    reason = "absent from platform panel"
                elif not np.isfinite(row["specificity_in_scrna_reference"]):
                    reason = "not detected in scRNA reference"
                elif row["broad_stress_interferon_flag"] == "yes":
                    reason = "broad/stress/interferon marker avoided"
                elif row["specificity_in_scrna_reference"] < MIN_SPECIFICITY:
                    reason = f"low scRNA specificity (<{MIN_SPECIFICITY:g} log2FC)"
                elif row["candidate_source"] == "scrna_fallback" and len(canonical_eligible) >= TARGET_N:
                    reason = "canonical platform-present alternatives selected"
                else:
                    reason = "not in top selected markers for lineage"
                row["selected_for_final_panel"] = "no"
                row["reason_for_exclusion_if_not_selected"] = reason
            audit_rows.append(row)

        for rank, row in enumerate(selected, start=1):
            final_rows.append({
                "cell_type": cell_type,
                "gene": row["candidate_marker"],
                "rank": rank,
                "scrna_log2fc": float(row["specificity_in_scrna_reference"]),
                "selection_source": row["candidate_source"],
            })

    final = pd.DataFrame(final_rows)
    audit = pd.DataFrame(audit_rows).drop(columns=["eligible"])
    final.to_csv(outdir / "_marker_genes_3.tsv", sep="\t", index=False)
    final.to_csv(outdir / "final_marker_list.tsv", sep="\t", index=False)
    audit.to_csv(outdir / "marker_audit_table.tsv", sep="\t", index=False)
    return final, audit
