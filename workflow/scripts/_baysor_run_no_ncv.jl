#!/usr/bin/env julia
# Baysor `run` with the NCV "local colors" step neutralised.
#
# WHY: On the densest ROI (atera_cervical: 1.45M molecules x 17.4k genes), the
# post-segmentation NCV gene-composition color estimation
# (`Baysor.gene_composition_colors(df_spatial, k; ...)` in run_segmentation)
# builds a 1.45M x 17.4k neighborhood count matrix and OOM-deadlocks on a 17 GB
# machine. Those colors are used ONLY for plotting (the `:ncv_color` column);
# they do not affect the segmentation/assignments/counts. There is no CLI flag
# to skip NCV in Baysor 0.7, so we monkey-patch the 3-arg method to return cheap
# placeholder colors, then call the standard Baysor.command_main("run", ...).
#
# Everything else (molecule clustering, BMM cell segmentation, count matrix,
# segmentation.csv) is byte-for-byte the standard Baysor pipeline.
#
# Usage: julia _baysor_run_no_ncv.jl <coordinates.csv> -- <baysor run args...>
#   e.g. julia _baysor_run_no_ncv.jl input.csv -- -c config.toml -o out/ -m 20 -s 6.0 \
#          --polygon-format none --count-matrix-format tsv

using Baysor
using Baysor: Processing
import DataFrames

# --- Monkey-patch: neutralise NCV color estimation -------------------------
# Original (neighborhood_composition.jl:189):
#   gene_composition_colors(df_spatial::DataFrame, k::Int; method, ...) -> Vector{Colors.Lab}
# We override it to skip the expensive neighborhood_count_matrix + UMAP and
# return a constant grey for every molecule. run_segmentation only uses the
# result to fill segmented_df[:, :ncv_color] (a plotting-only column).
@eval Baysor.Processing function gene_composition_colors(
        df_spatial::$(DataFrames.DataFrame), k::Int; kwargs...)
    @info "[no-ncv patch] Skipping NCV local-color estimation (plotting-only); " *
          "returning placeholder colors for $(size(df_spatial, 1)) molecules."
    return fill(Colors.Lab(50.0, 0.0, 0.0), size(df_spatial, 1))
end

# --- Build ARGS for Baysor.command_main -------------------------------------
# Split our wrapper args (before `--`) from the baysor run args (after `--`).
sep = findfirst(==("--"), ARGS)
sep === nothing && error("expected: <coordinates> -- <baysor run args...>")
coordinates = ARGS[1]
run_args = ARGS[sep+1:end]

# Baysor's Comonicon entrypoint reads from the global ARGS vector.
empty!(ARGS)
push!(ARGS, "run", run_args..., coordinates)

@info "[no-ncv] invoking: baysor " * join(ARGS, " ")
exit(Baysor.command_main())
