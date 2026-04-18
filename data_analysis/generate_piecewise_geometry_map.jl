using Serialization
using Statistics
using Printf
using DataFrames
using HDF5
using JSON

const DEFAULT_MAPPING = "/home/rwu4/Geothermal_Graph_Surrogate/smallerModel.jld2"
const DEFAULT_MASK = "/home/rwu4/Geothermal_Graph_Surrogate/data_test/v2.5_0111.h5"
const DEFAULT_OUTPUT = "/home/rwu4/Geothermal_Graph_Surrogate/configs/geometry_map_piecewise_xy.generated.json"

function usage()
    println("""
Generate precomputed piecewise geometry map JSON for NPV optimization.

Usage:
  julia --project=/home/rwu4/GeologicalSimulationWrapper.jl \\
    /home/rwu4/Geothermal_Graph_Surrogate/data_analysis/generate_piecewise_geometry_map.jl \\
    [--mapping PATH] [--mask PATH] [--output PATH] [--n-knots N] [--min-profile-steps N]

Defaults:
  --mapping $(DEFAULT_MAPPING)
  --mask    $(DEFAULT_MASK)
  --output  $(DEFAULT_OUTPUT)
  --n-knots 8
  --min-profile-steps 8
""")
end

function parse_args(args::Vector{String})
    opts = Dict(
        "mapping" => DEFAULT_MAPPING,
        "mask" => DEFAULT_MASK,
        "output" => DEFAULT_OUTPUT,
        "n-knots" => "8",
        "min-profile-steps" => "8",
    )

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ["-h", "--help"]
            usage()
            exit(0)
        end
        if !startswith(arg, "--")
            error("Unexpected argument: $(arg)")
        end

        key = arg[3:end]
        if !haskey(opts, key)
            error("Unknown option --$(key)")
        end
        if i == length(args)
            error("Missing value for option --$(key)")
        end

        opts[key] = args[i + 1]
        i += 2
    end

    return (
        mapping=opts["mapping"],
        mask=opts["mask"],
        output=opts["output"],
        n_knots=parse(Int, opts["n-knots"]),
        min_profile_steps=parse(Int, opts["min-profile-steps"]),
    )
end

function load_mapping(path::String)
    obj = open(path, "r") do io
        deserialize(io)
    end

    if obj isa DataFrame
        return obj
    end

    if obj isa AbstractDict
        for (_, v) in obj
            if v isa DataFrame
                return v
            end
        end
    end

    if hasproperty(obj, :df) && getproperty(obj, :df) isa DataFrame
        return getproperty(obj, :df)
    end

    error("Could not locate DataFrame in deserialized object of type $(typeof(obj)).")
end

function load_active_grid(h5_path::String, ni::Int, nj::Int, nk::Int)
    is_active = h5open(h5_path, "r") do f
        read(f["Input/IsActive"])
    end

    dims = size(is_active)
    activeg = falses(ni, nj, nk)

    if dims == (nk, nj, ni)
        for kk in 1:nk, jj in 1:nj, ii in 1:ni
            activeg[ii, jj, kk] = is_active[kk, jj, ii] != 0
        end
    elseif dims == (ni, nj, nk)
        for kk in 1:nk, jj in 1:nj, ii in 1:ni
            activeg[ii, jj, kk] = is_active[ii, jj, kk] != 0
        end
    else
        error(
            "Unexpected IsActive shape $(dims). Expected (nk,nj,ni)=($(nk),$(nj),$(ni)) " *
            "or (ni,nj,nk)=($(ni),$(nj),$(nk))."
        )
    end

    return activeg
end

function fill_profile_nans!(profile::Vector{Float64})
    valid = findall(!isnan, profile)
    isempty(valid) && error("No valid dz profile entries found to build piecewise map")

    first_valid = first(valid)
    for k in 1:(first_valid - 1)
        profile[k] = profile[first_valid]
    end

    last_seen = first_valid
    for k in (first_valid + 1):length(profile)
        if isnan(profile[k])
            continue
        end
        if k - last_seen > 1
            a = profile[last_seen]
            b = profile[k]
            gap = k - last_seen
            for t in 1:(gap - 1)
                profile[last_seen + t] = a + (b - a) * (t / gap)
            end
        end
        last_seen = k
    end

    for k in (last_seen + 1):length(profile)
        profile[k] = profile[last_seen]
    end
end

function sample_knot_indices(nk::Int, n_knots::Int)
    n_knots < 2 && error("n-knots must be >= 2")
    idx = round.(Int, collect(range(1, nk; length=n_knots)))
    return unique(sort(idx))
end

function save_geometry_map(output_path::String, payload::Dict)
    mkpath(dirname(output_path))
    open(output_path, "w") do io
        JSON.print(io, payload, 2)
        write(io, '\n')
    end
end

function main()
    opts = parse_args(ARGS)

    @printf("Loading mapping from %s\n", opts.mapping)
    df = load_mapping(opts.mapping)

    colnames_str = String.(names(df))
    req = vcat(["COORD1", "COORD2", "COORD3"], ["CORNER$(i)" for i in 1:24])
    missing = [c for c in req if !(c in colnames_str)]
    !isempty(missing) && error("Missing required columns: $(missing)")

    i_idx = Int.(df[!, "COORD1"])
    j_idx = Int.(df[!, "COORD2"])
    k_idx = Int.(df[!, "COORD3"])

    ni = maximum(i_idx)
    nj = maximum(j_idx)
    nk = maximum(k_idx)

    @printf("Grid extents from mapping: ni=%d nj=%d nk=%d\n", ni, nj, nk)

    activeg = load_active_grid(opts.mask, ni, nj, nk)
    @printf("Active cells: %d / %d\n", count(activeg), ni * nj * nk)

    zcols = "CORNER" .* string.([3, 6, 9, 12, 15, 18, 21, 24])
    zmat = Matrix{Float64}(select(df, zcols))

    cz = vec(sum(zmat, dims=2) ./ 8.0)
    czg = fill(NaN, ni, nj, nk)
    for r in eachindex(i_idx)
        ii = i_idx[r]
        jj = j_idx[r]
        kk = k_idx[r]
        czg[ii, jj, kk] = cz[r]
    end

    dzg = fill(NaN, ni, nj, nk - 1)
    for ii in 1:ni, jj in 1:nj, kk in 1:(nk - 1)
        if !(activeg[ii, jj, kk] && activeg[ii, jj, kk + 1])
            continue
        end
        z1 = czg[ii, jj, kk]
        z2 = czg[ii, jj, kk + 1]
        if !(isnan(z1) || isnan(z2))
            dzg[ii, jj, kk] = abs(z2 - z1)
        end
    end

    mean_dz_per_k = fill(NaN, nk - 1)
    for kk in 1:(nk - 1)
        vals = Float64[]
        for ii in 1:ni, jj in 1:nj
            v = dzg[ii, jj, kk]
            if !isnan(v)
                push!(vals, v)
            end
        end
        if !isempty(vals)
            mean_dz_per_k[kk] = mean(vals)
        end
    end
    fill_profile_nans!(mean_dz_per_k)

    # Match proxy convention where z=0 still maps to a positive first-cell distance.
    step_len = Vector{Float64}(undef, nk)
    step_len[1:(nk - 1)] = mean_dz_per_k
    step_len[nk] = mean_dz_per_k[end]
    distance_at_z = cumsum(step_len)

    knot_indices = sample_knot_indices(nk, opts.n_knots)
    z_knots = Float64[(idx - 1) for idx in knot_indices]
    distance_knots = Float64[distance_at_z[idx] for idx in knot_indices]

    # Build lateral multiplicative correction gamma(i,j): least-squares fit to mean profile.
    gamma_ij = fill(NaN, ni, nj)
    valid_profile_count = 0
    for ii in 1:ni, jj in 1:nj
        numer = 0.0
        denom = 0.0
        used = 0
        for kk in 1:(nk - 1)
            v = dzg[ii, jj, kk]
            m = mean_dz_per_k[kk]
            if isnan(v) || isnan(m)
                continue
            end
            numer += v * m
            denom += m * m
            used += 1
        end
        if used >= opts.min_profile_steps && denom > eps(Float64)
            gamma_ij[ii, jj] = numer / denom
            valid_profile_count += 1
        end
    end

    finite_gamma = Float64[]
    for ii in 1:ni, jj in 1:nj
        g = gamma_ij[ii, jj]
        if !isnan(g) && isfinite(g) && g > 0.0
            push!(finite_gamma, g)
        end
    end
    isempty(finite_gamma) && error("No valid lateral scale entries computed")

    gamma_fallback = median(finite_gamma)
    for ii in 1:ni, jj in 1:nj
        g = gamma_ij[ii, jj]
        if isnan(g) || !isfinite(g) || g <= 0.0
            gamma_ij[ii, jj] = gamma_fallback
        end
    end

    # Normalize around 1.0 so xy_scale_default can stay 1.0.
    gamma_norm = median(vec(gamma_ij))
    gamma_ij ./= gamma_norm

    # Proxy axes: x corresponds to j-index, y corresponds to i-index.
    x_knots = Float64[x for x in 0:(nj - 1)]
    y_knots = Float64[y for y in 0:(ni - 1)]
    scale_grid = [
        [Float64(gamma_ij[y + 1, x + 1]) for x in 0:(nj - 1)]
        for y in 0:(ni - 1)
    ]

    payload = Dict(
        "description" => "Precomputed geometry distance model generated from mapping JLD + IsActive mask",
        "source" => Dict(
            "mapping_jld" => opts.mapping,
            "mask_h5" => opts.mask,
        ),
        "generation" => Dict(
            "n_knots" => opts.n_knots,
            "min_profile_steps" => opts.min_profile_steps,
            "valid_xy_profiles" => valid_profile_count,
            "total_xy_locations" => ni * nj,
            "grid_ijk" => [ni, nj, nk],
        ),
        "depth_distance_profile" => Dict(
            "z_knots" => z_knots,
            "distance_knots" => distance_knots,
        ),
        "xy_scale_default" => 1.0,
        "xy_scale_map" => Dict(
            "x_knots" => x_knots,
            "y_knots" => y_knots,
            "scale_grid" => scale_grid,
        ),
    )

    save_geometry_map(opts.output, payload)

    @printf("Saved geometry map to %s\n", opts.output)
    @printf("z_knots: %s\n", string(z_knots))
    @printf("distance_knots: %s\n", string(distance_knots))
    @printf("xy_scale_map size: (%d, %d) [rows=y, cols=x]\n", ni, nj)
end

main()
