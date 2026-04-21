using Serialization
using Statistics
using Printf
using DataFrames
using HDF5

const DEFAULT_PATH = "/home/rwu4/Geothermal_Graph_Surrogate/smallerModel.jld2"
const DEFAULT_H5_MASK_PATH = "/home/rwu4/Geothermal_Graph_Surrogate/data_test/v2.5_0111.h5"

function pctl(v::Vector{Float64}, p::Float64)
    @assert !isempty(v)
    sv = sort(v)
    idx = clamp(Int(ceil(p * length(sv))), 1, length(sv))
    return sv[idx]
end

function stats(v::Vector{Float64})
    mu = mean(v)
    sd = std(v)
    mn = minimum(v)
    mx = maximum(v)
    cv = iszero(mu) ? NaN : sd / abs(mu)
    return (n=length(v), mean=mu, std=sd, min=mn, max=mx, cv=cv)
end

function print_stats(label::String, s)
    @printf("%s: n=%d mean=%.6f std=%.6f min=%.6f max=%.6f cv=%.6f\n",
        label, s.n, s.mean, s.std, s.min, s.max, s.cv)
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
    @printf("Loading active mask from %s\n", h5_path)
    is_active = h5open(h5_path, "r") do f
        read(f["Input/IsActive"])
    end

    dims = size(is_active)
    @printf("IsActive dims in H5: (%d, %d, %d)\n", dims[1], dims[2], dims[3])

    activeg = falses(ni, nj, nk)

    if dims == (nk, nj, ni)
        # v2.5 tensors are [k, j, i], convert to [i, j, k] for mapping indices.
        for kk in 1:nk, jj in 1:nj, ii in 1:ni
            activeg[ii, jj, kk] = is_active[kk, jj, ii] != 0
        end
    elseif dims == (ni, nj, nk)
        for kk in 1:nk, jj in 1:nj, ii in 1:ni
            activeg[ii, jj, kk] = is_active[ii, jj, kk] != 0
        end
    else
        error("Unexpected IsActive shape $(dims). Expected (nk,nj,ni)=($(nk),$(nj),$(ni)) or (ni,nj,nk)=($(ni),$(nj),$(nk)).")
    end

    return activeg
end

function direction_stats(cxg, cyg, czg, activeg, dir::Symbol)
    mags = Float64[]
    off_axis = Float64[]

    ni, nj, nk = size(cxg)

    if dir == :i
        for i in 1:(ni - 1), j in 1:nj, k in 1:nk
            if !(activeg[i, j, k] && activeg[i + 1, j, k])
                continue
            end
            x1 = cxg[i, j, k]; x2 = cxg[i + 1, j, k]
            y1 = cyg[i, j, k]; y2 = cyg[i + 1, j, k]
            z1 = czg[i, j, k]; z2 = czg[i + 1, j, k]
            if !(isnan(x1) || isnan(x2) || isnan(y1) || isnan(y2) || isnan(z1) || isnan(z2))
                dx = x2 - x1
                dy = y2 - y1
                dz = z2 - z1
                push!(mags, sqrt(dx * dx + dy * dy + dz * dz))
                push!(off_axis, sqrt(dy * dy + dz * dz) / max(abs(dx), eps(Float64)))
            end
        end
    elseif dir == :j
        for i in 1:ni, j in 1:(nj - 1), k in 1:nk
            if !(activeg[i, j, k] && activeg[i, j + 1, k])
                continue
            end
            x1 = cxg[i, j, k]; x2 = cxg[i, j + 1, k]
            y1 = cyg[i, j, k]; y2 = cyg[i, j + 1, k]
            z1 = czg[i, j, k]; z2 = czg[i, j + 1, k]
            if !(isnan(x1) || isnan(x2) || isnan(y1) || isnan(y2) || isnan(z1) || isnan(z2))
                dx = x2 - x1
                dy = y2 - y1
                dz = z2 - z1
                push!(mags, sqrt(dx * dx + dy * dy + dz * dz))
                push!(off_axis, sqrt(dx * dx + dz * dz) / max(abs(dy), eps(Float64)))
            end
        end
    elseif dir == :k
        for i in 1:ni, j in 1:nj, k in 1:(nk - 1)
            if !(activeg[i, j, k] && activeg[i, j, k + 1])
                continue
            end
            x1 = cxg[i, j, k]; x2 = cxg[i, j, k + 1]
            y1 = cyg[i, j, k]; y2 = cyg[i, j, k + 1]
            z1 = czg[i, j, k]; z2 = czg[i, j, k + 1]
            if !(isnan(x1) || isnan(x2) || isnan(y1) || isnan(y2) || isnan(z1) || isnan(z2))
                dx = x2 - x1
                dy = y2 - y1
                dz = z2 - z1
                push!(mags, sqrt(dx * dx + dy * dy + dz * dz))
                push!(off_axis, sqrt(dx * dx + dy * dy) / max(abs(dz), eps(Float64)))
            end
        end
    else
        error("Unknown direction: $dir")
    end

    mag_s = stats(mags)
    off_mean = mean(off_axis)
    off_p95 = pctl(off_axis, 0.95)
    return (mag=mag_s, off_mean=off_mean, off_p95=off_p95)
end

function safe_corr(a::Vector{Float64}, b::Vector{Float64})
    @assert length(a) == length(b)
    if length(a) < 2
        return NaN
    end

    ma = mean(a)
    mb = mean(b)
    da = a .- ma
    db = b .- mb
    denom = sqrt(sum(abs2, da) * sum(abs2, db))

    if denom <= eps(Float64)
        # Treat two flat profiles with equal mean as perfectly matched.
        return isapprox(ma, mb; atol=1e-12, rtol=1e-9) ? 1.0 : NaN
    end
    return sum(da .* db) / denom
end

function main(path::String, h5_mask_path::String)
    @printf("Loading mapping from %s\n", path)
    df = load_mapping(path)

    @printf("DataFrame type: %s\n", string(typeof(df)))
    @printf("Rows: %d, Cols: %d\n", nrow(df), ncol(df))

    colnames_str = String.(names(df))

    req = vcat(["COORD1", "COORD2", "COORD3"], ["CORNER$(i)" for i in 1:24])
    missing = [c for c in req if !(c in colnames_str)]
    if !isempty(missing)
        error("Missing required columns: $(missing)")
    end

    i = Int.(df[!, "COORD1"])
    j = Int.(df[!, "COORD2"])
    k = Int.(df[!, "COORD3"])

    ni = maximum(i)
    nj = maximum(j)
    nk = maximum(k)
    nfull = ni * nj * nk
    occ = nrow(df) / nfull

    @printf("Grid extents from mapping: ni=%d nj=%d nk=%d\n", ni, nj, nk)
    @printf("Unique coords: |I|=%d |J|=%d |K|=%d\n", length(unique(i)), length(unique(j)), length(unique(k)))
    @printf("Occupancy ratio nrow/(ni*nj*nk): %.6f\n", occ)

    activeg = load_active_grid(h5_mask_path, ni, nj, nk)
    nactive = count(activeg)
    @printf("Active cells from mask: %d / %d (%.6f)\n", nactive, nfull, nactive / nfull)

    xcols = "CORNER" .* string.([1, 4, 7, 10, 13, 16, 19, 22])
    ycols = "CORNER" .* string.([2, 5, 8, 11, 14, 17, 20, 23])
    zcols = "CORNER" .* string.([3, 6, 9, 12, 15, 18, 21, 24])

    xmat = Matrix{Float64}(select(df, xcols))
    ymat = Matrix{Float64}(select(df, ycols))
    zmat = Matrix{Float64}(select(df, zcols))

    dx_cell = vec(maximum(xmat, dims=2) .- minimum(xmat, dims=2))
    dy_cell = vec(maximum(ymat, dims=2) .- minimum(ymat, dims=2))
    dz_cell = vec(maximum(zmat, dims=2) .- minimum(zmat, dims=2))
    vol_est = dx_cell .* dy_cell .* dz_cell

    row_active = [activeg[i[r], j[r], k[r]] for r in eachindex(i)]
    dx_cell = dx_cell[row_active]
    dy_cell = dy_cell[row_active]
    dz_cell = dz_cell[row_active]
    vol_est = vol_est[row_active]

    println("\nCell-size variability (bounding-box per active cell):")
    print_stats("dx_cell", stats(dx_cell))
    print_stats("dy_cell", stats(dy_cell))
    print_stats("dz_cell", stats(dz_cell))
    print_stats("vol_est", stats(vol_est))

    cx = vec(sum(xmat, dims=2) ./ 8.0)
    cy = vec(sum(ymat, dims=2) ./ 8.0)
    cz = vec(sum(zmat, dims=2) ./ 8.0)

    cxg = fill(NaN, ni, nj, nk)
    cyg = fill(NaN, ni, nj, nk)
    czg = fill(NaN, ni, nj, nk)

    for r in eachindex(i)
        ii = i[r]; jj = j[r]; kk = k[r]
        cxg[ii, jj, kk] = cx[r]
        cyg[ii, jj, kk] = cy[r]
        czg[ii, jj, kk] = cz[r]
    end

    si = direction_stats(cxg, cyg, czg, activeg, :i)
    sj = direction_stats(cxg, cyg, czg, activeg, :j)
    sk = direction_stats(cxg, cyg, czg, activeg, :k)

    println("\nCentroid neighbor spacing statistics (active-to-active pairs only):")
    print_stats("i-step |d|", si.mag)
    @printf("i-step off-axis ratio: mean=%.6f p95=%.6f\n", si.off_mean, si.off_p95)
    print_stats("j-step |d|", sj.mag)
    @printf("j-step off-axis ratio: mean=%.6f p95=%.6f\n", sj.off_mean, sj.off_p95)
    print_stats("k-step |d|", sk.mag)
    @printf("k-step off-axis ratio: mean=%.6f p95=%.6f\n", sk.off_mean, sk.off_p95)

    println("\nZ-spacing profile consistency across x/y (active-to-active k pairs):")
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

    all_dz = Float64[]
    for ii in 1:ni, jj in 1:nj, kk in 1:(nk - 1)
        v = dzg[ii, jj, kk]
        if !isnan(v)
            push!(all_dz, v)
        end
    end
    print_stats("k-step dz (all active pairs)", stats(all_dz))

    mean_profile = fill(NaN, nk - 1)
    k_cv_xy = Float64[]
    k_counts = Int[]
    for kk in 1:(nk - 1)
        vals = Float64[]
        for ii in 1:ni, jj in 1:nj
            v = dzg[ii, jj, kk]
            if !isnan(v)
                push!(vals, v)
            end
        end
        if !isempty(vals)
            mean_profile[kk] = mean(vals)
            push!(k_counts, length(vals))
            if length(vals) >= 2
                cvk = stats(vals).cv
                if !isnan(cvk)
                    push!(k_cv_xy, cvk)
                end
            end
        end
    end

    if !isempty(k_cv_xy)
        @printf(
            "per-k variability across x/y (CV): n_k=%d median=%.6f p95=%.6f min=%.6f max=%.6f\n",
            length(k_cv_xy),
            pctl(k_cv_xy, 0.50),
            pctl(k_cv_xy, 0.95),
            minimum(k_cv_xy),
            maximum(k_cv_xy),
        )
    else
        println("per-k variability across x/y (CV): no valid layers")
    end

    profile_corr = Float64[]
    profile_rel_rmse = Float64[]
    profile_valid_steps = Int[]
    for ii in 1:ni, jj in 1:nj
        prof = Float64[]
        refp = Float64[]
        for kk in 1:(nk - 1)
            v = dzg[ii, jj, kk]
            m = mean_profile[kk]
            if !(isnan(v) || isnan(m))
                push!(prof, v)
                push!(refp, m)
            end
        end
        if isempty(prof)
            continue
        end
        push!(profile_valid_steps, length(prof))

        if length(prof) >= 2
            c = safe_corr(prof, refp)
            if !isnan(c)
                push!(profile_corr, c)
            end
        end

        rmse = sqrt(mean((prof .- refp) .^ 2))
        scale = max(abs(mean(refp)), eps(Float64))
        push!(profile_rel_rmse, rmse / scale)
    end

    @printf(
        "profiles analyzed: %d / %d (%.6f)\n",
        length(profile_valid_steps),
        ni * nj,
        length(profile_valid_steps) / (ni * nj),
    )
    if !isempty(profile_valid_steps)
        @printf(
            "valid k-steps per profile: mean=%.2f p50=%.2f p95=%.2f min=%d max=%d\n",
            mean(profile_valid_steps),
            pctl(Float64.(profile_valid_steps), 0.50),
            pctl(Float64.(profile_valid_steps), 0.95),
            minimum(profile_valid_steps),
            maximum(profile_valid_steps),
        )
    end

    if !isempty(profile_corr)
        @printf(
            "corr(profile, mean_profile): n=%d mean=%.6f p05=%.6f p50=%.6f p95=%.6f min=%.6f\n",
            length(profile_corr),
            mean(profile_corr),
            pctl(profile_corr, 0.05),
            pctl(profile_corr, 0.50),
            pctl(profile_corr, 0.95),
            minimum(profile_corr),
        )
    else
        println("corr(profile, mean_profile): insufficient overlapping profile lengths")
    end

    if !isempty(profile_rel_rmse)
        @printf(
            "rel_rmse(profile vs mean_profile): n=%d mean=%.6f p50=%.6f p95=%.6f max=%.6f\n",
            length(profile_rel_rmse),
            mean(profile_rel_rmse),
            pctl(profile_rel_rmse, 0.50),
            pctl(profile_rel_rmse, 0.95),
            maximum(profile_rel_rmse),
        )

        profile_similar = (
            !isempty(k_cv_xy) &&
            pctl(k_cv_xy, 0.95) < 0.05 &&
            pctl(profile_rel_rmse, 0.95) < 0.05
        )
        if profile_similar
            println("z-profile conclusion: near-identical across x/y (within rough 5% tolerance)")
        else
            println("z-profile conclusion: changes materially across x/y")
        end
    else
        println("z-profile conclusion: insufficient data to compare profiles across x/y")
    end

    cv_dx = stats(dx_cell).cv
    cv_dy = stats(dy_cell).cv
    cv_dz = stats(dz_cell).cv
    cv_v = stats(vol_est).cv

    rectilinear_like = (si.off_p95 < 0.05) && (sj.off_p95 < 0.05) && (sk.off_p95 < 0.05)
    spacing_constant_like = (si.mag.cv < 0.05) && (sj.mag.cv < 0.05) && (sk.mag.cv < 0.05)
    cells_uniform_like = (cv_dx < 0.05) && (cv_dy < 0.05) && (cv_dz < 0.05) && (cv_v < 0.05)

    println("\nInterpretation thresholds:")
    @printf("rectilinear_like (p95 off-axis < 0.05 all dirs): %s\n", string(rectilinear_like))
    @printf("spacing_constant_like (CV |d| < 0.05 all dirs): %s\n", string(spacing_constant_like))
    @printf("cells_uniform_like (CV dx/dy/dz/vol < 0.05): %s\n", string(cells_uniform_like))
end

path = length(ARGS) >= 1 ? ARGS[1] : DEFAULT_PATH
h5_mask_path = length(ARGS) >= 2 ? ARGS[2] : DEFAULT_H5_MASK_PATH
main(path, h5_mask_path)
