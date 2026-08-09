# cCKF TrackState Branch Reference

**ACTS fork revision:** dcd07763c (feat(cckf): write cluster shape/charge and track incidence angles)
**Verifier:** `scripts/instrumentation/check_trackstate_branches.py`

## Branches added

All branches below are **state-aligned**: `len(branch[i]) == nStates[i]` for
every track `i`, so they are index-parallel with `volume_id`, `layer_id`,
`module_id`, `stateType`, `chi2` and `pathLength`.

| Branch | Type | Patch | Meaning | Sentinel |
|---|---|---|---|---|
| `S00_prt` | float | A | Innovation covariance S(0,0) at the predicted state | NaN |
| `S01_prt` | float | A | S(0,1) | NaN; also NaN for 1D measurements |
| `S11_prt` | float | A | S(1,1) | NaN; also NaN for 1D measurements |
| `pathInX0_interval` | float | B | Material X/X0 since the previous measurement surface | NaN if the CKF column is absent |
| `clus_size_u` | int | C | Cluster size along local u, channels | -1 |
| `clus_size_v` | int | C | Cluster size along local v, channels | -1 |
| `clus_qtot` | float | C | Total cluster charge | NaN |
| `clus_sigma_uu` | float | C | Charge-weighted second central moment in u, mm^2 | NaN |
| `clus_sigma_uv` | float | C | Charge-weighted mixed second moment, mm^2, signed | NaN |
| `clus_sigma_vv` | float | C | Charge-weighted second central moment in v, mm^2 | NaN |
| `alpha_u` | float | C | atan2(d.u_hat, d.n_hat), radians | NaN |
| `alpha_v` | float | C | atan2(d.v_hat, d.n_hat), radians | NaN |

## Fields that must be joined offline

**None.** Every quantity in the Phase 1 request is written directly.

The cluster join that the task statement anticipated might be needed offline
happens in-memory in the writer instead: `ClusterContainer` is one-to-one with
measurements, and the writer already has the measurement index from the
`IndexSourceLink`, so `clusters[sl.index()]` resolves at write time.

## Known caveats

1. **Incidence angles are stored as angles, not ratios.** The spec asked for
   `d.u_hat / d.n_hat`. That diverges at grazing incidence. We store
   `atan2(d.u_hat, d.n_hat)`; recover the ratio with `np.tan(alpha_u)`.

2. **`sigma_*` are population moments in mm^2.** A single-channel cluster
   gives exactly 0, not NaN. Downstream code should not treat 0 as missing --
   use `clus_size_u < 0` to test for "no cluster".

3. **The legacy `*_hit` branches are NOT state-aligned.** `dim_hit`,
   `res_x_hit`, `err_x_hit`, `pull_x_hit` and their `_y` counterparts are
   shorter than `nStates` because the writer's parameter loop `continue`s past
   states without a source link. Do not zip them against the branches above.
   Use `S00_prt` / `S11_prt` for the innovation covariance instead of
   `err_*_hit`.

4. **`pathInX0_interval` is only populated by the CKF.** Fitter output (KF,
   GSF, GX2F) does not register the dynamic column, so it reads back NaN there.

5. **Patch C requires geometric digitization.** Smearing-only digi produces no
   clusters; the `clus_*` branches would be all-sentinel.

## Numerical no-op

Verify with:

    python scripts/instrumentation/check_trackstate_branches.py \
        output/instr_all/trackstates_ckf.root \
        --baseline output/instr_baseline/trackstates_ckf.root

## Commits

- Patch A: `737914255` feat(cckf): write full innovation covariance S00/S01/S11 per track state
- Patch C1: `cdcd53b81` feat(cckf): add cluster charge-moment and incidence-angle helpers
- Patch B: `d496745a7` feat(cckf): record accumulated material X/X0 per measurement interval
- Patch C2: `dcd07763c` feat(cckf): write cluster shape/charge and track incidence angles
