# Computational profile and reproduction

Four processes used one OpenBLAS/OMP thread each. The 20,000-iteration DLNS
chains took 133.7-136.0 seconds each, including their initial JIT
compilation. The one-inner-step SV pilots took 14.5-14.7 seconds per chain
at offset .001 and 17.5-19.1 seconds at offset zero, under potentially
different concurrent load.

An actual 2,000-transition corrected SV inner loop, conditioning on the
level OLS innovation series, took 7.66 seconds including first compilation.
After discarding 500 transitions, parameter-proposal and correction
acceptance were 0.521 and 0.227. Extrapolating this one-factor timing to
three factors and 18,000 outer iterations gives about 4.8 days per chain;
this is an order-of-magnitude estimate, not a measured full run. Memory
usage and effective posterior draws per CPU second were not measured. The
chains are unconverged, so time divided by their ESS is not a reliable
production-efficiency claim.

From the isolated worktree root in PowerShell, with the environment's
Python dependencies (NumPy, SciPy, Numba, pandas, ArviZ, Matplotlib):

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python -m unittest discover -s tests\hy_canada -v
python -m src.hy_canada.descriptive
python -m src.hy_canada.blocked_experiment --warmup 1000 --draws 2000 --grid-size 30 --seed-base 64101 --label dispersed30
python -m src.hy_canada.blocked_experiment --warmup 1000 --draws 2000 --grid-size 120 --seed-base 65101 --label dispersed120
python -m src.hy_canada.blocked_experiment --warmup 1000 --draws 2000 --grid-size 2001 --seed-base 66101 --label dispersed2001
python -m src.hy_canada.blocked_experiment --warmup 5000 --draws 15000 --grid-size 2001 --seed-base 67101 --label substantial
python -m src.hy_canada.blocked_sv_experiment --warmup 100 --draws 200 --grid-size 2001 --seed-base 77101 --offset 0.001 --label pilot
python -m src.hy_canada.blocked_sv_experiment --warmup 100 --draws 200 --grid-size 2001 --seed-base 77101 --offset 0 --label zero_offset
```

Check each run JSON for actual seeds and timings; commands overwrite outputs
with matching labels. Full parameter/factor/h paths are locally generated
under ignored `results/hy_canada/chains/`, not committed. Diagnostic CSVs,
run metadata and plots are retained. The older collapsed sampler and failed
outputs remain in Git history, not in the active implementation.
