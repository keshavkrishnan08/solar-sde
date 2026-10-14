# SolarSDE

**Probabilistic solar power nowcasting from sky images. One small latent stochastic differential
equation turns a sky-camera frame into a calibrated forecast distribution at every horizon from
1 to 30 minutes.**

![Two panels from the full run on held-out SKIPP'D data: on the left, SolarSDE has the lowest CRPS of six models at every horizon from 1 to 30 minutes; on the right, its 90% intervals cover 88% to 96% of outcomes, close to persistence, while the LSTM, MC-Dropout LSTM and CSDI baselines fall far below the 90% target](docs/crps-and-coverage.png)

A grid operator does not need one number for the next fifteen minutes. It needs to know how
much reserve to hold, at several lead times at once, and whether the forecast's uncertainty can
be trusted. Most sky-image forecasters generate future pixels and regress power from them at a
single horizon. SolarSDE treats the sky as a stochastic process instead. It encodes each frame
into a small cloud state, measures how turbulently that state is moving, and reads off a
distribution for any horizon in closed form. Intervals widen when the clouds churn and tighten
when the sky is clear.

```
SKIPP'D sky image (64x64) + rooftop PV, 1-minute cadence
    ->  cloud-state VAE (64-d latent) + optical-flow motion features
    ->  Cloud Turbulence Index: variance of latent velocity over the last 10 frames
    ->  transformer over 16 minutes of history (latent, clear-sky index, covariates)
    ->  mixture of 3 Ornstein-Uhlenbeck processes, volatility scaled by the CTI
    ->  closed-form marginal at horizon h, blended with a persistence component
    ->  conformal scaling per (horizon, CTI quartile)  ->  PV distribution, 1 to 30 min
```

Wide when the sky is churning, tight when it is clear.

## The core idea

1. **Forecast a distribution and audit it.** Every horizon reports CRPS together with the share
   of outcomes that land inside the 90% interval, so sharpness is never bought with bad coverage.
2. **Model the cloud state, not the pixels.** The latent evolves as a mixture of
   Ornstein-Uhlenbeck processes, whose mean and variance at any horizon have a closed form. One
   model serves 1, 5, 10, 15, 20 and 30 minutes with no image generation and no rollout. A second
   variant rolls the latent forward with Euler-Maruyama; the full run trains both.
3. **Let turbulence drive uncertainty.** The Cloud Turbulence Index (CTI) multiplies the SDE
   volatility, and the conformal correction is fit separately for each CTI quartile.
4. **Start from persistence.** The predictive distribution is a blend of a persistence component
   and the SDE mixture. The blend weight is capped per horizon, so the model only departs from
   persistence where that lowers CRPS.

## What it does (measured on held-out SKIPP'D test data)

One full run (`results_run_nb11_full.ipynb`, seed 42, Tesla T4). Unless noted, every model is
scored on the same window: the first 2,000 one-minute points of the chronological test split.
The SolarSDE rows are the rollout variant, which the run used for its downstream analyses. The
closed-form variant scored 0.33 to 1.32 kW CRPS on the same window and is the one in the SkyGPT
row.

| Capability | Metric | Result |
|---|---|---|
| Probabilistic accuracy, 1 to 30 min | CRPS (kW), SolarSDE vs persistence | **0.35 to 1.16** vs 0.50 to 2.17 |
| Skill over persistence | CRPS reduction at 1 min and at 30 min | **31%** and **47%** |
| Calibration | 90% interval coverage (PICP), 1 to 30 min | 0.96 to 0.88 |
| Learned baselines at 30 min | CRPS: LSTM / MC-Dropout LSTM / CSDI | 8.56 / 7.91 / 7.11 |
| Significance at 10 min | Diebold-Mariano vs persistence | DM = -15.3, p < 0.001 |
| Stratified at 10 min | CTI quartiles, sky regimes, ramp and non-ramp subsets won | **11 of 11** |
| Leave-one-month-out CV (6 folds) | CRPS / PICP at 15 min | 0.59 ± 0.29 / 0.93 ± 0.03 |
| Ramp warning | AUROC of interval width for ramp events | 0.90 to 0.91 |
| SkyGPT's own cloudy test (5 days, 2,582 windows), 15 min | CRPS vs published SkyGPT | 3.18 vs **2.81 (SkyGPT wins)** |

SolarSDE beats persistence and smart persistence at every horizon while holding 90% coverage
near target. The sequence baselines that work in observation space fall apart past 5 minutes:
the LSTM's 90% interval covers only 13% of outcomes at 20 minutes. The result that does not go
SolarSDE's way is the head-to-head on SkyGPT's hand-picked cloudy days, where it trails the
generative model and roughly ties smart persistence. The per-horizon tables are in
`results/RESULTS_nb11_full.md`, and every number above is printed in the notebook's cell outputs.

## How it works

**Data.** SKIPP'D (Stanford): 363,375 sky images at 64x64 with co-located rooftop PV at
1-minute cadence over 517 days. Days are split chronologically into 361 train, 78 validation and
78 test (246,234 / 62,554 / 54,587 frames); the test days fall in August to October. The target
is the change in clear-sky index, PV divided by an empirical clear-sky envelope (the 0.92
quantile of PV for each month and minute of day). A ramp is a one-minute PV change larger than
10% of clear-sky PV; 3.5% of frames are ramps.

**Cloud state.** A convolutional VAE compresses each frame to a 64-d latent (12 epochs,
beta = 0.01). Four optical-flow features describe cloud motion. The CTI is the norm of the
variance of latent velocity over a 10-frame window within each day.

**Forecaster.** A 2-layer transformer (d = 128, 4 heads) reads 16 steps of latent, clear-sky
index and 28 covariates. Linear heads give the weights, means, mean-reversion rates and
volatilities of 3 OU components; volatility is multiplied by (1 + gate(CTI)). The marginal at
horizon h is a Gaussian mixture in closed form, mixed with a zero-mean persistence component.
The model has 617,604 parameters and takes 12.8 ms for a 50-sample forecast on a T4.

**Training.** Closed-form Gaussian-mixture CRPS loss, AdamW (learning rate 5e-4, weight decay
1e-4, cosine to 1e-5), 60 epochs for the closed-form model and 35 for the rollout model, with
9,917 ramp anchors and 61,548 high-CTI anchors oversampled. Training took 85 minutes
(closed-form) and 171 minutes (rollout) on a T4.

**Calibration.** Mondrian split-conformal: a base scale per horizon and a multiplier per CTI
quartile, both chosen on validation and applied to the predicted standard deviations.

**Evaluation.** 50 Monte Carlo samples per forecast. CRPS, PICP, PINAW and RMSE against
persistence, smart persistence, an LSTM, an MC-Dropout LSTM and CSDI, all trained in the same
run. Diebold-Mariano tests, stratified analysis, leave-one-month-out CV, ramp AUROC, multi-level
reliability, and an exact replay of SkyGPT's published Nov to Dec 2019 cloudy test set.

## Run it

Everything runs from notebooks on a GPU runtime (Kaggle or Colab) with internet on. The
notebooks clone this repo and import the modules in `notebooks/`, then download SKIPP'D
(about 2.3 GB) from Hugging Face and SkyGPT's test set.

```bash
python notebooks/_final_master_generator.py      # rebuilds notebooks/11_final_publication.ipynb
# then open notebooks/11_final_publication.ipynb on a GPU runtime and Run All
```

`notebooks/12_fast_iterate.ipynb` is a lean loop (closed-form model, all horizons, SkyGPT
head-to-head) and `notebooks/13_arch_motion_sweep.ipynb` sweeps head architecture against motion
features; `_fast_iterate_generator.py` and `_sweep_notebook_generator.py` rebuild them. The
executed full run, with every output, is `results_run_nb11_full.ipynb`.

## Layout

```
notebooks/
  11_final_publication.ipynb    full run: data, VAE, both SDE variants, baselines, ablations, CV, SkyGPT test
  12_fast_iterate.ipynb         closed-form model only, all horizons + SkyGPT head-to-head
  13_arch_motion_sweep.ipynb    head architecture x motion-grid sweep
  08*, 09, 10                   earlier master notebooks (NREL Golden data, first SKIPP'D runs)
  _solarsde_v2.py               closed-form model: transformer history, OU mixture, persistence blend, conformal
  _solarsde_rollout.py          Euler-Maruyama latent rollout variant
  _skippd_pipeline.py           SKIPP'D download, clear-sky index, ramps, splits, VAE, CTI
  _ensemble_eval.py             dual-architecture training glue, ensemble, champion selection on validation
  _skygpt_eval.py               SkyGPT exact cloudy test set and published comparison numbers
  _multisite.py                 SURFRAD and NSRDB downloaders (not used in the reported run)
  _*_generator.py               build the notebooks from the code blocks above
results_run_nb11_full.ipynb     the executed full run with all outputs
results/                        run summaries (nb11, nb12, nb13) and a controlled multi-seed study (def_*.csv)
paper/                          manuscript drafts (Markdown), figures, literature review
colab_outputs/                  VAE checkpoints and latents from the earlier NREL Golden pipeline
data/                           SKIPP'D test parquet and labels, SURFRAD station files
```

## Honesty

* The headline table is one seed, scored on the first 2,000 points of the chronological test
  split (`N_EVAL` in `_skippd_pipeline.py`). That is roughly the first three of the 78 test days,
  at the August start of the test period, not the whole test set. Every model shares that window, and the
  leave-one-month-out CV is the broader check.
* SolarSDE does not beat SkyGPT on SkyGPT's cloudy test. At 15 minutes it scores 3.18 kW CRPS
  (3.03 in a later closed-form run, `results/RESULTS_nb12_base.md`) against SkyGPT's published
  2.81, and it roughly ties smart persistence on those days (3.18 vs 3.105). Adding grid-pooled
  optical flow to close the gap made it worse (3.63, `results/RESULTS_nb13_sweep.md`).
* The stochastic parts are not what wins. In the rollout ablation at 10 minutes, a deterministic
  version with no diffusion scored 1.940 CRPS against 1.961 for the full model, and removing the
  CTI gate cost little (1.976). The persistence anchor (2.141 without it) and the covariates
  (2.128 without them) matter more. The ablation used an 800-point window, so its absolute values
  do not compare with the main table.
* Calibration is tuned for the 90% interval. At other levels it is off at short horizons: at
  1 minute the nominal 50% interval covers 78% of outcomes, and the mean calibration error across
  levels is 0.073. In the controlled multi-seed study (`results/def_pit.csv`) the PIT uniformity
  test is rejected at every horizon, with a PIT mean of 0.54 to 0.56 (a small under-forecast bias).
  Out of sample, 30-minute coverage drops to 0.80 ± 0.13 in leave-one-month-out CV.
* The CTI is only weakly correlated with GHI variability (Spearman 0.30), although forecast error
  is much higher in the top CTI quartile (2.96 kW CRPS at 10 minutes against 0.16 to 0.28 in the
  other three).
* The notebook prints two CAISO reserve-savings estimates that disagree by a factor of about 77
  ($122M and $1.59M per GW-year), so no dollar value is claimed here.
* The drafts in `paper/` predate this run and quote figures it does not reproduce (0.43M
  parameters, about 4 ms per forecast, PIT not rejected). Use the run outputs.
* There is no requirements file, test suite or command-line entry point; the notebooks are the
  interface. SURFRAD files sit in `data/surfrad/`, but no multi-site results are in the repo.

MIT licensed.
