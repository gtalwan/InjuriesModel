# Athlete Recovery Heterogeneity with ODE + EM

This repository studies a simple question with a non-simple answer:

**do athletes recover in meaningfully different ways after training load, or is a single pooled recovery model enough?**

The project is built around a two-stage idea:

1. use a simple fatigue-recovery ODE to summarize how each athlete responds to workload over time;
2. use EM / Gaussian mixture modeling to test whether those athlete-level summaries are better described by one common population or by several hidden recovery profiles.

The analysis finds that a single pooled recovery law is too coarse. The strongest model is a **3-profile EM solution** built from ODE-derived recovery features plus burden and recovery context. Those three profiles are interpretable:

- a **fast recovery / lower burden** group,
- an **intermediate recovery / higher burden** group,
- and a **slow recovery / high persistence** group.

The code in this repository is not just the notebook. The notebook is the narrative version; the `src/athlete_recovery/` package is the reusable implementation of the main workflow.

## Why this project exists

A lot of sports injury work studies workload, fatigue, or recovery session by session. That can be useful, but it also misses a basic point: a training session does not act on a blank slate. The effect of workload depends in part on the athlete's current state. If one athlete is already carrying fatigue and another is fresh, the same workload may not mean the same thing.

That creates two modeling problems.

First, the data are **dynamic**. Fatigue should build with load and fade with recovery. A static model does not represent that well.

Second, the population may be **heterogeneous**. Even if the same dynamical model is a good first approximation, different athletes may have different recovery speeds, different workload sensitivity, and different burden patterns. If that is true, one pooled model averages over those differences.

This repository is an attempt to make those two ideas concrete. The ODE gives a simple dynamic summary. The EM layer tests whether those summaries point to hidden recovery profiles.

## Main result

The central result is not just that workload relates to fatigue. That was expected. The more important result is that the athlete population is better described by **multiple hidden recovery profiles** than by one pooled recovery pattern.

The strongest evidence comes from three places:

- the ODE stage becomes statistically coherent after smoothing the observed fatigue proxy;
- the EM stage finds a stable 3-profile solution from athlete-level recovery summaries;
- those profiles differ on variables that were **not** used to build the clusters, especially post-onset recovery, sleep quality, and average workload.

That last point matters. If the clusters were only separating on the variables used to define them, the result would be much weaker. Instead, the hidden groups also show different outside behavior.

## Repository structure

```text
github_minimal_repo/
├── analysis_outputs/
│   ├── figure_8_em_extension.png
│   ├── figure_8b_em_robustness.png
│   ├── figure_8d_em_onset_profiles.png
│   ├── table_em_cluster_summary_primary.csv
│   ├── table_em_model_selection_primary.csv
│   └── ... other generated outputs
├── multimodal_sports_injury_dataset.zip
├── notebook/
│   └── sports_injury_em_ode_master_analysis.ipynb
├── report/
│   ├── sports_injury_report_overleaf.tex
│   └── sports_injury_report_overleaf.pdf
├── scripts/
│   └── run_core_analysis.py
├── src/
│   └── athlete_recovery/
│       ├── __init__.py
│       ├── data.py
│       ├── dynamics.py
│       ├── mixture.py
│       ├── pipeline.py
│       └── plots.py
├── requirements.txt
└── README.md
```

## Mathematical setup

### Stage 1: fatigue as a dynamical state

The starting point is a first-order fatigue-recovery model:

$$
\dot{x}(t) = \alpha u(t) - kx(t).
$$

Here:

- $x(t)$ is latent fatigue,
- $u(t)$ is workload,
- $\alpha > 0$ is the accumulation rate,
- $k > 0$ is the recovery rate.

This equation says something simple:

- training pushes fatigue up,
- recovery pulls fatigue down,
- and the pull back to baseline is proportional to how much fatigue is already present.

This is intentionally a small model. It is not meant to explain every detail of injury biology. It is meant to answer a narrower question: does the dataset support a basic accumulation-and-recovery structure strongly enough that we can use it as a first-stage summary?

### Observation model

The dataset does not directly measure the latent state $x(t)$. Instead it gives an observed fatigue proxy. The measurement model is

$$
y_i = x(t_i; \theta) + \epsilon_i,
\qquad
\epsilon_i \sim \mathcal{N}(0,\sigma^2).
$$

In the repository:

- `training_load` is used for $u(t)$,
- `fatigue_index` is used for $y(t)$.

That mapping was not chosen blindly. The notebook checks that `training_load` tracks the constructed proxy

$$
\texttt{training\_intensity} \times \texttt{training\_duration},
$$

and that it has a strong empirical relationship with `fatigue_index`.

### Discrete-time approximation

The dataset is indexed by session number, not exact clock time. That means the continuous ODE is used through its one-step discrete implication:

$$
y_{a,t+1} \approx c + \rho y_{a,t} + \alpha u_{a,t},
\qquad
\rho = 1-k.
$$

This gives a direct statistical test of the model:

- if the ODE is sensible, workload should have a **positive** next-step effect, so $\alpha > 0$;
- fatigue should persist but decay, so $0 < \rho < 1$, equivalently $k > 0$.

That is why the ODE is justified through regression diagnostics rather than treated as unquestioned truth.

### Why smoothing was necessary

The raw observed fatigue series is noisy. If one regresses the raw next-session fatigue on current fatigue and workload, the workload coefficient can fail the sign check. That is not evidence against the model; it is evidence that the observed proxy is too noisy to behave like the latent state directly.

To deal with that, the analysis uses a **trailing 3-session rolling mean**

$$
\tilde{y}_{a,t} = \frac{1}{3}\left(y_{a,t} + y_{a,t-1} + y_{a,t-2}\right),
$$

with the usual edge handling through `min_periods=1`.

This trailing smoother matters for two reasons:

- it reduces measurement noise;
- it avoids look-ahead leakage, because it uses only current and past values.

The working transition equation is therefore

$$
\tilde{y}_{a,t+1} \approx c + \rho \tilde{y}_{a,t} + \alpha u_{a,t} + \varepsilon_{a,t}.
$$

### Athlete-level recovery features

The ODE is not the final model in this repository. Its main job is to produce athlete-level summaries that can be used in the mixture model.

For each athlete, the code fits

$$
\tilde{y}_{a,t+1} = c_a + \rho_a \tilde{y}_{a,t} + \alpha_a u_{a,t} + \varepsilon_{a,t},
$$

then extracts:

- $\hat{\alpha}_a$: workload sensitivity,
- $\hat{k}_a = 1 - \hat{\rho}_a$: recovery rate,
- the recovery half-life,
- the athlete-specific transition $R^2$.

The half-life is defined by

$$
\text{half-life}_a = \frac{\log 2}{\hat{k}_a}.
$$

Because the time index is session order, this half-life is measured in **sessions**, not days.

Interpretation:

- a shorter half-life means fatigue decays more quickly;
- a longer half-life means fatigue persists longer after load.

This translation from $k$ to half-life is important because it turns a regression coefficient into something easier to interpret.

## Stage 2: EM for latent recovery profiles

### What is latent?

The latent variable is the athlete's unobserved recovery-profile label:

$$
z_a \in \{1,\dots,G\}.
$$

There is no dataset column for "fast responder" or "slow responder." If those patterns exist, they must be inferred from observed athlete-level summaries. That is the core reason EM is appropriate here.

### Primary clustering vector

The primary EM feature vector for athlete $a$ is

$$
\mathbf{r}_a
=
\left(
\hat{\alpha}_a,
\widehat{\text{half-life}}_a,
R_a^2,
\text{injury burden}_a,
\overline{\text{recovery}}_a
\right).
$$

These five variables were chosen on purpose.

The first three are dynamic summaries:

- how strongly workload moves fatigue,
- how quickly fatigue decays,
- and how coherent the simple transition model is for that athlete.

The last two anchor the dynamic summaries to actual burden and sustained recovery context:

- injured-session rate,
- mean recovery score.

This is a restricted feature set, and that is deliberate. The primary EM is trying to recover a **recovery profile**, not just cluster athletes on every measured variable.

### Gaussian mixture model

The model assumes

$$
z_a \sim \text{Categorical}(\pi_1,\dots,\pi_G),
$$

and conditional on profile membership,

$$
\mathbf{r}_a \mid z_a = g \sim \mathcal{N}(\mu_g, \Sigma_g).
$$

So each profile has:

- a mixing weight $\pi_g$,
- a mean vector $\mu_g$,
- and a covariance matrix $\Sigma_g$.

### EM algorithm

The EM algorithm alternates between two steps.

#### E-step

Compute the soft profile membership weights

$$
\tau_{ag}
=
P(z_a = g \mid \mathbf{r}_a)
=
\frac{
\pi_g \, \phi(\mathbf{r}_a; \mu_g, \Sigma_g)
}{
\sum_{h=1}^G \pi_h \, \phi(\mathbf{r}_a; \mu_h, \Sigma_h)
}.
$$

These are sometimes called **responsibilities**. They allow an athlete to be near the boundary between two profiles rather than forcing a hard assignment too early.

#### M-step

Update the mixture parameters from those weights:

$$
\pi_g = \frac{1}{n}\sum_{a=1}^n \tau_{ag},
$$

$$
\mu_g =
\frac{\sum_{a=1}^n \tau_{ag}\mathbf{r}_a}{\sum_{a=1}^n \tau_{ag}},
$$

$$
\Sigma_g =
\frac{\sum_{a=1}^n \tau_{ag}(\mathbf{r}_a-\mu_g)(\mathbf{r}_a-\mu_g)^\top}{\sum_{a=1}^n \tau_{ag}}.
$$

The code uses scikit-learn's `GaussianMixture`, but the logic is the same as the standard EM derivation.

## Why the restricted EM is the main model

A natural question is: why not put every variable into the clustering step?

The short answer is that doing so changes the question.

The primary EM is trying to uncover **latent recovery phenotype**. If too many contextual or burden variables are added at once, the clusters can drift toward:

- high burden vs low burden,
- sleep-rich vs sleep-poor,
- or other broad context splits,

instead of staying focused on the recovery mechanism.

There is also a sample-size issue. The raw dataset has more than 15,000 sessions, but the clustering step has only **156 athlete-level observations**. High-dimensional full-covariance mixtures can become parameter-heavy very quickly.

That is why the repo treats many additional variables as **withheld validation variables first**, including:

- mean sleep quality,
- mean workload,
- stress,
- hydration,
- physiology summaries,
- onset-based recovery summaries,
- age,
- BMI,
- sport,
- gender.

The logic is stricter this way:

1. define profiles from recovery-focused dynamic summaries;
2. then test whether those profiles differ on outside variables that were not used to build them.

If they do, that is stronger evidence than forcing those variables into the clustering step from the start.

## Statistical checks in the code

The repository does not rely on a single fit number.

### 1. ODE transition checks

The code checks whether the smoothed transition recovers:

- positive workload effect,
- positive recovery rate,
- and better one-step prediction than simpler baselines.

Functions:

- `athlete_recovery.dynamics.build_transition_panel`
- `athlete_recovery.dynamics.summarize_transition_models`
- `athlete_recovery.dynamics.grouped_cv_table`

### 2. Athlete-level stability

The code fits the smoothed transition separately within athletes and checks whether the signs remain sensible across the panel.

Function:

- `athlete_recovery.dynamics.athlete_transition_features`

### 3. Repeated EM fitting

The repository does not trust one lucky initialization. It refits the GMM many times for each profile count and stores:

- best BIC,
- median BIC,
- posterior certainty,
- entropy,
- silhouette,
- and pairwise agreement across restarts.

Functions:

- `athlete_recovery.mixture.repeated_gmm_selection`
- `athlete_recovery.mixture.select_stable_component_count`

### 4. External validation

After the clusters are estimated, the code checks whether they differ on variables that were not used to create them. For continuous variables it uses:

- ANOVA,
- Kruskal-Wallis,
- and $\eta^2$ effect size.

For categorical variables it uses:

- chi-square,
- and Cramer's $V$.

Functions:

- `athlete_recovery.mixture.fit_primary_em`
- `athlete_recovery.mixture.cluster_onset_trajectories`
- `athlete_recovery.mixture.cluster_specific_transitions`

## Key numerical results

The main restricted EM produced a stable **3-profile** solution. The three profile means are stored in:

- `analysis_outputs/table_em_cluster_summary_primary.csv`

The main profile summary is:

| Profile | Athletes | Mean half-life (sessions) | Injured-session rate | Mean recovery score |
|---|---:|---:|---:|---:|
| Fast recovery / lower burden | 53 | 1.75 | 0.120 | 57.6 |
| Intermediate recovery / higher burden | 70 | 2.10 | 0.166 | 53.9 |
| Slow recovery / high persistence | 33 | 2.87 | 0.165 | 54.2 |

The main model-selection output is stored in:

- `analysis_outputs/table_em_model_selection_primary.csv`

The most important interpretation is not that "3 is magic." It is that:

- one pooled profile is too simple;
- more complex solutions can sometimes fit better in a single run;
- but the **3-profile solution** is the best balance of fit, stability, and interpretability.

## Code overview

### `src/athlete_recovery/data.py`

Handles:

- loading the zipped dataset,
- validating required columns,
- imputing missing values,
- constructing `u_t`, `y_t`, smoothing variables, and injury-onset indicators.

### `src/athlete_recovery/dynamics.py`

Handles:

- building the one-step transition panel,
- pooled and demeaned transition regressions,
- grouped cross-validation,
- athlete-level ODE feature extraction,
- injury-onset summaries.

### `src/athlete_recovery/mixture.py`

Handles:

- athlete-level feature matrix construction,
- repeated GMM fitting,
- stable component-count selection,
- profile summaries,
- external validation,
- cluster-specific onset and transition diagnostics.

### `src/athlete_recovery/plots.py`

Generates the three main EM figures used in the report:

- model selection,
- profile structure,
- onset trajectories.

### `src/athlete_recovery/pipeline.py`

Ties the whole analysis together:

- run the core restricted ODE + EM workflow,
- return the key tables,
- and write the selected GitHub-ready outputs.

### `scripts/run_core_analysis.py`

Small command-line runner for reproducibility. This is the fastest way to regenerate the core outputs without stepping through the notebook.

## How to run the analysis

### Option 1: run the core pipeline

From the repository root:

```bash
python scripts/run_core_analysis.py
```

That writes selected tables and figures into `analysis_outputs/`.

To write them somewhere else:

```bash
python scripts/run_core_analysis.py --output-dir /tmp/recovery_outputs
```

### Option 2: run the notebook

Open:

```text
notebook/sports_injury_em_ode_master_analysis.ipynb
```

and run it top to bottom.

The notebook is useful if you want the full exploratory path and all of the intermediate tables. The source package is better if you want reproducible code that maps cleanly to the math.

## Report assets

The final report lives in:

- `report/sports_injury_report_overleaf.tex`
- `report/sports_injury_report_overleaf.pdf`

It uses selected outputs from `analysis_outputs/`, especially:

- `figure_8_em_extension.png`
- `figure_8b_em_robustness.png`
- `figure_8d_em_onset_profiles.png`
- `table_em_cluster_summary_primary.csv`
- `table_em_model_selection_primary.csv`

## Limitations

This repository is careful about what it claims.

The main limitations are:

1. **athlete-level sample size**  
   The raw panel is large, but the EM step still clusters only 156 athlete summaries.

2. **sequential uncertainty**  
   The ODE summaries are estimated first and clustered second, so first-stage uncertainty is not fully propagated into the EM stage.

3. **model simplicity**  
   The ODE is intentionally first-order. It is a useful summary model, not a complete physiological theory.

4. **taxonomy caution**  
   The results strongly support latent heterogeneity, but they do not prove that the current 3-profile solution is the final biological truth.

## Why this repository is structured the way it is

This GitHub version is deliberately small, but it is not stripped down to the point of being vague.

It keeps:

- the data archive,
- the main notebook,
- the final report,
- the generated outputs,
- and a proper source package with reusable modeling code.

It leaves out:

- duplicate working directories,
- stale notebook copies,
- scratch LaTeX files,
- and unrelated intermediate clutter.

The goal is a repo that is easy to read, honest about the modeling choices, and strong enough that the code actually supports the math being claimed.

## References

1. Impellizzeri, F. M., Ward, P., Coutts, A. J., Bornn, L., and McCall, A. (2020). *Training load and injury part 1: The devil is in the detail*. Journal of Orthopaedic and Sports Physical Therapy, 50(10), 574-576.
2. Imbach, F., Sutton-Charani, N., Montmain, J., Candau, R., and Perrey, S. (2022). *The use of fitness-fatigue models for sport performance modelling*. Sports Medicine - Open, 8, 29.
3. Dempster, A. P., Laird, N. M., and Rubin, D. B. (1977). *Maximum likelihood from incomplete data via the EM algorithm*. Journal of the Royal Statistical Society: Series B, 39(1), 1-22.
4. Bhegam, A. (2025). *Multimodal Sports Injury Prediction Dataset*. Kaggle dataset archive included in this repository.
