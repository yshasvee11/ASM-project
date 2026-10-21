# Applied Stochastic Modelling (ASM) Course Project
# Master Mathematical Formulation & Empirical Stochastic Analysis

**Project Title:** Hybrid Multi-Scale Queueing & Continuous-Time Markov Network for Hospital Emergency-to-ICU Patient Flow Under Resource Contention  
**Domain:** Operations Research & Applied Stochastic Processes  
**Datasets:** eICU Collaborative Research Database Demo v2.0.1 & CMS Timely and Effective Care (yv7e-xc69)  
**Authoring Group:** ASM Course Project Team  

---

## Part 1: Mathematical Foundations & Stochastic Architecture

### 1. System Topology & Stochastic Network Formulation

We consider a two-tier stochastic service network operating in continuous time $t \ge 0$. The system consists of:
1. **Stage 1 (Upstream):** An Emergency Department (ED) modeled as an Erlang-A queue ($M_t / M / c_1 + M$) with non-stationary arrivals and patient abandonment.
2. **Stage 2 (Downstream):** An Intensive Care Unit (ICU) modeled as a $G / G / c_2 / K$ queue with general service times, finite bed capacity $c_2$, and upstream boarding/blocking.
3. **Stage 3 (Routing):** A Continuous-Time Markov Chain (CTMC) over a finite state space governing patient transfers between hospital wards until absorption (discharge or mortality).

```
   NHPP Arrival Process
      lambda(t)
         │
         ▼
  +─────────────────────────────────────────────────────────────+
  │ STAGE 1: Emergency Department (ED)                          │
  │ Queue Capacity: Infinite (Waiting Room)                     │
  │ Service Capacity: c_1 Treatment Bays                        │
  │ Patience Time: tau ~ Exp(theta)  --> Abandonment Rate theta │
  │ Service Duration: S_ED ~ Exp(mu_ED)                         │
  +─────────────────────────────────────────────────────────────+
         │                                       │
         │ (1 - p_ICU)                           │ p_ICU
         ▼                                       ▼
  [General Floor / Discharge]             ICU Bed Available?
                                            /           \
                                      YES  /             \  NO
                                          ▼               ▼
                       +──────────────────────+  +──────────────────────+
                       │ STAGE 2: ICU Service │  │ ED Boarding Block    │
                       │ c_2 Dedicated Beds   │  │ Patient holds c_1    │
                       │ S_ICU ~ Weibull/Gamma│  │ bay until ICU bed    │
                       +──────────────────────+  │ becomes available    │
                                  │              +──────────────────────+
                                  ▼                         │
                       +──────────────────────+             │
                       │ Multi-Ward CTMC      │◄────────────┘
                       │ SDU, Floor, Home,    │
                       │ Expired (Absorbing)  │
                       +──────────────────────+
```

---

### 2. Zero-Scale-Mismatch Diurnal Arrival Process ($\text{NHPP}$)

#### Mathematical Formulation
Let $\{N(t), t \ge 0\}$ be a Non-Homogeneous Poisson Process (NHPP) representing the cumulative number of patient arrivals to the Emergency Department in $[0, t]$. The process is characterized by:
1. $N(0) = 0$.
2. For any $0 \le t_1 < t_2 < \dots < t_k$, the increments $N(t_i) - N(t_{i-1})$ are independent random variables.
3. For any $s, t \ge 0$, the increment $N(t + s) - N(t)$ follows a Poisson distribution:
   $$\mathbb{P}(N(t + s) - N(t) = k) = \frac{\left(m(t+s) - m(t)\right)^k}{k!} \exp\left(-\left(m(t+s) - m(t)\right)\right), \quad k \in \{0, 1, 2, \dots\}$$
   where the mean value function is:
   $$m(t) = \int_0^t \lambda(u) \, du$$

#### Diurnal Intensity Calibration
The continuous arrival intensity $\lambda(t)$ exhibits a 24-hour diurnal cycle:
$$\lambda(t + 24) = \lambda(t), \quad \forall t \ge 0$$

To eliminate scale mismatch between patient-level timestamps and facility-level annual reports, we decompose $\lambda(t)$ into a piecewise-constant hourly rate over hour intervals $h \in \{0, 1, \dots, 23\}$:
$$\lambda(t) = \lambda(h), \quad \text{for } t \in [h, h+1)$$

Let $n_h$ denote the empirical count of ED patient arrivals occurring during hour $h$ extracted from `hospitaladmittime24` in `eicu_demo_data/patient.csv`. The normalized diurnal arrival probability vector is:
$$p_h = \frac{n_h}{\sum_{k=0}^{23} n_k}, \quad \text{such that } \sum_{h=0}^{23} p_h = 1.0$$

Let $N_{\text{annual}}$ denote the facility's total annual ED arrival volume from `cms_data/` (Measure `OP_22`, column `sample`). The mean daily arrival volume is:
$$\Lambda_{\text{day}} = \frac{N_{\text{annual}}}{365}$$

The hourly arrival rate $\lambda(h)$ is defined as:
$$\lambda(h) = \Lambda_{\text{day}} \cdot p_h, \quad h \in \{0, 1, \dots, 23\}$$

**Conservation Property (Zero Scale Mismatch):**
$$\int_0^{24} \lambda(t) \, dt = \sum_{h=0}^{23} \lambda(h) = \sum_{h=0}^{23} (\Lambda_{\text{day}} \cdot p_h) = \Lambda_{\text{day}} \sum_{h=0}^{23} p_h = \Lambda_{\text{day}}$$

---

### 3. Stage 1: Upstream Emergency Department Queue ($M_t / M / c_1 + M$)

#### Queue Specification
* **Arrival Process:** NHPP with time-varying intensity $\lambda(t)$.
* **Server Capacity:** $c_1$ acute care ED treatment bays.
* **Service Mechanism:** Individual patient service times are independent and identically distributed (i.i.d.) exponential random variables:
  $$S_{\text{ED}} \sim \text{Exponential}(\mu_{\text{ED}}), \quad \mathbb{E}[S_{\text{ED}}] = \frac{1}{\mu_{\text{ED}}}$$
* **Patient Abandonment (Reneging):** Arriving patients have finite patience durations $\tau$ before abandoning the queue without being seen (Left Without Being Seen, LWBS):
  $$\tau \sim \text{Exponential}(\theta), \quad \mathbb{E}[\tau] = \frac{1}{\theta}$$

#### Birth-Death Rate Formulation
Let $X(t) \in \{0, 1, 2, \dots\}$ denote the total number of patients present in the ED (waiting + in service) at time $t$. Under stationary arrival rate $\lambda$, the transition rates $q_{n, m}$ of the underlying birth-death process are:

$$\lambda_n = q_{n, n+1} = \lambda, \quad \forall n \ge 0$$

$$\mu_n = q_{n, n-1} = \begin{cases}
n \mu_{\text{ED}}, & \text{if } 1 \le n \le c_1 \quad (\text{all patients in service}) \\
c_1 \mu_{\text{ED}} + (n - c_1) \theta, & \text{if } n > c_1 \quad (\text{service plus queue abandonment})
\end{cases}$$

#### Steady-State Probability Distribution
The stationary distribution $\pi_n = \lim_{t \to \infty} \mathbb{P}(X(t) = n)$ satisfies the global balance equations:
$$\pi_n = \pi_0 \prod_{k=1}^n \frac{\lambda_{k-1}}{\mu_k}$$

Explicitly:
$$\pi_n = \begin{cases}
\pi_0 \dfrac{(\lambda / \mu_{\text{ED}})^n}{n!}, & 0 \le n \le c_1 \\[12pt]
\pi_0 \dfrac{(\lambda / \mu_{\text{ED}})^{c_1}}{c_1!} \prod_{j=1}^{n - c_1} \dfrac{\lambda}{c_1 \mu_{\text{ED}} + j \theta}, & n > c_1
\end{cases}$$
where $\pi_0$ is obtained from the normalization condition $\sum_{n=0}^\infty \pi_n = 1$:
$$\pi_0 = \left[ \sum_{n=0}^{c_1} \frac{(\lambda / \mu_{\text{ED}})^n}{n!} + \frac{(\lambda / \mu_{\text{ED}})^{c_1}}{c_1!} \sum_{m=1}^\infty \prod_{j=1}^m \frac{\lambda}{c_1 \mu_{\text{ED}} + j \theta} \right]^{-1}$$

#### Erlang-A Equilibrium Relations & Empirical Calibration
In the Erlang-A model, the steady-state abandonment probability $P_{\text{abandon}}$ and the mean waiting time in the queue $\mathbb{E}[W_q]$ satisfy the exact structural relation:
$$\mathbb{E}[W_q] = \frac{P_{\text{abandon}}}{\theta}$$

From Little's Law, the total sojourn time $W_{\text{ED}}$ (queue waiting time + treatment service time) is:
$$W_{\text{ED}} = \mathbb{E}[W_q] + \mathbb{E}[S_{\text{ED}}] = \frac{P_{\text{abandon}}}{\theta} + \frac{1}{\mu_{\text{ED}}}$$

**Empirical Identification:**
1. From CMS Measure `OP_22`:
   $$P_{\text{abandon}} = \frac{\text{Score}_{\text{OP\_22}}}{100}$$
2. From CMS Measure `OP_18b`:
   $$W_{\text{ED}} = \text{Score}_{\text{OP\_18b}} \quad (\text{in minutes})$$
3. Calibrating patient patience rate $\theta$ and service rate $\mu_{\text{ED}}$:
   $$\frac{1}{\mu_{\text{ED}}} = W_{\text{ED}} - \frac{P_{\text{abandon}}}{\theta}$$

---

### 4. Stage 2: Downstream ICU Queue ($G / G / c_2 / K$) with Boarding Block

#### Routing & Patient Classification
Arriving ED patients are routed independently via a Bernoulli filter:
* A critical fraction $p_{\text{ICU}} \in [0.10, 0.15]$ require ICU admission.
* The remaining fraction $1 - p_{\text{ICU}}$ are treated and discharged or transferred to general medical floors.

#### Parametric Service Time Distribution Families ($S_{\text{ICU}}$)
We evaluate four parametric families for ICU length of stay $S_{\text{ICU}}$:

1. **Exponential Distribution ($M/M/c_2$ baseline):**
   $$f(t; \lambda) = \lambda e^{-\lambda t}, \quad t \ge 0, \quad \mathbb{E}[S] = \frac{1}{\lambda}, \quad \text{Var}(S) = \frac{1}{\lambda^2}$$
   *Hazard function:* $h(t) = \lambda$ (Constant / Memoryless).

2. **Weibull Distribution:**
   $$f(t; c, \beta) = \frac{c}{\beta} \left(\frac{t}{\beta}\right)^{c-1} \exp\left(-\left(\frac{t}{\beta}\right)^c\right), \quad t \ge 0$$
   *Hazard function:* $h(t) = \frac{c}{\beta} \left(\frac{t}{\beta}\right)^{c-1}$.
   *Shape property:* $c < 1 \implies$ decreasing failure rate (heavy tail); $c = 1 \implies$ exponential; $c > 1 \implies$ increasing failure rate (aging).

3. **Gamma Distribution:**
   $$f(t; \alpha, \beta) = \frac{1}{\Gamma(\alpha) \beta^\alpha} t^{\alpha - 1} \exp\left(-\frac{t}{\beta}\right), \quad t \ge 0$$
   $$\mathbb{E}[S] = \alpha \beta, \quad \text{Var}(S) = \alpha \beta^2$$

4. **Log-Normal Distribution:**
   $$f(t; \mu, \sigma) = \frac{1}{t \sigma \sqrt{2\pi}} \exp\left( -\frac{(\ln t - \mu)^2}{2\sigma^2} \right), \quad t > 0$$
   $$\mathbb{E}[S] = \exp\left(\mu + \frac{\sigma^2}{2}\right), \quad \text{Var}(S) = \left[\exp(\sigma^2) - 1\right] \exp(2\mu + \sigma^2)$$

#### Maximum Likelihood Estimation (MLE) Formulation
Given an independent sample of observed durations $\mathbf{x} = (x_1, x_2, \dots, x_n)$, the parameter vector $\hat{\boldsymbol{\theta}}$ is obtained by maximizing the log-likelihood function:
$$\hat{\boldsymbol{\theta}}_{\text{MLE}} = \arg\max_{\boldsymbol{\theta}} \ln L(\boldsymbol{\theta}; \mathbf{x}) = \arg\max_{\boldsymbol{\theta}} \sum_{i=1}^n \ln f(x_i; \boldsymbol{\theta})$$

#### Model Selection Criteria
1. **Akaike Information Criterion (AIC):**
   $$\text{AIC} = 2k - 2\ln L(\hat{\boldsymbol{\theta}})$$
   where $k$ is the number of estimated parameters.
2. **Bayesian Information Criterion (BIC):**
   $$\text{BIC} = k \ln(n) - 2\ln L(\hat{\boldsymbol{\theta}})$$
3. **Kolmogorov-Smirnov (KS) Goodness-of-Fit Statistic:**
   $$D = \sup_{x \in \mathbb{R}} \left| F_n(x) - F(x; \hat{\boldsymbol{\theta}}) \right|$$
   where $F_n(x) = \frac{1}{n} \sum_{i=1}^n \mathbb{I}_{x_i \le x}$ is the empirical cumulative distribution function.

#### Upstream Boarding / Server Blocking Formulation
Let $c_2$ denote the number of physical ICU beds. Let $B(t)$ denote the number of patients who have completed ED treatment, have been admitted to the ICU, but are physically held in an ED bay because all $c_2$ ICU beds are occupied:
$$B(t) = \max\left(0, N_{\text{ICU}}(t) - c_2\right)$$

Under this resource-contention coupling, the effective number of operational ED treatment bays $c_1^{\text{eff}}(t)$ available to serve incoming acute ED patients is reduced dynamically:
$$c_1^{\text{eff}}(t) = \max\left(0, c_1 - B(t)\right)$$

This cross-stage coupling induces state-dependent upstream queuing bottlenecks, causing dramatic surges in ED wait times $\mathbb{E}[W_q]$ and patient abandonment $P_{\text{abandon}}$.

---

### 5. Multi-Ward Patient Care Trajectory (Absorbing DTMC & CTMC)

#### State Space Definition
We define the patient clinical routing state space as:
$$S = \{1: \text{ED}, \, 2: \text{ICU}, \, 3: \text{SDU}, \, 4: \text{Floor}, \, 5: \text{Home}, \, 6: \text{Expired}\}$$
* **Transient States:** $S_T = \{1, 2, 3, 4\}$
* **Absorbing States:** $S_A = \{5: \text{Home (Live Discharge)}, \, 6: \text{Expired (Mortality)}\}$

#### Discrete-Time Markov Chain (DTMC) Formulation
The one-step transition probability matrix $P$ in canonical block form is:
$$P = \begin{pmatrix}
Q_T & R \\
\mathbf{0} & I
\end{pmatrix}$$
where:
* $Q_T \in \mathbb{R}^{4 \times 4}$ governs transitions between transient hospital wards.
* $R \in \mathbb{R}^{4 \times 2}$ governs transitions from transient wards into absorbing states (Home vs. Expired).
* $I \in \mathbb{R}^{2 \times 2}$ is the identity matrix for absorbing states.
* $\mathbf{0} \in \mathbb{R}^{2 \times 4}$ represents zero exit probability from absorbing states.

#### Fundamental Matrix & Absorption Analysis
The fundamental matrix $N$ of the absorbing Markov chain is:
$$N = (I - Q_T)^{-1} = \sum_{k=0}^\infty Q_T^k$$

**Physical Interpretation:**
* The entry $N_{ij}$ represents the expected number of times a patient visits transient ward $j$ prior to absorption, given that the patient started in ward $i$.
* The vector of expected total stages before discharge/death is:
  $$\mathbf{t} = N \mathbf{1}$$
  where $\mathbf{1} = (1, 1, 1, 1)^T$.
* The absorption probability matrix $B \in \mathbb{R}^{4 \times 2}$ is:
  $$B = N R$$
  where $B_{i, 5}$ is the probability that a patient initially admitted to ward $i$ eventually leaves the hospital alive (Home), and $B_{i, 6}$ is the overall probability of hospital mortality.

#### Continuous-Time Markov Chain (CTMC) Formulation
Let $\{Y(t), t \ge 0\}$ be the continuous-time Markov process on state space $S$. The process is governed by the infinitesimal generator matrix $Q = [q_{ij}]$:

$$q_{ij} \ge 0, \quad \forall i \neq j$$

$$q_{ii} = - \sum_{j \neq i} q_{ij} = -\frac{1}{\mathbb{E}[S_i]}$$
where $\mathbb{E}[S_i]$ is the mean holding duration in ward $i$.

The off-diagonal transition rates are given by:
$$q_{ij} = -q_{ii} \cdot p_{ij} = \frac{p_{ij}}{\mathbb{E}[S_i]}, \quad \forall i \neq j$$

#### Transient State Occupancy Distribution
The continuous-time transition probability function $P(t) = [p_{ij}(t)]$, where $p_{ij}(t) = \mathbb{P}(Y(t) = j \mid Y(0) = i)$, satisfies the Kolmogorov Forward Equation:
$$\frac{d}{dt} P(t) = P(t) Q, \quad P(0) = I$$

The unique analytical solution is given by the matrix exponential:
$$P(t) = \exp(Q t) = \sum_{k=0}^\infty \frac{(Q t)^k}{k!}$$
computed numerically via Padé approximation (`scipy.linalg.expm`).

---

## Part 2: Empirical Outputs & Statistical Validation

### 1. Cohort Statistics Summary

#### Micro Patient-Level Data (`eicu_demo_data/patient.csv`)
* **Total ICU Stays ($N_{\text{stays}}$):** 2,520
* **Unique Hospital Stays ($N_{\text{admissions}}$):** 2,174
* **Unique Patients ($N_{\text{patients}}$):** 1,841
* **Hospitals Represented:** 186
* **ICU Discharge Mortality:** 5.00%
* **Hospital Discharge Mortality:** 8.41%
* **ED-Admitted ICU Stays:** 1,304 (51.75% of all ICU stays)
* **Valid ED Boarding Records:** 1,253

#### Pre-ICU Boarding Duration ($B_{\text{ED}\to\text{ICU}} = -\text{hospitaladmitoffset}$):
* $\text{Mean} = 267.7 \text{ minutes } (4.46 \text{ hours})$
* $\text{Median} = 158.0 \text{ minutes } (2.63 \text{ hours})$
* $\text{Interquartile Range (IQR)} = [81.0, 290.0] \text{ minutes}$

#### ICU Length of Stay ($S_{\text{ICU}} = \text{unitdischargeoffset}$):
* $\text{Mean} = 58.07 \text{ hours } (2.42 \text{ days})$
* $\text{Median} = 35.33 \text{ hours } (1.47 \text{ days})$
* $\text{Standard Deviation } \sigma = 82.95 \text{ hours}$
* $\text{Coefficient of Variation (CV)} = \frac{\sigma}{\mu} = \frac{82.95}{58.07} = 1.429 > 1.0$

---

### 2. CMS Facility-Level ED Benchmarks by Volume Tier

$$\text{Data Source: CMS Timely and Effective Care (Measure OP\_22 and OP\_18b)}$$

| Volume Tier | Facility Count | Annual Volume ($N_{\text{annual}}$) | Daily Rate ($\Lambda_{\text{day}}$) | Abandonment ($P_{\text{abandon}}$) | Median Sojourn ($W_{\text{ED}}$) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Low** | 19 | $10,780.2 \pm 3,421$ | $29.53 \text{ pts/day}$ | $2.21\%$ | $128.2 \text{ min } (2.14 \text{ h})$ |
| **Medium** | 15 | $30,747.3 \pm 6,180$ | $84.24 \text{ pts/day}$ | $2.87\%$ | $181.7 \text{ min } (3.03 \text{ h})$ |
| **High** | 8 | $51,920.6 \pm 5,890$ | $142.25 \text{ pts/day}$ | $2.25\%$ | $173.0 \text{ min } (2.88 \text{ h})$ |
| **Very High** | 6 | $100,448.2 \pm 28,140$ | $275.20 \text{ pts/day}$ | $2.33\%$ | $202.5 \text{ min } (3.38 \text{ h})$ |

---

### 3. Diurnal NHPP Arrival Intensity ($\lambda(h) = \Lambda_{\text{day}} \cdot p_h$)

* **Sample Size:** 1,304 ED patient arrival timestamps.
* **Circadian Peak:** $h = 20$ (8:00 PM), $p_{20} = 0.0621 \implies 6.21\%$ of daily arrivals.
* **Circadian Trough:** $h = 11$ (11:00 AM), $p_{11} = 0.0215 \implies 2.15\%$ of daily arrivals.
* **Diurnal Ratio (Peak / Trough):** $\frac{p_{20}}{p_{11}} = \frac{0.0621}{0.0215} = 2.89 \times$

$$\text{Hourly Arrival Intensity } \lambda(h) \text{ (Patients per Hour):}$$

| Hour ($h$) | $p_h$ | Low Tier ($\Lambda = 29.53$) | Med Tier ($\Lambda = 84.24$) | High Tier ($\Lambda = 142.25$) | Very High Tier ($\Lambda = 275.20$) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **00:00** | 0.0483 | 1.43 | 4.07 | 6.87 | 13.29 |
| **03:00** | 0.0399 | 1.18 | 3.36 | 5.67 | 10.97 |
| **06:00** | 0.0284 | 0.84 | 2.39 | 4.04 | 7.81 |
| **09:00** | 0.0238 | 0.70 | 2.00 | 3.38 | 6.54 |
| **11:00 (Trough)** | **0.0215** | **0.63** | **1.81** | **3.05** | **5.91** |
| **15:00** | 0.0460 | 1.36 | 3.88 | 6.55 | 12.66 |
| **18:00** | 0.0544 | 1.61 | 4.59 | 7.75 | 14.98 |
| **20:00 (Peak)** | **0.0621** | **1.83** | **5.23** | **8.83** | **17.09** |
| **23:00** | 0.0514 | 1.52 | 4.33 | 7.31 | 14.14 |

---

### 4. Downstream ICU Service Time ($S_{\text{ICU}}$) Distribution Fitting

$$\text{Sample Size: } n = 2,508 \text{ non-zero ICU stays } (x > 0)$$

$$\text{Parameter Estimates: } \hat{\boldsymbol{\theta}}_{\text{MLE}}, \quad \text{Model Selection: AIC, BIC, and Kolmogorov-Smirnov Test}$$

| Distribution | $k$ | Fitted Parameters ($\hat{\boldsymbol{\theta}}$) | $\ln L$ | $\text{AIC}$ | $\Delta\text{AIC}$ | $\text{BIC}$ | KS $D$ | KS $p$-value | Rank |
|:---|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Weibull** | 2 | $c = 0.9126, \; \beta = 55.57 \text{ h}$ | $-12,685.10$ | **25,374.20** | **0.00** | **25,385.85** | 0.0823 | $3.28 \times 10^{-15}$ | **1** |
| **Gamma** | 2 | $\alpha = 0.9171, \; \beta = 63.62 \text{ h}$ | $-12,700.19$ | 25,404.38 | +30.18 | 25,416.03 | **0.0771** | $2.07 \times 10^{-13}$ | **2** |
| **Exponential** | 1 | $\lambda = 0.01714 \text{ h}^{-1} \; (\mu = 58.35 \text{ h})$ | $-12,706.50$ | 25,414.99 | +40.80 | 25,420.82 | 0.0845 | $5.15 \times 10^{-16}$ | 3 |
| **Log-Normal** | 2 | $\mu = 3.4304, \; \sigma = 1.3622$ | $-12,937.27$ | 25,878.54 | +504.34 | 25,890.19 | 0.1149 | $2.62 \times 10^{-29}$ | 4 |

#### Mathematical Proof of Non-Exponentiality
1. **Likelihood Ratio Test ($\text{Weibull vs. Exponential}$):**
   $$\Lambda_{\text{LRT}} = 2 \left( \ln L_{\text{Weibull}} - \ln L_{\text{Exponential}} \right) = 2 \left(-12,685.10 - (-12,706.50)\right) = 42.80$$
   Under $H_0: c = 1$ (Exponential), the test statistic $\Lambda_{\text{LRT}} \sim \chi^2_1$.
   Since $\chi^2_{1, 0.999} = 10.83 \ll 42.80$, we reject the memoryless Exponential hypothesis at $p < 10^{-9}$.
2. **Sub-Exponential Shape:**
   The estimated shape parameter $c = 0.9126 < 1.0$ proves that ICU discharge hazard decreases over time: patients who survive the initial 48 hours have significantly longer remaining stays than newly admitted patients.

---

### 5. Pre-ICU ED Boarding Duration ($B_{\text{ED}\to\text{ICU}}$) Distribution Fitting

$$\text{Sample Size: } n = 1,253 \text{ boarded ED patients } (x > 0)$$

| Distribution | $k$ | Fitted Parameters ($\hat{\boldsymbol{\theta}}$) | $\ln L$ | $\text{AIC}$ | $\Delta\text{AIC}$ | $\text{BIC}$ | KS $D$ | KS $p$-value | Rank |
|:---|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Weibull** | 2 | $c = 0.8316, \; \beta = 3.92 \text{ h}$ | $-3,069.97$ | **6,143.93** | **0.00** | **6,154.20** | **0.0881** | $6.59 \times 10^{-9}$ | **1** |
| **Log-Normal** | 2 | $\mu = 0.7726, \; \sigma = 1.3082$ | $-3,082.55$ | 6,169.11 | +25.17 | 6,179.37 | 0.1134 | $1.70 \times 10^{-14}$ | **2** |
| **Gamma** | 2 | $\alpha = 0.8185, \; \beta = 5.45 \text{ h}$ | $-3,109.05$ | 6,222.09 | +78.16 | 6,232.36 | 0.0973 | $8.77 \times 10^{-11}$ | 3 |
| **Exponential** | 1 | $\lambda = 0.2241 \text{ h}^{-1} \; (\mu = 4.46 \text{ h})$ | $-3,126.81$ | 6,255.62 | +111.69 | 6,260.75 | 0.1000 | $2.31 \times 10^{-11}$ | 4 |

#### Mathematical Proof of Heavy-Tailed Boarding Delays
1. **Severe Model Mismatch of Exponential Service:**
   $$\Delta\text{AIC} = 111.69 \implies \text{Evidence Ratio } \frac{\mathbb{P}(\text{Exponential})}{\mathbb{P}(\text{Weibull})} = e^{-\Delta\text{AIC} / 2} = e^{-55.85} \approx 5.58 \times 10^{-25}$$
   The memoryless assumption is completely untenable for modeling ED boarding delays.
2. **Weibull Dispersion ($c = 0.8316 \ll 1.0$):**
   The pronounced decrease in hazard $h(t) \propto t^{-0.1684}$ reflects clinical reality: the longer a patient is blocked waiting for an ICU bed due to downstream saturation, the higher the probability that downstream beds are locked by other long-stay patients, exacerbating delays.

---

## Part 3: Verified Artifacts & Code Deliverables

All deliverables have been tested and generated in the project workspace:

### Codebase
* [`01_eda_and_preprocessing.py`](file:///d:/study/college/5th%20semester/ASM/courseProject/01_eda_and_preprocessing.py): End-to-end data ingestion, normalization, and cohort summary generation.
* [`02_nhpp_arrival_and_service_fitting.py`](file:///d:/study/college/5th%20semester/ASM/courseProject/02_nhpp_arrival_and_service_fitting.py): MLE parametric distribution fitting, AIC/BIC model selection, and NHPP rate construction.

### Generated Publication Figures (`output_plots/`)
* [`01_eda_distributions.png`](file:///d:/study/college/5th%20semester/ASM/courseProject/output_plots/01_eda_distributions.png): 4-panel exploratory visualization of empirical LOS, boarding delays, inflow sources, and CMS abandonment.
* [`02_nhpp_diurnal_arrival_rates.png`](file:///d:/study/college/5th%20semester/ASM/courseProject/output_plots/02_nhpp_diurnal_arrival_rates.png): Dual-axis diurnal probability $p_h$ curve and scaled arrival intensities $\lambda(h)$ by volume tier.
* [`02_icu_los_fitting_diagnostics.png`](file:///d:/study/college/5th%20semester/ASM/courseProject/output_plots/02_icu_los_fitting_diagnostics.png): 6-panel diagnostic sheet (Histogram + fitted PDFs, empirical vs. theoretical CDFs, and 4-way Q-Q plots for ICU LOS).
* [`02_ed_boarding_fitting_diagnostics.png`](file:///d:/study/college/5th%20semester/ASM/courseProject/output_plots/02_ed_boarding_fitting_diagnostics.png): 6-panel diagnostic sheet for Pre-ICU ED Boarding Delays.

### Generated Calibration Tables (`output_tables/`)
* [`01_overall_cohort_overview.csv`](file:///d:/study/college/5th%20semester/ASM/courseProject/output_tables/01_overall_cohort_overview.csv)
* [`01_cms_volume_tiers_breakdown.csv`](file:///d:/study/college/5th%20semester/ASM/courseProject/output_tables/01_cms_volume_tiers_breakdown.csv)
* [`01_cms_facilities_benchmarks.csv`](file:///d:/study/college/5th%20semester/ASM/courseProject/output_tables/01_cms_facilities_benchmarks.csv)
* [`02_nhpp_diurnal_hourly_rates.csv`](file:///d:/study/college/5th%20semester/ASM/courseProject/output_tables/02_nhpp_diurnal_hourly_rates.csv)
* [`02_icu_los_distribution_fitting.csv`](file:///d:/study/college/5th%20semester/ASM/courseProject/output_tables/02_icu_los_distribution_fitting.csv)
* [`02_ed_boarding_distribution_fitting.csv`](file:///d:/study/college/5th%20semester/ASM/courseProject/output_tables/02_ed_boarding_distribution_fitting.csv)
