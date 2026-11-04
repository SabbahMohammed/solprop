# SEP 44 — Trace absorbing gases and neutral dissociation channels for photochemistry coupling

| | |
|---|---|
| **SEP** | 44 |
| **Title** | Trace absorbing gases and neutral dissociation channels for photochemistry coupling |
| **Author** | Mohammed Sabbah |
| **Drafting agent** | Claude Opus 5.5 (2026-11-03; revised 2026-11-04) |
| **Status** | Draft |
| **Review status** | Not reviewed |
| **Reviewers** | — |
| **Type** | Feature |
| **Created** | 2026-11-03 |
| **Resolution** | — |
| **Requires** | [SEP 3](archive/sep-0003-optmat-materials.md): optmat materials and the Raman mixture rules (D13, D16); [SEP 27](archive/sep-0027-held-plasma-current.md): the held field-grid plasma current |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | Phases 1 (gas-only loss) and 2 (neutral channels): implemented ahead of acceptance (Part III) on local branches, merged with `main` at kernel schema 45 on the local branch `feature/air-ozone` (kernel schema 46) |
| **Manual pages** | — |

## Part I — Proposal

### 1. Abstract

A pulse propagating through a gas-filled hollow guide can drive photochemistry in the
gas, and that chemistry changes the gas the next pulse meets: ultrashort pulses in air
make ozone and nitrogen oxides, which absorb in the ultraviolet where the pulse's
dispersive wave lies. Coupling solprop to a slow chemistry model needs four things that
solprop either lacks or makes awkward: (A) a guide loss that keeps the **gas's
absorption** and drops the **wall's**; (B) a field-driven loss channel whose events
release **no charge**, for strong-field dissociation into neutral fragments; (C) a
documented route for **trace absorbers** whose data are the caller's own; and (D) a
clear statement of what the **Raman response** of the major constituents can and cannot
do beside such traces and a composition that drifts along the guide. This SEP derives
the physics, proposes (A) as a loss setting `loss="gas"` and (B) as a per-channel
electrons-per-event weight of zero or one in the plasma channel (a new admitted value of
a field the channel already has, with a kernel-schema bump so that a stale compiled build
is refused), documents (C), and states (D) and its options without implementing them.
Success is judged by closed forms and invariants (an exact attenuation identity, energy
balance per event, the ionising limit recovered exactly), parity of `numpy_ref` and the
native CPU build, and unchanged results for every run that does not ask for either.

### 2. Motivation and scope

#### The use case

An air-filled capillary or antiresonant fibre at a few bar, driven at a kilohertz
repetition rate by microjoule pulses of tens of femtoseconds, self-compresses the pulse
and emits a resonant dispersive wave in the deep ultraviolet. Each pulse also ionises
and dissociates some of the gas. Over seconds the photochemistry of air builds up ozone
and nitrogen oxides (NO, NO₂, NO₃, N₂O₅) with densities that vary along the guide,
because the field that drives them does. Ozone's Hartley band peaks near 255 nm with a
cross-section of about $1.1\times10^{-21}\,\mathrm{m^2}$ [7, 8], so a trace of
$10^{23}\,\mathrm{m^{-3}}$, under a part in a thousand at 6 bar, attenuates the
ultraviolet by about 100 m⁻¹: enough to switch the dispersive wave off over a few centimetres.

The coupled model alternates two solvers on two time scales. A chemistry model (rate
equations on a grid along the guide, outside solprop) integrates over many pulses and
hands solprop the number density of every species along $z$. solprop propagates one
pulse through that composition and hands back, along $z$, the energy each species
absorbed per unit frequency (the photolysis source) and the fraction of each population
that a field-driven channel removed (ionisation of O₂ and N₂, strong-field
dissociation of O₃). The composition changes slowly, so solprop re-solves with an
updated mixture many times; the solve must therefore be cheap to update
(`Simulation.with_mixture`), and its outputs must be exactly the quantities the
chemistry consumes.

#### In scope

- **(A) Gas-only guide loss** for the Marcatili capillary and the Vincetti tube lattice:
  the absorption of the fill kept, the guide's own loss removed. The capillary model
  is often a stand-in for a lower-loss guide of the same bore, or the wall loss is
  accounted for elsewhere, while the trace absorber's loss is what couples to the
  chemistry and must be resolved.
- **(B) Neutral loss channels**: a field-driven channel that depletes a population at a
  rate $W(|E|)$ and charges the field an energy $U_b$ per event, with no free-electron
  current, on the field grid, the mode-averaged and mode-resolved capillary routes and
  the Tier 1 backends.
- **(C) Trace absorbers as caller-supplied species**: the existing
  `Species`/`TabulatedSusceptibility` route, documented with its one pitfall.
- **(D) Raman of the major constituents beside trace species**: the problem and the
  options; the route that works today, verified.

#### Not in scope

- The chemistry itself, its rate data and its time stepping.
- Curated optical data for O₃, NO₂, NO₃ or N₂O₅ in optmat. SEP 3 left O₃ out until
  absorbing gas indices and non-ideal density are handled properly (D13); this SEP
  does not change that. A caller supplies the tables (C).
- Fragment dynamics: branching ratios, fragment kinetic energy, recombination,
  alignment and multiple events per molecule. A neutral channel is one event per
  molecule, like single ionisation.
- Heating of the gas by the absorbed energy, and any change of density on the time scale
  of one pulse.
- Finite-density mixture optics ([SEP 11](sep-0011-finite-density-gas-index.md)).
- A general user response hook. Responses stay compiled within the kernel contract
  (`docs/dev/architecture.md`, §6.6).

### 3. Context

- **Mixtures.** `Mixture.of(Component(species, density_path, label), ...)` carries
  labelled populations with their own density paths along $z$; the linear operator sums
  $\chi = \sum_s \rho_s(z)\gamma_s(\omega)$ inside the mode's index, and more than two
  independent populations are tabulated along $z$
  ([gas mixtures](../../guide/media/mixtures.md)). `Simulation.with_mixture` rebuilds a
  run for a new composition, keeping its grid, rates and unchanged Raman kernels.
- **Absorption outputs.** `GasFillMixin.absorption_by_species` attributes the material
  loss $\rho_s\omega\operatorname{Im}\gamma_s/c$ to each population, and
  `processing.absorbed_spectral_energy` turns it into an energy loss density along $z$
  ([saved material analysis](../../guide/media/mixtures.md)). A callable `loss`, which
  replaces the total loss, has no species attribution and is refused there.
- **Capillary loss.** `loss=True` is the model's loss, `False` none, a float scales it
  and a callable replaces it ([capillary model](../../guide/waveguides/capillary-model.md)).
  `False` removes the gas's absorption with the wall's, so before this SEP an absorbing
  trace could be resolved only beside the full Marcatili leakage: 82 m⁻¹ at 800 nm in a
  15 µm silica capillary, against which the pulse is gone within centimetres.
- **Plasma.** `PlasmaResponse` carries the free-electron current and the ionisation loss
  current of Geissler *et al.* [2], one channel per labelled population
  ([field plasma](../../guide/media/field-plasma.md)); the plasma is single ionisation and
  "no neutral fragment yield, branching ratio, recombination or ozone production is
  modelled" ([ionisation](../../guide/media/ionisation.md)). The kernel contract's
  `PlasmaChannel` leaf carries a `charge_mass_ratio` and an `electrons_per_event` per
  channel, the second fixed at one: "Dissociation and multiple electrons per event are
  unsupported" ([kernel contract](../kernel-contract.md), §5). ADR-027 leaves
  dissociation channels to "a later value of `electrons_per_event`, not a schema change"
  ([decisions](../decisions.md)). The backends read each channel's `charge_mass_ratio`
  in their multi-channel paths (native packs it as a column of its channel rows), and a
  problem-level copy in their one-channel specialisations. The envelope plasma channel
  carries `electrons_per_event` too, also fixed at one.
- **Raman.** A mixture of built-in gases takes optmat's one response for its composition,
  with partner-resolved broadening; a population optmat does not resolve is refused
  beside a Raman-active molecule (SEP 3, D16), and so is a composition whose mole
  fractions vary along $z$ by more than `COMPOSITION_TOLERANCE`, 1 %
  ([Raman data](../../guide/media/raman-data.md)). A caller's species carrying its own
  `MolecularRaman` takes the per-component route of
  [several responding species](../../guide/media/raman.md).
- **Related proposals.** [SEP 10](sep-0010-low-rank-raman.md) represents Raman kernels
  along density paths; [SEP 11](sep-0011-finite-density-gas-index.md) the finite-density
  index of a mixture.
- **Issues.** The issue tracker was not searched for this draft; that, and the tracking
  issue SEP 0 asks for, are left to the author (Part III).

### 4. Theory

SI units throughout; the optics convention of `docs/dev/conventions.md` §2 for the
physics, with $\operatorname{Im}n_{\rm eff} > 0$ loss and the power attenuation
$\alpha = 2(\omega/c)\operatorname{Im}n_{\rm eff}$ (§5).

#### 4.1 A trace absorber's susceptibility

A dilute population of number density $\rho_s$ with refractivity per particle
$(n_s-1)/\rho_s$ and intensity absorption cross-section $\sigma_s(\omega)$ has the
per-particle susceptibility

$$\gamma_s(\omega) = 2\,\frac{n_s(\omega)-1}{\rho_s} + i\,\frac{c\,\sigma_s(\omega)}{\omega},
\qquad \chi(\omega, z) = \sum_s \rho_s(z)\,\gamma_s(\omega),$$

since $n = \sqrt{1+\chi}\approx 1+\chi/2$ gives the intensity attenuation
$2(\omega/c)\operatorname{Im}n = \sum_s\rho_s\sigma_s$ (Beer–Lambert) in the dilute
limit. This is what `TabulatedSusceptibility` forms. The real and imaginary parts are
related by causality; a caller's table is consistent only if its real part is the
Kramers–Kronig transform of its absorption over a wide enough band, which the caller
must ensure (a constant "atom-equivalent" refractivity beside a measured band is a
modelling approximation, not a causal law). The linear operator evaluates $\gamma_s$ on
the grid's whole active band, $\omega_{\min}/2 < \omega < 1.1\,\omega_{\max}$
(conventions §3), not only between the requested wavelength limits.

#### 4.2 Gas-only loss in a capillary

Marcatili and Schmeltzer's hollow-guide index [1], in the full form solprop uses, is

$$n_{\rm eff}^2 = \varepsilon_{\rm co} - (u_{nm}s)^2\,(1 - i v_n s)^2,
\qquad s = \frac{c}{\omega a},\quad \varepsilon_{\rm co} = 1 + \chi,$$

with $u_{nm}$ the Bessel zero of the mode and $v_n$ the cladding term of its family. For
a transparent cladding $v_n$ is real and

$$(u_{nm}s)^2(1 - iv_ns)^2 = (u_{nm}s)^2 - 2i\,(u_{nm}s)^2 v_n s - (u_{nm}s)^2 (v_n s)^2 .$$

The second term is the wall's leakage, first order in $s$ beyond the waveguide term;
the third is a real shift of second order, beyond the accuracy of the expansion, which
is first order in $1/(ka)$. The **ideal wall**, $v_n = 0$, is the limit in which the
transverse field $J_{n-1}(u_{nm}r/a)$ vanishes at $r = a$ exactly: nothing penetrates
the wall, nothing leaks, and

$$n_{\rm eff}^2 = 1 + \chi - (u_{nm}s)^2 .$$

The waveguide term $(u_{nm}s)^2$ is real, so with $n_{\rm eff} = n' + in''$,
$\operatorname{Im}n_{\rm eff}^2 = 2n'n'' = \operatorname{Im}\chi$ **exactly**, and the
mode's power attenuation is

$$\alpha = \frac{2\omega}{c}\,n'' = \frac{1}{n'}\sum_s \rho_s\sigma_s ,$$

the gas's own absorption, with the waveguide entering only through $n' =
\operatorname{Re}n_{\rm eff}$. The reduced (leading-order) model,
$n_{\rm eff} = 1 + \chi/2 - (u_{nm}s)^2/2$, gives $\alpha = \sum_s\rho_s\sigma_s$. The
anomalous waveguide dispersion, which the soliton dynamics and the dispersive wave depend
on, is kept: only the wall goes.

Dropping the whole of $v_n$ changes the real index by about $(u_{nm}s)^2(v_ns)^2/2$:
$3.3\times10^{-8}$ at 800 nm and $5\times10^{-7}$ at 1.6 µm for HE₁₁ in a 15 µm silica
capillary. Keeping it would mean keeping the real part of the lossy square root and
replacing only its imaginary part, which does not separate cleanly once
$\operatorname{Im}\chi \ne 0$; the shift is beyond the model's order, and the ideal wall
is a definite physical limit, so (A) drops it.

#### 4.3 Gas-only loss in a tube lattice

The Vincetti fits [4–6] in solprop's form give

$$n_{\rm eff} = n_g - \frac{W}{n_g} + \Delta n + i\,\frac{c\,\alpha_{\rm CL}}{2\omega},$$

with $n_g = \sqrt{1+\chi}$ the gas index, $W = (u_{nm}c/\omega R_{\rm eff})^2/2$ the
waveguide decrement, $\Delta n$ the real resonance correction and $\alpha_{\rm CL}$ the
fitted confinement loss ([antiresonant fibres](../../guide/waveguides/antiresonant.md)).
The confinement term is purely the guide's; the gas's absorption enters through the
complex $n_g$ in the first two terms. Gas-only loss drops the confinement term and
nothing else; then $\operatorname{Im}n_{\rm eff} = \operatorname{Im}n_g\,(1 +
W/|n_g|^2) + \dots$, the gas's absorption with a waveguide correction of order $W$.

#### 4.4 A neutral loss channel

Let a population of initial density $\rho$ undergo a field-driven event at the rate
$W(|\mathbf{E}|)$ per surviving molecule, each event costing the field an energy $U_b$
and releasing $q$ free electrons. The fraction that has undergone the event obeys
$\dot f = W(1-f)$, so

$$f(t) = 1 - (1 - f_0)\exp\left(-\int_{-\infty}^{t} W\,\mathrm{d}t'\right),$$

with $f_0$ the fraction already gone before the pulse. Poynting's theorem in the
medium, $\partial_t u + \nabla\cdot\mathbf{S} = -\mathbf{J}\cdot\mathbf{E}$, charges the
field the work the medium absorbs. The events absorb the power density $\rho U_b \dot f$;
a current parallel to the field that carries exactly that work is

$$\mathbf{J}_{\rm loss} = \rho\,U_b\,W(1-f)\,\frac{\mathbf{E}}{|\mathbf{E}|^2},
\qquad \mathbf{J}_{\rm loss}\cdot\mathbf{E} = \rho\,U_b\,\dot f ,$$

the loss current of Geissler *et al.* [2] with the ionisation potential replaced by
$U_b$. The freed electrons, $n_e = q\rho f$, are accelerated by the field,

$$\partial_t\mathbf{J}_{\rm free} = \frac{e^2}{m_e}\,q\rho f\,\mathbf{E},$$

the Drude current that lowers the index. Single ionisation is $q = 1$, $U_b = I_p$.
Strong-field dissociation of a molecule into **neutral** fragments [10], ozone into
O₂ + O for example, is $q = 0$: the population is depleted and the field pays $U_b$ per
event, but no current builds up and the index is untouched. The polarisation entering
the propagation equation [3] is $\mathbf{P} = \int (\mathbf{J}_{\rm free} +
\mathbf{J}_{\rm loss})\,\mathrm{d}t$, as for the plasma, so a neutral channel is the
plasma channel with $q = 0$, and every property of that channel's numerics — the running
integrals ([SEP 18](archive/sep-0018-band-exact-plasma-integrals.md)), the held post-pulse current ([SEP 27](archive/sep-0027-held-plasma-current.md)), the mode-averaged area-mean
field — carries over unchanged; with $q = 0$ there is no free current to hold.

What $W$ and $U_b$ are is the caller's model, declared with the channel. A tunnelling
form with an effective barrier (an ADK-like rate [9, 11] at a fitted energy) is one
choice; $U_b$ is the energy the field loses per event, which need not equal the
barrier the rate is evaluated at (the bond energy plus the fragments' kinetic energy,
for a dissociation). The approximations are those of the plasma channel: an
instantaneous rate on the real field, no orientation dependence, one event per
molecule, no recombination, and no change of the population during the pulse other
than the event itself. A channel with $q \ge 2$ would need sequential charge states, a
different population model, and is not proposed.

#### 4.5 Raman of the major constituents beside traces

optmat's air response is one response for one composition, broadened by curated
partner pairs; a molecule optmat does not resolve cannot be a partner, and a
composition that varies along $z$ cannot be one response (SEP 3, D16 and §4.4 there).
Both hold in the use case: the traces are caller species, and the oxygen consumed into
ozone moves its mole fraction by more than 1 % where the chemistry is strongest. A trace
at $10^{-3}$ of the total changes the N₂ and O₂ linewidths by at most its fraction times
the ratio of its broadening coefficient to air's, a per-mille effect, and its own Raman
response is negligible at that density; a drift of the O₂ fraction changes the O₂
amplitude in proportion and the widths through the total density only. The physics is
therefore benign; the obstacle is that one response cannot represent it (§6.4).

### 5. Proposed interface

```text
# (A) the fill's absorption, without the wall's
MarcatiliMode(15e-6, mixture=fill, loss="gas")
VincettiMode(15e-6, mixture=fill, lattice=TubeLattice(...), loss="gas")
Simulation.from_capillary(15e-6, 0.225, fill, ..., loss="gas")   # and prop_capillary

# (B) a field-driven channel that frees no electron
PlasmaResponse(ozone, rate, electrons_per_event=0.0, species="O3")
Simulation.from_capillary(..., plasma={"N2": ..., "O2": ..., "O3": NeutralChannel(rate)})

# (C) a trace absorber of the caller's own
ozone = Species("O3", TabulatedSusceptibility(omega, refractivity, cross_section,
                wavelength_range, citation=...), gamma3=..., ionisation=IonisationData(U_b))
fill = Mixture.of(*air_components, Component(ozone, density_path, "O3"))
```

- `loss="gas"` is a fourth value of the existing `LossSpec`. It composes with mixtures
  whose populations vary along $z$, with `mode_average=True` and the mode-resolved
  quadrature, and with `absorption_by_species`, which attributes the whole remaining
  loss as it does for `True`. It is refused by name by the modes that do not implement
  it (rectangular, planar, tabulated) and beside a coherent wall, which is its channels'
  loss model.
- A neutral channel reports the event fraction through `species_ionisation`, the energy
  paid through `species_ionisation_energy`, and zero through the electron-density
  statistics; it is refused where the route has no field-grid plasma. The final names
  and the statistics' wording are Phase 2's.

### 6. Implementation strategies

#### 6.1 (A) Gas-only loss

The index is built in the physics layer (`solprop.modes`) and translated on the host
into the linear operator; no backend sees anything new, the kernel contract is
unchanged, and every backend runs it with no code of its own.

- **A1, an ideal wall in the mode (recommended).** The capillary forms its index with
  $v_n = 0$ when `loss="gas"`, in the full and reduced models, their analytic frequency
  derivatives and `dneff_dz`; the tube lattice omits its confinement term. The no-gain
  clamp and the evanescent floor apply as for `True`. Cost: none; it removes work.
- **A2, subtract the wall's imaginary part.** Keep the real part of the lossy index and
  replace its imaginary part by the gas's. It keeps the wall's real shift but has no
  clean definition once $\operatorname{Im}\chi \neq 0$ (the square root mixes the two),
  and its derivative needs a second evaluation.
- **A3, a callable replacing the loss by $\sum_s\rho_s\sigma_s$.** A callable has no
  $z$ dependence, so it cannot follow a composition along $z$, and it loses the species
  attribution.

#### 6.2 (B) Neutral loss channels

- **B1, a per-channel electrons-per-event weight in $\{0, 1\}$ (recommended).** The
  physics object (`PlasmaResponse`, with a `NeutralChannel` declaration for the keyword
  routes) carries the weight; translation writes it into the `electrons_per_event` field
  each `PlasmaChannel` leaf already has, which admits zero beside one. The backends fold
  it into the free-current constant they already read, $q\,e^2/m_e$, per channel and in
  the problem-level copy of the one-channel specialisations; no field, buffer, kernel
  signature or kernel arithmetic changes, and the native build's validation admits a
  zero constant. `KERNEL_SCHEMA_VERSION` is bumped all the same, in the same commit and
  with a release-note fragment: a compiled build of the previous schema would run a
  neutral channel as an ionisation, and the bump makes it refuse the problem instead
  (ADR-027's "not a schema change" is amended accordingly). $q = 1$ must reproduce the
  present arithmetic exactly (a product with one), so no existing result moves. `mlx`
  (Tier 3) refuses a neutral channel by name. `numba_cpu`, which SEP 43 removes, needs
  nothing beyond packing the same constant on a tree that still has it.
- **B2, a separate loss-only response type.** A new leaf with its own kernels: more code
  in every backend for what is one coefficient of an existing leaf.
- **B3, emulate with an ionising channel.** Works today with no change, but adds a
  spurious Drude current from $\rho f$ electrons (about $2\times10^{-5}$ of the O₂
  plasma's in a typical air case) and reports electrons that do not exist: physically
  wrong, and the error grows with the trace.
- **B4, a user callback in the loop.** Breaks the compiled-response contract and the
  backends' performance; out of scope (§2).

Cost of B1: none per sample. The weight is folded into a constant the kernels already
multiply by, so a neutral channel still evaluates its free-current integrand, as zero;
skipping it for $q = 0$ is an optimisation left for later, not needed for correctness.

#### 6.3 (C) Trace absorbers

No code: a caller's `Species` with a `TabulatedSusceptibility` (and, if a channel acts
on it, `IonisationData` with the energy per event) already runs on every backend,
including the native $z$ table of more than two independent populations. The manual
gains the frequency-support rule of §4.1, because a table cut at the requested
wavelength limits is refused only when the operator is built.

#### 6.4 (D) Raman beside traces and a drifting composition

- **D1, per-component supplied responses (works today).** The major constituents are
  supplied as renamed species carrying their own `MolecularRaman`; each amplitude takes
  its own density and each width the total density. It runs with any number of caller
  species and any drift along $z$ (verified on `numpy_ref` and native, Part II), at the
  cost of the curated partner-resolved broadening, and with constants the caller must
  supply and cite.
- **D2, inert partners.** Let optmat's response accept unresolved trace populations as
  spectators that broaden with air's coefficients (or not at all), admitted below a
  declared fraction with the error bounded by fraction times coefficient ratio.
- **D3, a composition along $z$.** Tabulate the response's amplitudes and decay rates
  along the composition path, as [SEP 10](sep-0010-low-rank-raman.md) does for density.
- **D4, the status quo.** Refuse, and point to D1 or `raman=False`.

D1 is the recommendation for now; D2 and D3 need optmat's agreement on what an inert
partner is, and belong in their own proposal.

#### 6.5 Placement and dependencies

(A) is physics-layer only; (B) touches the physics, translation and backend layers and
`native/` (its validation only); (C) and (D) are documentation. No new dependency.

### 7. Verification and validation

- **(A)** The ideal-wall index against its closed form at 40 digits, to 64 eps relative
  (a few tens of roundings in the rationalised square root). The identity
  $2n'n'' = \operatorname{Im}\chi$, so $\alpha n' = \sum_s\rho_s\sigma_s$, to 64 eps.
  A transparent fill exactly lossless. The dropped real shift against
  $(u_{nm}s)^2(v_ns)^2/2$ to 1 %, the size of the next order. The reduced model's real
  part equal to `loss=False`'s exactly. Frequency derivatives against Cauchy's contour
  formula (the capillary module's $10^{-11}$) and a three-step difference ladder;
  `dneff_dz` against a fourth-order difference on a taper with a rising absorber
  ($10^{-8}$). The tube lattice: real parts equal to `True`'s exactly for a transparent
  fill, the gas's attenuation to 1 %. Linear runs with an absorber density linear in
  $z$: every bin attenuated by $\exp(-\int\alpha\,\mathrm{d}z)$ of the mode's own
  $\alpha$, to $10^{-10}$ in the exponent (the operator table's $10^{-12}$ of its
  scale), mode-averaged and resolved, capillary and lattice, on each CPU backend. A
  weakly nonlinear run matched against `numpy_ref` within the march rounding budget.
- **(B)** Constant rate: $f = 1 - (1-f_0)e^{-Wt}$. Energy balance: the field's energy
  loss equals $\rho U_b \Delta f$ integrated over the cross-section, to the running
  integrals' certified error. $q = 0$: zero free current and electron density,
  structurally; no phase shift of a weak probe. $q = 1$: bit-for-bit the present
  `PlasmaResponse`. Parity of `numpy_ref` and the native CPU build on ten steps,
  mode-averaged and resolved, with a mixture of ionising and neutral channels.
- **(C)** A table narrower than the active band refused when the operator is built,
  naming the band.
- **(D)** The D1 route, a $z$-varying composition with a caller trace species, on
  `numpy_ref` and native.

### 8. Documentation

- (A): `docs/guide/waveguides/capillary-model.md` ("Gas absorption without the wall"),
  `antiresonant.md`, `docs/guide/media/mixtures.md`, the API page's loss vocabulary,
  the capabilities list of `docs/guide/workflows/luna.md`.
- (B): `docs/guide/media/ionisation.md` (neutral loss channels), `field-plasma.md`, the
  statistics catalogue, the kernel contract's `PlasmaChannel` leaf.
- (C), (D): `mixtures.md` (the active-band rule) and `raman-data.md` (the D1 route
  beside a refusal).
- A worked example of an air-filled capillary with an ozone column, once (B) lands, is
  the natural gallery addition.

### 9. Compatibility and impact

No default changes. (A) adds a value to `LossSpec`; the signatures of the hollow modes
change in their annotation only. (B) bumps `KERNEL_SCHEMA_VERSION`; saved runs keep
their layout (a neutral channel's record gains `electrons_per_event`; a single-ionisation
response leaves it out, so its record and hash are unchanged, and an absent value reads
as one),
and every existing channel has $q = 1$, so no validated result moves. Luna.jl
has neither feature; both are listed among the capabilities beyond it.

### 10. Alternatives considered

- **`loss=False` plus a callable for the gas**: the callable cannot follow the
  composition along $z$, and `False` removes the attribution (§6.1).
- **A "material" spelling of (A)**: the wall is a material too (its absorption enters
  $v_n$); "gas" names what is kept.
- **Dissociation as an ionising channel with a tiny electron yield**: B3 above.
- **A dissociation model inside solprop** (branching, fragment energies): a chemistry
  model's business, not the propagation's (§2).

### 11. Phases

1. **Gas-only loss (A)** on `numpy_ref` (the only place it lives), with its tests and
   manual pages; the other backends receive the same operator.
2. **Neutral channels (B)**: `numpy_ref` with tests and manual, then the native CPU
   build; the schema bump; `mlx` refusing by name.
3. **Documentation of (C) and (D)**, and an issue for D2/D3.

### 12. Open questions

1. Should `loss="gas"` combine with a measured wall loss, an additive $\alpha_{\rm
   wall}(\omega)$ beside the gas's (for instance `loss=("gas", curve)`)? Deferred to an
   issue.
2. Should the rectangular and planar guides (their wall-pair correction zeroed) and the
   evacuated table (`TabulatedHollowMode`, its imaginary part dropped) take
   `loss="gas"`? They refuse it now.
3. The free-space coupling model of SEP 31 reads the capillary's sampled cladding term
   for its profile; under `loss="gas"` it still sees the real wall. Is that the intended
   reading?
4. Neutral channels on the envelope grid, whose channel already carries
   `electrons_per_event`: Phase 2 refuses them by name. They are simple in principle
   (keep the loss moment, drop the Drude drive); wanted, and in which phase?
5. D2 or D3 for Raman, and with optmat or in solprop? A separate proposal.
6. Should the total `peak_ionisation_rate` count ionising channels only? It takes the
   largest event rate of any channel, neutral ones included (Phase 2, D4); making it
   ionising-only needs a change to native's one-channel statistic. For review.
7. Should neutral events get statistic names of their own (for instance
   `species_event_fraction`), or are the existing names, documented for a neutral
   channel, enough? For review.
8. Should `plasma_boundary_level`
   ([SEP 18](archive/sep-0018-band-exact-plasma-integrals.md)) weigh channels by their
   electrons per event? It counts a neutral channel's $|fE|$
   as if it carried free current. An issue.
9. Should `NeutralChannel` be registered for recipes (SEP 41)? Additive; an issue.
10. Is the release tag "0.2.0" right for the compatibility entry that reads an absent
    `electrons_per_event` as one? It follows SEP 41's entries. For review.

### 13. References

1. E. A. J. Marcatili and R. A. Schmeltzer, "Hollow metallic and dielectric waveguides
   for long distance optical transmission and lasers," *Bell Syst. Tech. J.* **43**,
   1783–1809 (1964). doi:10.1002/j.1538-7305.1964.tb04108.x
2. M. Geissler, G. Tempea, A. Scrinzi, M. Schnürer, F. Krausz and T. Brabec, "Light
   propagation in field-ionizing media: extreme nonlinear optics," *Phys. Rev. Lett.*
   **83**, 2930 (1999). doi:10.1103/PhysRevLett.83.2930
3. M. Kolesik and J. V. Moloney, "Nonlinear optical pulse propagation simulation: from
   Maxwell's to unidirectional equations," *Phys. Rev. E* **70**, 036604 (2004).
   doi:10.1103/PhysRevE.70.036604
4. L. Vincetti, "Empirical formulas for calculating loss in hollow core tube lattice
   fibers," *Opt. Express* **24**, 10313 (2016). doi:10.1364/OE.24.010313
5. L. Vincetti and L. Rosa, "A simple analytical model for confinement loss estimation
   in hollow-core Tube Lattice Fibers," *Opt. Express* **27**, 5230 (2019).
   doi:10.1364/OE.27.005230
6. L. Rosa, F. Melli and L. Vincetti, "Analytical formulas for dispersion and effective
   area in hollow-core tube lattice fibers," *Fibers* **9**, 58 (2021).
   doi:10.3390/fib9100058
7. A. Gorshelev, A. Serdyuchenko, M. Weber, W. Chehade and J. P. Burrows, "High
   spectral resolution ozone absorption cross-sections — Part 1: Measurements, data
   analysis and comparison with previous measurements around 293 K," *Atmos. Meas.
   Tech.* **7**, 609 (2014). doi:10.5194/amt-7-609-2014
8. A. Serdyuchenko, V. Gorshelev, M. Weber, W. Chehade and J. P. Burrows, "High
   spectral resolution ozone absorption cross-sections — Part 2: Temperature
   dependence," *Atmos. Meas. Tech.* **7**, 625 (2014). doi:10.5194/amt-7-625-2014
9. M. V. Ammosov, N. B. Delone and V. P. Krainov, "Tunnel ionization of complex atoms
   and of atomic ions in an alternating electromagnetic field," *Sov. Phys. JETP*
   **64**, 1191 (1986).
10. J. H. Posthumus, "The dynamics of small molecules in intense laser fields," *Rep.
    Prog. Phys.* **67**, 623 (2004). doi:10.1088/0034-4885/67/5/R01
11. X. M. Tong, Z. X. Zhao and C. D. Lin, "Theory of molecular tunneling ionization,"
    *Phys. Rev. A* **66**, 033402 (2002). doi:10.1103/PhysRevA.66.033402

## Part II — Implementation record

### Phase 1 report — gas-only loss

- **Scope:** Phase 1, 2026-11-03, Opus 5.5 (coding agent) for the author, on the local
  branch `feature/air-ozone-loss`; not yet pushed or reviewed.
- **Delivered:** `src/solprop/modes/fill.py` (the `"gas"` value of `LossSpec`, its
  validation and refusal by modes that do not implement it, the species attribution,
  the loss application); `src/solprop/modes/capillary.py` (`_cladding_terms`: the
  ideal wall in the full and reduced models, their derivatives and `dneff_dz`);
  `src/solprop/modes/antiresonant.py` (the confinement term dropped);
  `src/solprop/modes/tabulated.py` (refusal); the `loss` docstrings of the run
  interfaces, collections and design tools; `src/solprop/recipe/metadata.py` (the
  `"gas"` choice). Tests: `tests/modes/test_gas_loss.py` (22, fast) and
  `tests/driver/test_gas_loss_runs.py` (16 runs, medium, on the CPU backends the
  registry offers, and a check that the list follows the registry); the two signature
  witnesses `tests/data/fill_before_extraction.json` and `rectangular_before_slab.json`
  gained the literal. Pages: those of §8 (A), and `changelog.d/gas-only-capillary-loss.added.md`.
- **Deviations from the proposal:** none.
- **Decisions:** the reduced model honours `model="reduced"` under `"gas"` (drops its
  loss term) rather than switching to the full form; the no-gain clamp applies to
  `"gas"` as to `True`, though it cannot bind for a non-negative cross-section.
- **Verification:** the tests of §7 (A) pass: 22 and 16; the changed and affected
  modules (`tests/modes`, `tests/recipe`, the tube-lattice, tabulated, display,
  capability and design-tool tests) pass, 1974 tests; ruff, basedpyright (no errors on
  the changed files), numpydoc and codespell on the changed files. Measured: the
  ideal-wall index matches its 40-digit closed form within 64 eps; the linear runs'
  exponent error is $1.6\times10^{-15}$ against an exponent of one; native and
  `numpy_ref` agree to $2.3\times10^{-16}$ (mode-averaged) and $2.1\times10^{-16}$
  (resolved) on the weakly nonlinear run, against a march budget of $1.3\times10^{-12}$. Not run: the
  full gate, the slow and coverage lanes, the docs build.
- **Sources checked:** the Marcatili form against solprop's own manual and module; no
  new constant is transcribed.
- **Limitations:** rectangular, planar and tabulated modes and the coherent wall refuse
  `"gas"` (Q2); the free-space coupling profile is unaffected by it (Q3).
- **Open questions:** Q1–Q3 of §12.
- **Handoff:** the option is complete for the capillary and the tube lattice; Phase 2
  is independent of it.

### Phase 2 report — neutral loss channels

- **Scope:** Phase 2, 2026-11-03, Opus 5.5 (coding agent) for the author, on the local
  branch `feature/air-ozone`, in two commits (the channels at kernel schema 42, then the
  schema literals left in a native test and two doctests); not yet pushed or reviewed.
  Both phases' branches start from `main` before SEP 43, where `numba_cpu` was still a
  backend.
- **Delivered:** physics: `src/solprop/nonlinear/ionisation.py`
  (`ELECTRONS_PER_EVENT = (0.0, 1.0)`, its check, the frozen `NeutralChannel(rate)`),
  `src/solprop/nonlinear/responses.py` (`PlasmaResponse.electrons_per_event`,
  keyword-only, default one, validated; the `neutral` property; the free current
  weighed by it), `src/solprop/nonlinear/mixtures.py` (`responses_for` takes a
  `NeutralChannel` per label; the envelope grid refuses it by name); the run
  interfaces `src/solprop/driver/simulation.py`, `interface.py`, `mixtures.py` and
  `summary.py` (`plasma=` accepts `NeutralChannel`; the summary names a neutral loss
  channel) and the session export in `src/solprop/sol.py`. Translation:
  `src/solprop/translate/responses.py` (`PlasmaChannel.electrons_per_event` admits zero
  beside one; `free_current_coefficient`, `charge_mass_ratio` times
  `electrons_per_event` on the channel and `charge_mass_ratio` on
  `PlasmaCoefficients`), `src/solprop/translate/envelope_plasma.py` (the same property,
  electrons still fixed at one), `src/solprop/translate/problem.py`
  (`KERNEL_SCHEMA_VERSION` 41 to 42). Statistics: `src/solprop/stats.py` and
  `src/solprop/backends/populations.py` (electrons are events times the weight;
  fractions and energies count events). Backends: `numpy_ref` (its one-channel scan
  reads `free_current_coefficient`), `numba_cpu` and native
  (`native/src/solprop_native/backend.py`) pack `free_current_coefficient` into the
  constants their kernels already read; `native/src/general.cpp` and
  `native/include/solprop_native/core.hpp` admit a zero constant (`>= 0`, was `> 0`);
  `src/solprop/backends/mlx_gpu/backend.py` refuses a neutral channel by name.
  Records: `src/solprop/output/compatibility.py` (an absent `electrons_per_event`
  reads as one, so earlier records resume). Tests: `tests/nonlinear/test_neutral_channel.py`
  (15) and `tests/backends/test_neutral_channels.py` (19), new; additions to
  `native/tests/test_mixtures.py`, `tests/backends/test_mlx.py`,
  `tests/mixture_cases.py`, `tests/translate/test_mixture_validation.py`,
  `tests/test_sol.py` and `tests/driver/test_summary.py`; the schema literals of the
  driver, problem, Kerr-term, mixture-channel and native API tests. Pages:
  `docs/guide/media/ionisation.md` (a section on neutral loss channels: derivation,
  interface, statistics, limits, example), `docs/guide/media/mixtures.md`, the
  statistics catalogue, `docs/dev/kernel-contract.md`, `docs/dev/open-questions.md`,
  `docs/dev/decisions.md`, and `changelog.d/neutral-plasma-channels.added.md`.
- **Deviations from the proposal:** none against §6.2 as revised on 2026-11-04. The
  first draft expected B1 to change the stage bundles and the backend surface and to
  cost a multiply per channel and sample; the implementation showed neither necessary
  (D2 below), and Part I, still a draft, was corrected.
- **Decisions:**
  D1, the spelling: `PlasmaResponse.electrons_per_event`, zero or one, the name the
  kernel contract had reserved, and a frozen `NeutralChannel(rate)` that a `plasma=`
  selection takes per label where a rate goes, for instance
  `plasma={"N2": True, "O2": True, "O3": NeutralChannel(rate)}`. Rejected: an attribute
  on the rate law (the charge an event frees belongs to the channel, and every rate
  class would need it) and a wrapping rate (it would hide the `OptmatRate` attributes
  that the energy binding, the manifest and the peak-field report read). Two or more
  electrons need sequential charge states, not another weight, and stay an open
  question in `docs/dev/open-questions.md`.
  D2, the kernel crossing: the backends pack `free_current_coefficient` into the
  constant their kernels already read. For an ionisation the product with one is
  exact, so the packed numbers are bit-identical to before; for a neutral channel it is
  exactly zero. No kernel arithmetic, signature or buffer changed; native's change is
  its validation only.
  D3, a schema bump although no field moved: the kernel contract needs no bump to
  remove an admitted value, because no consumer can misread a problem that no longer
  carries it; adding one is the converse, since a compiled build of schema 41 would run
  a neutral channel as an ionisation. The bump makes `check_build()` refuse such a
  build. ADR-027's "not a schema change" is amended in place.
  D4, the statistics keep their names, documented for a neutral channel:
  `species_ionisation` is the fraction that underwent the event,
  `species_ionisation_energy` the energy the events paid, `species_peak_ionisation_rate`
  their peak rate, and `species_electron_density` exactly zero; `electron_density` and
  `on_axis_electron_density` count ionising channels only. The total
  `peak_ionisation_rate` keeps the largest event rate of any channel, neutral included
  (Q6).
  D5, run records: `electrons_per_event` is recorded for every `PlasmaResponse`; the
  compatibility table reads an absent value as one under the release tag "0.2.0" (Q10).
  D6, backends: `numpy_ref` and native implemented and in parity; `numba_cpu` by
  packing alone; `mlx` refuses by name (its field path already multiplies by
  `electrons_per_event`, but it is not qualified: mlx could not be installed on the
  implementing machine); the envelope grid refuses by name (Q4).
- **Verification:** Apple M5 Pro, macOS, Python 3.14.7, NumPy 2.5.2, Numba 0.67.0; the
  native build from the branch's tree with Clang 23.1.4 (Release, OpenMP),
  `SOLPROP_THREADS=1`. Fast checks clean: ruff, basedpyright, numpydoc and codespell on
  the changed files, `git diff --check`, `poe changelog-check`, clang-format on
  `native/`; the whole documentation built with `-W -n`. Tests: the two new modules,
  15 and 19 passed; nonlinear, translate, session, guide snippets, summary and output,
  3156 passed and 3 skipped (pre-existing skips); records, recipes, mixtures and driver,
  779 passed; the plasma parity set (medium), 299 passed; `native/tests`, 422 passed and
  11 skipped (platform); the package doctests, 1010 passed; further plasma and
  statistics modules, 359 passed and one deselected, a test that hangs identically on
  `main` on that machine (two OpenMP runtimes loaded: an environment fault, not this
  change). `tests/backends/test_mlx.py` skipped (no mlx). Analytic checks on
  `numpy_ref`: the energy identity, for an eighth-power rate that converts
  $5.7\times10^{-3}$ of the population, $(-\int E'P\,\mathrm{d}t/N)/(U\Delta f) - 1 =
  -9.9\times10^{-5}$, $-2.5\times10^{-5}$ and $-6.2\times10^{-6}$ at 8193, 16385 and
  32769 samples (second order, ratio 4.00; bound $2\times10^{-4}$); the neutral
  polarisation against $\rho\int U W (1-f)/E\,\mathrm{d}t$ written out independently,
  within $8n$ eps of its peak; ionising minus neutral against
  $\rho(e^2/m_e)\iint f E\,\mathrm{d}t\,\mathrm{d}t$, within $16n$ eps, the Drude part
  above $10^{-3}$ of the total so the check is not empty; a fully pre-reacted neutral
  population gives exactly zero polarisation; the fraction is the same for zero and one
  electrons; an explicit one is bitwise the default; 2.0, 0.5, −1 and NaN are refused,
  as are a `PlasmaChannel` with 2.0 and a neutral channel on the envelope grid. Parity
  on a synthetic two-population case (pre-ionisation 0.03 and 0.10; "mixture", an
  ionising and a neutral channel; "single", the neutral one alone on the one-channel
  path), bounds fixed before running ($2.92\times10^{-11}$ at the stage,
  $1.02\times10^{-10}$ and $8.15\times10^{-10}$ after one and eight steps): at most
  $3.7\times10^{-16}$ at the stage and $8.2\times10^{-18}$ after eight steps, on
  `numba_cpu` and native alike; every species statistic and both totals identical to
  `numpy_ref`; the neutral population's electron density a hard zero on every backend.
  An all-ionising twin departs from the neutral run by 6.5 and 30 in the deviation
  norm, above $10^3$ times the bound, so a backend that ignored the declaration could
  not pass. Existing behaviour, on an air mixture with N₂ and O₂ channels and an argon
  capillary, ten fixed steps on every CPU backend, `main` against the branch:
  `numpy_ref` fields and statistics bitwise identical; every plasma statistic bitwise
  identical on `numba_cpu` and native; `numba_cpu` fields within $5.9\times10^{-16}$
  relative (the two environments' Numba and NumPy versions differ, and each is bitwise
  repeatable), native fields within $5.8$–$8.7\times10^{-16}$, the size of native's own
  run-to-run spread ([SEP 25](archive/sep-0025-rounding-level-reproducibility.md)).
  An end-to-end air run (6 bar, a 15 µm mode-averaged capillary, 3 mm, 10 fs, 4 µJ) with
  a $10^{-3}$ trace given a neutral channel: `numpy_ref` and native agree on every
  population fraction to the digits printed and on the field to $1.0\times10^{-13}$
  (adaptive runs); the trace's electron density is zero, against
  $1.09\times10^{18}\,\mathrm{m^{-3}}$, $2.0\times10^{-5}$ of the total, when the same
  rate runs as an ionisation (B3). Not run: the full gate, the slow and coverage lanes,
  `poe check-native` (clang-tidy, sanitizers, wheel), the native strict type check,
  `reuse lint`, and a benchmark comparison (no compiled kernel, buffer or per-sample
  operation changed).
- **Sources checked:** solprop's own plasma response, translation, backend plasma paths
  and statistics, the kernel contract and ADR-027. No constant is transcribed. Not
  checked: any published value of an effective barrier or rate for a molecule's
  strong-field dissociation; those are the caller's, labelled as effective.
- **Limitations:** one irreversible event per molecule within a pulse; no fragments,
  branching, recombination or chemistry, and the composition between pulses is the
  caller's. Field grid only, the envelope refusing; `mlx` refuses, unqualified.
  `plasma_boundary_level` still counts a neutral channel's $|fE|$ as if it carried free
  current, on every backend alike, so a pre-reacted neutral population can raise it
  spuriously (Q8). The total `peak_ionisation_rate` includes neutral event rates (Q6).
  Recipes cannot name `NeutralChannel` (Q9). An `OptmatRate` holds its rate above its
  model's maximum field, so a run whose field exceeds it sees the held rate; a caller
  who needs the rate law beyond it supplies a rate of their own.
- **Open questions:** Q4 and Q6–Q10 of §12.
- **Handoff:** the channel is complete on `numpy_ref` and native. At a rebase onto
  `main` after SEP 43, `numba_cpu`'s packing drops out with the backend, and the schema
  number follows `main`'s: 44 there on 2026-11-04, so this change would become 45.

### Integration with `main`

- **Scope:** 2026-11-04, Opus 5.5 (coding agent) for the author, on the local branch
  `feature/air-ozone`: `main` (`f865adbb`, kernel schema 45, `numba_cpu` retired by
  SEP 43) merged into the Phase 2 branch, then the Phase 1 branch merged in; merges
  rather than a rebase, so the reviewed commits keep their hashes. Not yet pushed or
  reviewed.
- **Delivered:** the neutral channel at kernel schema 46 (the literals, the contract's
  version note, the native backend's schema note); `numba_cpu`'s packing of the
  channel dropped with the backend; `electrons_per_event` named in `PlasmaResponse`'s
  recipe table (SEP 41), which upstream's metadata tests require. Phase 1 merged
  without conflict. D5 revised: `electrons_per_event` is recorded only where it is not
  one (a dataclass field's `record_default=False`, beside `main`'s `record_none=False`),
  so a single-ionisation run keeps its record and configuration hash, as `main` keeps
  them for its own optional fields; `main`'s pinned record of a capillary call without
  TDSE (`tests/driver/test_tdse_entry.py`) caught the change of every plasma record.
- **Verification:** the new and affected modules on `numpy_ref` and native (the neutral
  channel, optmat rates, the mixture ladder, the driver, the problem, the summary, the
  modes and the recipes, the native contract, backend and mixture tests) pass on the
  merged tree, run serially; ruff and codespell are clean.

### Verification of (D) — supplied Raman beside a trace and a drifting composition

Not a phase; a check of the route §6.4 recommends, made while drafting. N₂ and O₂ were
supplied as renamed species carrying solprop's former registry constants (the test
inputs `tests/former_raman.py`, not curated data), O₂ depleted by 5 % along a 2 cm,
6 bar, 15 µm capillary, a synthetic absorber rising to $10^{-3}$ of the total, with
`loss="gas"`, Kerr and both Raman responses, no plasma, on a fixed 40-step schedule.
optmat's air response refuses both the drift (a mole fraction varying by 2 %) and the
caller species (D16), as documented. The supplied route translates and runs on
`numpy_ref` and native, mode-averaged and resolved. At 0.1 µJ native and `numpy_ref`
agree to $10^{-14}$; at 2.7 µJ, a strongly nonlinear run, to $1.6\times10^{-7}$ and
$3.9\times10^{-7}$ (rounding amplified by the nonlinear propagation; a fixed-order
march). The Raman response changes the output field by 2 % (0.1 µJ) and 62 % (2.7 µJ).

## Part III — History

- 2026-11-03: Drafted, with the Phase 1 implementation already made on a local branch
  at the author's request, ahead of acceptance; that is a departure from SEP 0 §6 to be
  resolved at review (accept the phase as made, or revise it). The tracking issue and
  the issue search of SEP 0 §6 have not been done. SEP 43 is taken by a proposal on
  another branch, so this one is numbered 44.
- 2026-11-04: Revised after an internal review of the local branches (not a review of
  the proposal under SEP 0 §6, so the header records no reviewer). §3 and §6.2 B1
  corrected to the code: the plasma leaf already carries `electrons_per_event`, the
  weight folds into constants the backends already read, and the schema bump serves
  only to refuse a stale build. `numba_cpu` dropped from the plan: SEP 43, final on
  `main` since 2026-11-03, removes it. The Phase 2 report appended, and its open
  questions added as Q6–Q10. On this date 44 was still free on `main`, whose kernel
  schema is 44 (this branch's is 41, Phase 2's 42); both branches need a rebase before
  a pull request. The **Author** field names the person who asked for the work, who is
  not a maintainer: SEP 0 §5 wants a maintainer there, to be settled before the SEP
  goes upstream.
- 2026-11-04: `main` merged into the Phase 2 branch and the Phase 1 branch merged in
  (Part II, "Integration with `main`"); the kernel schema is 46, `main`'s being 45. 44
  is still free among the SEPs on `main`.
