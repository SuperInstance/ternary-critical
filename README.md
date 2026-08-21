# Ternary Critical — Critical Phenomena in Ternary Ising Models

**Ternary Critical** simulates critical phenomena in ternary Ising models — lattice
systems where each site carries a spin in `{-1, 0, +1}`. The zero state acts as a
**topological insulator**: it neither aligns with nor opposes its neighbors, creating
a third phase between ordered and disordered. The crate provides energy/magnetization
measurement, a deterministic Monte-Carlo-style relaxation sweep, susceptibility and
Binder-cumulant estimation, and a critical-temperature search.

The crate is `#![no_std]` (with `alloc`) and has **zero dependencies**.

## Why It Matters

The classical Ising model (spins ±1) is the paradigmatic model of phase transitions in
statistical mechanics. Adding a third spin state (0) fundamentally changes the critical
behavior: the model develops a **tricritical point** where first-order and second-order
phase transitions meet. This is directly relevant to ternary neural networks: near the
critical temperature, information propagation is maximized (long-range correlations),
which means critical-state ternary networks can learn faster than frozen or chaotic
ones. Locating that transition helps tune ternary-network initialization toward the
critical point.

> **Scope note.** This crate is a small, dependency-free, *deterministic* model of a
> ternary-valued lattice intended for teaching, prototyping, and use as a `no_std`
> building block. Every observable is reported as a ternary `i8`, so it is **not** a
> high-precision physics simulator — reach for a floating-point Monte Carlo toolkit
> when you need sub-quantization accuracy.

## How It Works

### Ternary Ising Hamiltonian

The energy of a configuration is:

```
E = -J · Σ sᵢ · sⱼ   (sum over nearest-neighbor bonds; J = 1)
```

where `sᵢ ∈ {-1, 0, +1}`. The zero spin contributes zero energy regardless of
neighbors — it is an energy-neutral "insulator" that breaks interaction chains.

### Local Energy

`local_energy(x, y) = -s(x,y) · Σ s(neighbors)`. For a 2D lattice with 4-connected
neighbors this is O(1) per site (interior sites have four neighbors; edges and corners
fewer).

### Total Energy

`total_energy()` sums the bond energy over all sites, counting only the *right* and
*down* neighbor of each site to avoid double-counting: O(N) for N sites. For example an
8×8 fully-ordered (all `+1`) lattice has `8·7 + 8·7 = 112` bonds and energy `-112`.

### Relaxation Sweep (deterministic)

`mc_sweep()` is a **deterministic** Metropolis-style sweep — there is no random-number
generator, so results are fully reproducible run-to-run. It visits every site in
row-major order and, for each site, tries the candidate spin values different from the
current one. A candidate is accepted according to the ternary `temperature` knob:

| `temperature` | Acceptance rule                         |
|---------------|-----------------------------------------|
| `-1` (cold)   | accept only strict energy decreases (`ΔE < 0`) |
| `0` (critical)| accept non-increasing energy (`ΔE ≤ 0`) |
| `+1` (hot)    | accept every proposed flip              |

The first accepted candidate is kept; if none is accepted the spin is left unchanged.

### Observables

- **`magnetization()`** — the mean spin `(Σ sᵢ)/N`, quantized to `{-1, 0, +1}` with a
  ±1/3 threshold.
- **`susceptibility(history)`** — the variance of the magnetization over a recorded
  history, quantized to `{-1, 0, +1}`.
- **`binder_cumulant(history)`** — `U4 = 1 - ⟨m⁴⟩/(3·⟨m²⟩²)`, a dimensionless ratio
  that is approximately universal at a critical point.
- **`find_critical_temperature(w, h, sweeps)`** — scans the three ternary temperatures
  and reports the one with peak susceptibility.

## Quick Start

`examples/quickstart.rs` cools an 8×8 lattice from the mixed critical seed toward its
ground state and reports observables:

```rust
use ternary_critical::{find_critical_temperature, TernaryIsing};

fn main() {
    // An 8x8 lattice, started from the mixed critical seed (-1/0/+1 pattern).
    let mut model = TernaryIsing::new(8, 8);
    model.critical_seed();
    println!("Initial total energy: {}", model.total_energy());

    // Cool to the "very cold" ternary temperature and run 50 sweeps. The sweep
    // is fully deterministic (no RNG), so this is reproducible run-to-run.
    model.temperature = -1;
    let history = model.run(50);
    println!("Final total energy:   {}", model.total_energy());
    println!("Final magnetization:  {}", model.magnetization());

    // Fluctuation (susceptibility) of the magnetization over the history.
    println!("Susceptibility:       {}", TernaryIsing::susceptibility(&history));

    // Scan the three ternary temperatures and report the one with peak
    // susceptibility as the estimated critical temperature.
    println!("Critical temperature: {}", find_critical_temperature(8, 8, 20));
}
```

Run it:

```bash
cargo run --example quickstart
```

Output (deterministic):

```
Initial total energy: 38
Final total energy:   -112
Final magnetization:  -1
Susceptibility:       0
Critical temperature: -1
```

The disordered seed (energy `+38`) relaxes under cold sweeps to the aligned ground
state (energy `-112`, magnetization `-1`).

```bash
cargo add ternary-critical
```

## API

| Type / Function | Description |
|---|---|
| [`TernaryIsing`] | 2D lattice: `width`, `height`, `spins` (`{-1,0,+1}`), `temperature` (`-1/0/+1`) |
| `TernaryIsing::new(w, h)` | Create a `w×h` lattice of neutral `0` spins |
| `get(x, y)` / `set(x, y, v)` | Read / write a spin (`set` clamps into `{-1,0,+1}`) |
| `ordered()` | Initialize every spin to `+1` |
| `critical_seed()` | Initialize to the repeating `-1/0/+1` pattern |
| `local_energy(x, y)` | Site energy, O(1) |
| `total_energy()` | System energy (right+down bonds only), O(N) |
| `magnetization()` | Mean spin quantized to `{-1,0,+1}` |
| `mc_sweep()` | One deterministic relaxation sweep |
| `run(sweeps)` | Run `sweeps` sweeps, returning a `(magnetization, energy)` history |
| `susceptibility(history)` | Variance of magnetization, quantized to `{-1,0,+1}` |
| `binder_cumulant(history)` | Binder cumulant `U4 = 1 - ⟨m⁴⟩/(3⟨m²⟩²)` |
| `find_critical_temperature(w, h, sweeps)` | Ternary temperature with peak susceptibility |

[`TernaryIsing`]: https://docs.rs/ternary-critical/latest/ternary_critical/struct.TernaryIsing.html

## Architecture Notes

Critical phenomena theory guides **SuperInstance** system tuning: fleet behavior is
optimized at the critical point between order (static, γ-dominated) and chaos (random,
η-dominated). See [Architecture](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md).

## References

- Ising, Ernst. "Beitrag zur Theorie des Ferromagnetismus," *Z. Physik*, 31, 1925 — original Ising model.
- Baxter, Rodney J. *Exactly Solved Models in Statistical Mechanics*, Academic Press, 1982 — tricritical points.
- Binder, Kurt. "Finite Size Scaling Analysis of Ising Model Block Distribution Functions," *Z. Phys. B*, 43, 1981 — the Binder cumulant.

## License

MIT — see [LICENSE](LICENSE).
