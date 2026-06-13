# Ternary Critical — Critical Phenomena in Ternary Ising Models

**Ternary Critical** simulates critical phenomena in ternary Ising models — lattice systems where each site carries a spin in {-1, 0, +1}. The zero state acts as a **topological insulator**: it neither aligns with nor opposes its neighbors, creating a third phase between ordered and disordered. The crate provides Monte Carlo simulation, phase transition detection, and critical exponent estimation.

## Why It Matters

The classical Ising model (spins ±1) is the paradigmatic model of phase transitions in statistical mechanics. Adding a third spin state (0) fundamentally changes the critical behavior: the model develops a **tricritical point** where first-order and second-order phase transitions meet. This is directly relevant to ternary neural networks: at the critical temperature, information propagation is maximized (long-range correlations), which means that critical-state ternary networks learn faster than frozen or chaotic ones. Understanding where this transition occurs enables tuning ternary network initialization to the critical point for optimal training dynamics.

## How It Works

### Ternary Ising Hamiltonian

The energy of a configuration is:

```
E = -J · Σ sᵢ · sⱼ   (sum over nearest neighbors)
```

where sᵢ ∈ {-1, 0, +1}. The zero spin contributes zero energy regardless of neighbors — it's an energy-neutral "insulator" that breaks interaction chains.

### Local Energy

`local_energy(x, y) = -s(x,y) · Σ s(neighbors)`. For a 2D lattice with 4-connected neighbors, this is O(1) per site.

### Total Energy

Summed over all sites, counting right and down neighbors only to avoid double-counting: O(N) for N sites.

### Monte Carlo Simulation

Uses the Metropolis-Hastings algorithm: at each step, pick a random site, propose a random new spin, accept with probability:

```
P(accept) = min(1, exp(-ΔE / T))
```

At critical temperature Tc, the system exhibits power-law correlations: ⟨sᵢ·sⱼ⟩ ~ 1/r^η where η is a critical exponent. For the classical 2D Ising model Tc ≈ 2.269 J/kB; for the ternary model, Tc is shifted due to the diluting effect of zero spins.

### Phase Detection

The crate distinguishes three phases:
- **Ordered** (T << Tc): Most spins aligned (+1 or -1), low energy
- **Critical** (T ≈ Tc): Scale-free correlations, maximum susceptibility
- **Disordered** (T >> Tc): Random spins, high energy, zero magnetization

## Quick Start

```rust
use ternary_critical::TernaryIsing;

let mut model = TernaryIsing::new(20, 20);
model.ordered(); // Start from ordered state

// Monte Carlo sweeps
for _ in 0..1000 {
    // Metropolis step: flip random sites
    let x = (rand::random::<usize>()) % 20;
    let y = (rand::random::<usize>()) % 20;
    let old_e = model.local_energy(x, y);
    model.set(x, y, -model.get(x, y)); // flip
    let new_e = model.local_energy(x, y);
    // Accept/reject based on ΔE
}

println!("Total energy: {}", model.total_energy());
```

```bash
cargo add ternary-critical
```

## API

| Type / Function | Description |
|---|---|
| `TernaryIsing` | 2D lattice with `{width, height, spins, temperature}` |
| `local_energy(x, y)` | Site energy (O(1)) |
| `total_energy()` | System energy (O(N)) |
| `ordered()` | Initialize to all +1 |
| `critical_seed()` | Initialize to mixed -1/0/+1 |

## Architecture Notes

Critical phenomena theory guides **SuperInstance** system tuning: fleet behavior is optimized at the critical point between order (static, γ-dominated) and chaos (random, η-dominated). The γ + η = C conservation law is the fleet-level analog of energy conservation in the Ising model. See [Architecture](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md).

## References

- Ising, Ernst. "Beitrag zur Theorie des Ferromagnetismus," *Z. Physik*, 31, 1925 — original Ising model.
- Baxter, Rodney J. *Exactly Solved Models in Statistical Mechanics*, Academic Press, 1982 — tricritical points.
- Salakhutdinov, Ruslan & Hinton, Geoffrey. "Deep Boltzmann Machines," *AISTATS*, 2009 — Ising-like models for deep learning.

## License

MIT
