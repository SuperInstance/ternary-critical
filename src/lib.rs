//! # ternary-critical
//!
//! Critical phenomena in ternary Ising models: a `no_std`/`alloc` crate that
//! simulates a two-dimensional lattice whose sites carry a spin in
//! `{-1, 0, +1}` and estimates the associated phase-transition observables
//! (energy, magnetization, susceptibility, Binder cumulant, critical
//! temperature).
//!
//! ## When to use this
//!
//! Use `ternary-critical` when you want a small, dependency-free, fully
//! reproducible (deterministic) model of a ternary-valued Ising lattice — for
//! teaching, prototyping ternary/quantized neural-network dynamics, or as a
//! building block in a larger `no_std` fleet. The spin set is deliberately
//! quantized to three states and every observable is reported as a ternary
//! `i8`, so this crate is *not* a high-precision physics simulator; reach for a
//! floating-point Monte Carlo toolkit when you need sub-quantization accuracy.

#![forbid(unsafe_code)]
#![no_std]

extern crate alloc;
use alloc::{vec, vec::Vec};

/// A two-dimensional ternary Ising lattice.
///
/// Each cell holds a spin in `{-1, 0, +1}` stored in row-major order. The
/// `0` state is an energy-neutral "insulator": it contributes nothing to the
/// bond energy regardless of its neighbors, so it breaks interaction chains
/// without adding energy.
#[derive(Debug, Clone)]
pub struct TernaryIsing {
    /// Lattice width (number of columns).
    pub width: usize,
    /// Lattice height (number of rows).
    pub height: usize,
    /// Flat row-major spin buffer; each entry is clamped to `{-1, 0, +1}`.
    pub spins: Vec<i8>,
    /// Ternary temperature knob: `-1` = cold, `0` = critical, `+1` = hot.
    /// Controls the acceptance rule used by [`mc_sweep`](TernaryIsing::mc_sweep).
    pub temperature: i8,
}

impl TernaryIsing {
    /// Create an `width × height` lattice with every spin initialized to the
    /// neutral `0` state and `temperature = 0` (critical).
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            spins: vec![0; width * height],
            temperature: 0,
        }
    }

    /// Read the spin at column `x`, row `y` (row-major indexing).
    pub fn get(&self, x: usize, y: usize) -> i8 {
        self.spins[y * self.width + x]
    }

    /// Set the spin at column `x`, row `y`, clamping the value into
    /// `{-1, 0, +1}` so the lattice invariant can never be violated.
    pub fn set(&mut self, x: usize, y: usize, v: i8) {
        self.spins[y * self.width + x] = v.clamp(-1, 1);
    }

    /// Initialize all spins to +1 (ordered state)
    pub fn ordered(&mut self) {
        for s in &mut self.spins {
            *s = 1;
        }
    }

    /// Initialize with a "critical" seed: mix of all three states
    pub fn critical_seed(&mut self) {
        let n = self.width * self.height;
        for i in 0..n {
            self.spins[i] = match i % 3 {
                0 => -1,
                1 => 0,
                _ => 1,
            };
        }
    }

    /// Local energy at site (x,y): -sum of spin * neighbor_spins
    pub fn local_energy(&self, x: usize, y: usize) -> i8 {
        let s = self.get(x, y);
        let mut sum = 0i8;
        if x > 0 {
            sum += self.get(x - 1, y);
        }
        if x + 1 < self.width {
            sum += self.get(x + 1, y);
        }
        if y > 0 {
            sum += self.get(x, y - 1);
        }
        if y + 1 < self.height {
            sum += self.get(x, y + 1);
        }
        -(s * sum)
    }

    /// Total energy
    pub fn total_energy(&self) -> i32 {
        let mut e = 0i32;
        for y in 0..self.height {
            for x in 0..self.width {
                let s = self.get(x, y) as i32;
                if x + 1 < self.width {
                    e -= s * self.get(x + 1, y) as i32;
                }
                if y + 1 < self.height {
                    e -= s * self.get(x, y + 1) as i32;
                }
            }
        }
        e
    }

    /// Magnetization, quantized to a ternary value in `{-1, 0, +1}`.
    ///
    /// Computed as the mean spin `(Σ sᵢ) / N` and then mapped to the nearest
    /// ternary state with a ±1/3 threshold: a net majority beyond one third of
    /// the sites rounds to `±1`, otherwise `0`. The arithmetic is performed in
    /// `i64` so that `sum * 3` cannot overflow even for very large lattices
    /// (the previous `i32` intermediate overflowed once the lattice exceeded
    /// ~715 million sites).
    pub fn magnetization(&self) -> i8 {
        let n = self.spins.len() as i64;
        if n == 0 {
            return 0;
        }
        let sum: i64 = self.spins.iter().map(|&s| s as i64).sum();
        ((sum * 3 / n).clamp(-1, 1)) as i8
    }

    /// One Monte Carlo sweep: try to flip each spin
    /// Simplified Metropolis: at temperature 0 (cold), only accept energy-lowering flips
    /// At temperature 1 (hot), accept all flips
    /// At temperature -1 (very cold), only accept strict energy decreases
    pub fn mc_sweep(&mut self) {
        for y in 0..self.height {
            for x in 0..self.width {
                let current = self.get(x, y);
                let e_current = self.local_energy(x, y);

                // Try all possible flips
                let candidates = if current == 1 {
                    vec![-1, 0]
                } else if current == -1 {
                    vec![0, 1]
                } else {
                    vec![-1, 1]
                };

                for new_spin in candidates {
                    self.set(x, y, new_spin);
                    let e_new = self.local_energy(x, y);
                    let de = e_new - e_current;

                    let accept = match self.temperature {
                        -1 => de < 0, // very cold: only energy decreases
                        0 => de <= 0, // critical: accept non-increasing
                        1 => true,    // hot: accept everything
                        _ => de <= 0,
                    };

                    if accept {
                        break; // keep the flip
                    } else {
                        self.set(x, y, current); // revert
                    }
                }
            }
        }
    }

    /// Run N Monte Carlo sweeps
    pub fn run(&mut self, sweeps: usize) -> Vec<(i8, i32)> {
        let mut history = vec![];
        for _ in 0..sweeps {
            self.mc_sweep();
            history.push((self.magnetization(), self.total_energy()));
        }
        history
    }

    /// Susceptibility: variance of magnetization
    pub fn susceptibility(history: &[(i8, i32)]) -> i8 {
        if history.is_empty() {
            return 0;
        }
        let n = history.len() as i32;
        let mean_m: i32 = history.iter().map(|(m, _)| *m as i32).sum::<i32>() / n;
        let var: i32 = history
            .iter()
            .map(|(m, _)| (*m as i32 - mean_m).pow(2))
            .sum::<i32>()
            / n;
        var.clamp(-1, 1) as i8
    }
}

/// Find the critical temperature: sweep temperature and find peak susceptibility
pub fn find_critical_temperature(width: usize, height: usize, sweeps: usize) -> i8 {
    let mut max_suscept = -1i8;
    let mut critical_t = 0i8;

    for temp in -1..=1 {
        let mut model = TernaryIsing::new(width, height);
        model.critical_seed();
        model.temperature = temp;
        let history = model.run(sweeps);
        let chi = TernaryIsing::susceptibility(&history);
        if chi > max_suscept {
            max_suscept = chi;
            critical_t = temp;
        }
    }

    critical_t
}

/// Binder cumulant: `U4 = 1 - <m⁴>/(3<m²>²)`.
///
/// At a critical point this ratio is (approximately) universal. Because the
/// crate is integer/ternary-valued the result is quantized to `{-1, 0, +1}`;
/// for a history whose magnetization is never zero this yields `1` (the
/// ternary rounding of the ideal `2/3`), and `0` when `<m²> = 0`.
pub fn binder_cumulant(history: &[(i8, i32)]) -> i8 {
    if history.len() < 2 {
        return 0;
    }
    let n = history.len() as i32;
    let m_vals: Vec<i32> = history.iter().map(|(m, _)| *m as i32).collect();
    let m2: i32 = m_vals.iter().map(|m| m * m).sum::<i32>() / n;
    let m4: i32 = m_vals.iter().map(|m| m * m * m * m).sum::<i32>() / n;
    if m2 == 0 {
        return 0;
    }
    // Factor of 3 belongs in the DENOMINATOR: U4 = 1 - <m^4> / (3 * <m^2>^2).
    // (A previous version put it in the numerator, yielding 1 - 3<m^4>/<m^2>^2,
    // which is mathematically wrong and produced spurious -1 values.)
    let u4 = 1 - m4 / (3 * m2 * m2);
    u4.clamp(-1, 1) as i8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ising_new() {
        let m = TernaryIsing::new(4, 4);
        assert_eq!(m.get(0, 0), 0);
        // all spins default to the neutral 0 state
        assert_eq!(m.width, 4);
        assert_eq!(m.height, 4);
        assert_eq!(m.spins.len(), 16);
    }

    #[test]
    fn test_ising_ordered() {
        let mut m = TernaryIsing::new(4, 4);
        m.ordered();
        // all +1 -> mean spin = +1 -> ternary magnetization +1
        assert_eq!(m.magnetization(), 1);
    }

    #[test]
    fn test_ising_critical_seed() {
        let mut m = TernaryIsing::new(3, 3);
        m.critical_seed();
        // 3x3 seed is the repeating pattern -1,0,1 three times: sum = 0
        assert_eq!(m.magnetization(), 0);
    }

    #[test]
    fn test_local_energy_aligned() {
        let mut m = TernaryIsing::new(4, 4);
        m.ordered();
        // interior site (1,1): s=+1, four +1 neighbors -> -1*(1*4) = -4
        assert_eq!(m.local_energy(1, 1), -4);
    }

    #[test]
    fn test_local_energy_corner() {
        let mut m = TernaryIsing::new(4, 4);
        m.ordered();
        // corner (0,0): only 2 neighbors (right + down) -> -1*(1*2) = -2.
        // This exercises the x>0 / y>0 boundary guards in local_energy.
        assert_eq!(m.local_energy(0, 0), -2);
    }

    #[test]
    fn test_local_energy_misaligned() {
        let mut m = TernaryIsing::new(4, 4);
        m.ordered();
        m.set(1, 1, -1); // anti-aligned with its four +1 neighbors
        assert_eq!(m.local_energy(1, 1), 4); // -(-1 * 4) = +4
    }

    #[test]
    fn test_local_energy_insulator() {
        let mut m = TernaryIsing::new(4, 4);
        m.ordered();
        m.set(1, 1, 0); // the neutral "insulator" state
        assert_eq!(m.local_energy(1, 1), 0); // 0 * anything = 0
    }

    #[test]
    fn test_total_energy_ordered() {
        let mut m = TernaryIsing::new(4, 4);
        m.ordered();
        // 4x4 has 4*3 horizontal + 4*3 vertical = 24 bonds; each -1 -> -24.
        let e = m.total_energy();
        assert_eq!(e, -24);
    }

    #[test]
    fn test_set_clamps_out_of_range() {
        let mut m = TernaryIsing::new(2, 2);
        m.set(0, 0, 5); // above +1 clamps to +1
        m.set(1, 0, -9); // below -1 clamps to -1
        assert_eq!(m.get(0, 0), 1);
        assert_eq!(m.get(1, 0), -1);
    }

    #[test]
    fn test_empty_lattice_guards() {
        // 0x0 lattice: must not divide by zero (n == 0 early returns).
        let m = TernaryIsing::new(0, 0);
        assert_eq!(m.magnetization(), 0);
        assert_eq!(m.total_energy(), 0);
    }

    #[test]
    fn test_mc_sweep_keeps_valid_spins() {
        let mut m = TernaryIsing::new(4, 4);
        m.critical_seed();
        m.temperature = 0;
        m.mc_sweep();
        for &s in &m.spins {
            assert!((-1..=1).contains(&s));
        }
    }

    #[test]
    fn test_run_history_length() {
        let mut m = TernaryIsing::new(4, 4);
        m.critical_seed();
        m.temperature = 1;
        let history = m.run(5);
        assert_eq!(history.len(), 5);
    }

    #[test]
    fn test_cold_energy_non_increasing() {
        // At temperature -1 only strict energy decreases are accepted, so the
        // total energy can never increase across sweeps (greedy descent).
        let mut m = TernaryIsing::new(4, 4);
        m.critical_seed();
        m.temperature = -1;
        let e_before = m.total_energy();
        m.run(20);
        let e_after = m.total_energy();
        assert!(e_after <= e_before);
    }

    #[test]
    fn test_dynamics_are_deterministic() {
        // The sweep has no RNG, so identical seeds + temperatures must produce
        // bit-identical final configurations.
        let mut a = TernaryIsing::new(4, 4);
        a.critical_seed();
        a.temperature = 1;
        a.run(10);

        let mut b = TernaryIsing::new(4, 4);
        b.critical_seed();
        b.temperature = 1;
        b.run(10);

        assert_eq!(a.spins, b.spins);
    }

    #[test]
    fn test_susceptibility_zero() {
        // constant magnetization -> zero variance -> 0
        let history = vec![(1i8, -10i32), (1, -10), (1, -10)];
        assert_eq!(TernaryIsing::susceptibility(&history), 0);
    }

    /// Susceptibility for a fluctuating history: m = [+1, +1, -1, -1].
    /// mean = 0, variance = (1+1+1+1)/4 = 1 -> ternary 1.
    #[test]
    fn test_susceptibility_nonzero_exact() {
        let history = vec![(1i8, -10i32), (1, -10), (-1, -10), (-1, -10)];
        assert_eq!(TernaryIsing::susceptibility(&history), 1);
    }

    #[test]
    fn test_susceptibility_empty() {
        assert_eq!(TernaryIsing::susceptibility(&[]), 0);
    }

    /// Exact check of the Binder cumulant formula.
    ///
    /// For a history of constant magnetization m = +1 the ideal value is
    /// U4 = 1 - <m^4>/(3<m^2>^2) = 1 - 1/(3*1) = 2/3, which quantizes to the
    /// ternary value +1. The old code computed `1 - 3*<m^4>/<m^2>^2` = -2
    /// (clamped to -1); this test pins the corrected value.
    #[test]
    fn test_binder_cumulant_formula_exact() {
        let history = vec![(1i8, -10i32), (1, -10), (1, -10)];
        assert_eq!(binder_cumulant(&history), 1);
    }

    #[test]
    fn test_binder_cumulant_too_short() {
        // fewer than 2 samples -> defined as 0
        assert_eq!(binder_cumulant(&[(1i8, -10i32)]), 0);
    }

    #[test]
    fn test_find_critical_temperature_deterministic() {
        // The Monte Carlo sweep is fully deterministic (no RNG), so for fixed
        // inputs this returns a constant. It scans the three ternary
        // temperatures and reports the one with peak susceptibility.
        let tc = find_critical_temperature(4, 4, 5);
        assert_eq!(tc, -1);
        // a second call must reproduce the same value (no hidden state)
        assert_eq!(find_critical_temperature(4, 4, 5), tc);
    }
}
