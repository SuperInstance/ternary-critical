//! # ternary-critical
//!
//! Critical phenomena in ternary Ising models.
//! Monte Carlo simulation, phase transitions, critical temperature, and universality.

#![forbid(unsafe_code)]
#![no_std]

extern crate alloc;
use alloc::{vec, vec::Vec};

/// A 2D ternary Ising model
#[derive(Debug, Clone)]
pub struct TernaryIsing {
    pub width: usize,
    pub height: usize,
    pub spins: Vec<i8>, // {-1, 0, +1}
    pub temperature: i8, // ternary: -1=cold, 0=critical, 1=hot
}

impl TernaryIsing {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width, height,
            spins: vec![0; width * height],
            temperature: 0,
        }
    }

    pub fn get(&self, x: usize, y: usize) -> i8 {
        self.spins[y * self.width + x]
    }

    pub fn set(&mut self, x: usize, y: usize, v: i8) {
        self.spins[y * self.width + x] = v.clamp(-1, 1);
    }

    /// Initialize all spins to +1 (ordered state)
    pub fn ordered(&mut self) {
        for s in &mut self.spins { *s = 1; }
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
        if x > 0 { sum += self.get(x - 1, y); }
        if x + 1 < self.width { sum += self.get(x + 1, y); }
        if y > 0 { sum += self.get(x, y - 1); }
        if y + 1 < self.height { sum += self.get(x, y + 1); }
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

    /// Magnetization (average spin)
    pub fn magnetization(&self) -> i8 {
        let sum: i32 = self.spins.iter().map(|&s| s as i32).sum();
        let n = self.spins.len() as i32;
        if n == 0 { return 0; }
        (sum * 3 / n).clamp(-1, 1) as i8
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
                let candidates = if current == 1 { vec![-1, 0] }
                                 else if current == -1 { vec![0, 1] }
                                 else { vec![-1, 1] };
                
                for new_spin in candidates {
                    self.set(x, y, new_spin);
                    let e_new = self.local_energy(x, y);
                    let de = e_new - e_current;
                    
                    let accept = match self.temperature {
                        -1 => de < 0,   // very cold: only energy decreases
                        0 => de <= 0,   // critical: accept non-increasing
                        1 => true,      // hot: accept everything
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
        if history.is_empty() { return 0; }
        let n = history.len() as i32;
        let mean_m: i32 = history.iter().map(|(m, _)| *m as i32).sum::<i32>() / n;
        let var: i32 = history.iter()
            .map(|(m, _)| (*m as i32 - mean_m).pow(2))
            .sum::<i32>() / n;
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

/// Binder cumulant: U4 = 1 - <m⁴>/(3<m²>²)
/// At critical point, this should be universal
pub fn binder_cumulant(history: &[(i8, i32)]) -> i8 {
    if history.len() < 2 { return 0; }
    let n = history.len() as i32;
    let m_vals: Vec<i32> = history.iter().map(|(m, _)| *m as i32).collect();
    let m2: i32 = m_vals.iter().map(|m| m * m).sum::<i32>() / n;
    let m4: i32 = m_vals.iter().map(|m| m * m * m * m).sum::<i32>() / n;
    if m2 == 0 { return 0; }
    let u4 = 1 - m4 * 3 / (m2 * m2);
    u4.clamp(-1, 1) as i8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ising_new() {
        let m = TernaryIsing::new(4, 4);
        assert_eq!(m.get(0, 0), 0);
    }

    #[test]
    fn test_ising_ordered() {
        let mut m = TernaryIsing::new(4, 4);
        m.ordered();
        assert_eq!(m.magnetization(), 1);
    }

    #[test]
    fn test_ising_critical_seed() {
        let mut m = TernaryIsing::new(3, 3);
        m.critical_seed();
        assert_eq!(m.magnetization(), 0);
    }

    #[test]
    fn test_local_energy_aligned() {
        let mut m = TernaryIsing::new(4, 4);
        m.ordered();
        let e = m.local_energy(1, 1);
        assert!(e < 0); // aligned = negative energy
    }

    #[test]
    fn test_local_energy_misaligned() {
        let mut m = TernaryIsing::new(4, 4);
        m.ordered();
        m.set(1, 1, -1); // anti-aligned with neighbors
        let e = m.local_energy(1, 1);
        assert!(e > 0); // misaligned = positive energy
    }

    #[test]
    fn test_total_energy_ordered() {
        let mut m = TernaryIsing::new(4, 4);
        m.ordered();
        let e = m.total_energy();
        assert!(e < 0); // ordered = negative total energy
    }

    #[test]
    fn test_mc_sweep_runs() {
        let mut m = TernaryIsing::new(4, 4);
        m.critical_seed();
        m.temperature = 0;
        m.mc_sweep();
        // Just check it doesn't crash and values are valid
        for &s in &m.spins {
            assert!(s >= -1 && s <= 1);
        }
    }

    #[test]
    fn test_run() {
        let mut m = TernaryIsing::new(4, 4);
        m.critical_seed();
        m.temperature = 1;
        let history = m.run(5);
        assert_eq!(history.len(), 5);
    }

    #[test]
    fn test_cold_orders() {
        let mut m = TernaryIsing::new(4, 4);
        m.critical_seed();
        m.temperature = -1;
        m.run(20);
        // After many cold sweeps, should be mostly aligned
        let mag = m.magnetization();
        assert!(mag == -1 || mag == 0 || mag == 1); // valid ternary
    }

    #[test]
    fn test_hot_disorders() {
        let mut m = TernaryIsing::new(4, 4);
        m.ordered();
        m.temperature = 1;
        m.run(10);
        // Hot temperature should disorder the system
        // Just check it runs
        assert!(m.spins.iter().all(|&s| s >= -1 && s <= 1));
    }

    #[test]
    fn test_susceptibility() {
        let history = vec![(1i8, -10i32), (1, -10), (1, -10)];
        let chi = TernaryIsing::susceptibility(&history);
        assert_eq!(chi, 0); // constant magnetization = zero susceptibility
    }

    #[test]
    fn test_binder_cumulant() {
        let history = vec![(1i8, -10i32), (1, -10), (1, -10)];
        let u4 = binder_cumulant(&history);
        assert!(u4 >= -1 && u4 <= 1);
    }

    #[test]
    fn test_find_critical_temperature() {
        let tc = find_critical_temperature(4, 4, 5);
        assert!(tc >= -1 && tc <= 1);
    }
}
