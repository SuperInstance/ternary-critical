//! Runnable Quick Start for `ternary-critical`.
//!
//! Build & run: `cargo run --example quickstart`

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
    println!(
        "Susceptibility:       {}",
        TernaryIsing::susceptibility(&history)
    );

    // Scan the three ternary temperatures and report the one with peak
    // susceptibility as the estimated critical temperature.
    println!(
        "Critical temperature: {}",
        find_critical_temperature(8, 8, 20)
    );
}
