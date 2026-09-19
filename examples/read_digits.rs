//! The flagship: a real fruit-fly brain reading handwritten digits.
//!
//! Downloads the connectome on first run (~1 MB), presents each 8x8 digit to the
//! fixed brain, trains a one-layer readout on its activity, and reports accuracy.
//! The wiring never changes — only the readout learns.
//!
//! ```sh
//! cargo run --release --example read_digits
//! ```

use wetware::data;
use wetware::tasks::digits;

fn main() {
    println!("loading the fruit-fly connectome...");
    let conn = data::load().expect("failed to load the connectome");
    let nnz = conn.w.iter().filter(|&&v| v != 0.0).count();
    println!("brain: {} neurons, {} connections\n", conn.n, nnz);

    println!("presenting handwritten digits, training a readout on the brain's activity...");
    let r = digits::demo(&conn, &Default::default()).expect("digits demo failed");

    let acc = report_f(&r, "accuracy");
    let base = report_f(&r, "baseline_accuracy");
    let n = report_i(&r, "test_n");
    println!(
        "\n  a real fly brain read handwriting at {:.0}%",
        acc * 100.0
    );
    println!(
        "  (majority-class baseline: {:.0}%, over {n} test digits)",
        base * 100.0
    );
    println!("\nThe connectome was never trained. Only a one-layer readout learned.");
}

fn report_f(r: &wetware::Report, key: &str) -> f64 {
    r.iter()
        .find(|(k, _)| k == key)
        .and_then(|(_, v)| match v {
            wetware::Metric::F(x) => Some(*x),
            _ => None,
        })
        .unwrap_or(f64::NAN)
}

fn report_i(r: &wetware::Report, key: &str) -> i64 {
    r.iter()
        .find(|(k, _)| k == key)
        .and_then(|(_, v)| match v {
            wetware::Metric::I(x) => Some(*x),
            _ => None,
        })
        .unwrap_or(0)
}
