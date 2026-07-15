//! Lightweight per-phase span profiler for locating protocol bottlenecks.
//!
//! Enabled only when the `VELOX_PROFILE` env var is set to a non-empty value,
//! so normal runs pay nothing (every method early-returns when disabled).
//! Works identically for CPU and GPU builds; each node aggregates its own
//! numbers and logs them, so a distributed run is profiled by grepping the
//! per-node log files (see scripts/profile_report.sh).
//!
//! Two kinds of measurement, because the protocol is asynchronous and nested:
//!
//!  - **spans** (`start`/`stop`): bracket one instance of a building block —
//!    e.g. a single CTRBC broadcast. Overlapping instances are fine; each is
//!    tracked independently and aggregated per label into count/total/min/max.
//!    Building blocks run concurrently, so the SUM of a label's spans can
//!    exceed wall-clock time — that is expected and is why the report says so.
//!
//!  - **marks** (`mark`): record the first time each top-level phase boundary
//!    is crossed. The report prints the wall-clock gap between consecutive
//!    marks — a true, non-overlapping timeline of the high-level phases.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Combine an instance id and a sub-identifier (e.g. dealer/sender) into a
/// single span key. Distinct (instance, sub) pairs never collide for the
/// instance-id ranges this protocol uses.
#[inline]
fn key_of(instance_id: usize, sub: usize) -> u64 {
    ((instance_id as u64) << 20) ^ (sub as u64)
}

#[derive(Clone)]
struct Agg {
    count: u64,
    total: Duration,
    min: Duration,
    max: Duration,
}

impl Agg {
    fn new() -> Self {
        Self {
            count: 0,
            total: Duration::ZERO,
            min: Duration::MAX,
            max: Duration::ZERO,
        }
    }
    fn add(&mut self, d: Duration) {
        self.count += 1;
        self.total += d;
        if d < self.min {
            self.min = d;
        }
        if d > self.max {
            self.max = d;
        }
    }
}

/// Preferred left-to-right column order for the per-instance pivot; any other
/// labels are appended after these, alphabetically.
const INSTANCE_COLS: &[&str] = &["ACSS_total", "ACSS_dealer_compute", "CTRBC", "AVID", "RA"];

pub struct PhaseProfiler {
    enabled: bool,
    tag: String,
    spans_open: HashMap<(&'static str, u64), Instant>,
    agg: HashMap<&'static str, Agg>,
    /// Per-instance aggregates: instance_id -> label -> agg. Lets the report
    /// break each ACSS instance out instead of only showing the sum.
    per_instance: HashMap<usize, HashMap<&'static str, Agg>>,
    marks: Vec<(&'static str, Instant)>,
    last_report: Option<Instant>,
}

impl PhaseProfiler {
    /// Reads `VELOX_PROFILE` once. `tag` identifies the emitter (e.g. node id)
    /// in the logged report.
    pub fn new(tag: impl Into<String>) -> Self {
        let enabled = std::env::var_os("VELOX_PROFILE").map_or(false, |v| !v.is_empty());
        Self {
            enabled,
            tag: tag.into(),
            spans_open: HashMap::new(),
            agg: HashMap::new(),
            per_instance: HashMap::new(),
            marks: Vec::new(),
            last_report: None,
        }
    }

    #[inline]
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Open a span for building block `label` at instance `instance_id`,
    /// sub-keyed by `sub` (e.g. the dealer/sender). Pair with `stop`.
    #[inline]
    pub fn start(&mut self, label: &'static str, instance_id: usize, sub: usize) {
        if !self.enabled {
            return;
        }
        self.spans_open.insert((label, key_of(instance_id, sub)), Instant::now());
    }

    /// Close the span opened by `start(label, instance_id, sub)` and fold its
    /// duration into the aggregate for `label`. No-op if never started.
    #[inline]
    pub fn stop(&mut self, label: &'static str, instance_id: usize, sub: usize) {
        if !self.enabled {
            return;
        }
        if let Some(t0) = self.spans_open.remove(&(label, key_of(instance_id, sub))) {
            let d = t0.elapsed();
            self.agg.entry(label).or_insert_with(Agg::new).add(d);
            self.per_instance
                .entry(instance_id)
                .or_default()
                .entry(label)
                .or_insert_with(Agg::new)
                .add(d);
        }
    }

    /// Record the first crossing of phase boundary `name` (later crossings are
    /// ignored, so repeated calls from a per-level loop are safe).
    #[inline]
    pub fn mark(&mut self, name: &'static str) {
        if !self.enabled {
            return;
        }
        if self.marks.iter().any(|(n, _)| *n == name) {
            return;
        }
        self.marks.push((name, Instant::now()));
    }

    /// Emit the report, but at most once per second — safe to call from hot
    /// completion paths that have no natural end-of-protocol signal.
    pub fn report_throttled(&mut self) {
        if !self.enabled {
            return;
        }
        let now = Instant::now();
        if let Some(last) = self.last_report {
            if now.duration_since(last) < Duration::from_secs(1) {
                return;
            }
        }
        self.last_report = Some(now);
        self.report();
    }

    /// Log the current span aggregates and phase timeline at `info` level.
    pub fn report(&self) {
        if !self.enabled {
            return;
        }
        let build = if cfg!(feature = "gpu") { "gpu" } else { "cpu" };

        let mut rows: Vec<(&&'static str, &Agg)> = self.agg.iter().collect();
        rows.sort_by(|a, b| b.1.total.cmp(&a.1.total));

        let ms = |d: Duration| d.as_secs_f64() * 1000.0;
        let mut out = String::new();
        out.push_str(&format!(
            "\n[profile:{}:{}] building-block spans (concurrent — sum may exceed wall-clock)\n",
            build, self.tag
        ));
        out.push_str(&format!(
            "  {:<18} {:>6} {:>12} {:>10} {:>9} {:>9}\n",
            "block", "count", "total(ms)", "mean(ms)", "min(ms)", "max(ms)"
        ));
        for (label, a) in rows {
            let mean = if a.count > 0 {
                ms(a.total) / a.count as f64
            } else {
                0.0
            };
            out.push_str(&format!(
                "  {:<18} {:>6} {:>12.1} {:>10.2} {:>9.2} {:>9.2}\n",
                label,
                a.count,
                ms(a.total),
                mean,
                if a.min == Duration::MAX { 0.0 } else { ms(a.min) },
                ms(a.max),
            ));
        }

        // Per-instance pivot: one row per ACSS instance id, so a slow instance
        // (or a phase's whole id-range) stands out instead of being summed away.
        if !self.per_instance.is_empty() {
            // Column labels: the preferred ACSS lifecycle order, then any extras.
            let mut cols: Vec<&'static str> = INSTANCE_COLS
                .iter()
                .copied()
                .filter(|c| self.per_instance.values().any(|m| m.contains_key(c)))
                .collect();
            let mut extras: Vec<&'static str> = self
                .per_instance
                .values()
                .flat_map(|m| m.keys().copied())
                .filter(|l| !INSTANCE_COLS.contains(l))
                .collect();
            extras.sort_unstable();
            extras.dedup();
            cols.extend(extras);

            // Sort instances by ACSS_total total desc (fallback: sum of all labels).
            let sort_val = |m: &HashMap<&'static str, Agg>| -> Duration {
                m.get("ACSS_total")
                    .map(|a| a.total)
                    .unwrap_or_else(|| m.values().map(|a| a.total).fold(Duration::ZERO, |x, y| x + y))
            };
            let mut insts: Vec<(usize, &HashMap<&'static str, Agg>)> =
                self.per_instance.iter().map(|(k, v)| (*k, v)).collect();
            insts.sort_by(|a, b| sort_val(b.1).cmp(&sort_val(a.1)).then(a.0.cmp(&b.0)));

            let top_n = std::env::var("VELOX_PROFILE_TOPN")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|n| *n > 0)
                .unwrap_or(64);
            let total_insts = insts.len();
            let shown = total_insts.min(top_n);

            out.push_str(&format!(
                "  -- per-instance spans (ms), {} instance(s){} --\n",
                total_insts,
                if shown < total_insts {
                    format!(", showing slowest {} (VELOX_PROFILE_TOPN)", shown)
                } else {
                    String::new()
                }
            ));
            out.push_str(&format!("  {:>10}", "instance"));
            for c in &cols {
                out.push_str(&format!(" {:>18}", c));
            }
            out.push('\n');
            for (inst, m) in insts.into_iter().take(shown) {
                out.push_str(&format!("  {:>10}", inst));
                for c in &cols {
                    match m.get(c) {
                        Some(a) => out.push_str(&format!(" {:>18.1}", ms(a.total))),
                        None => out.push_str(&format!(" {:>18}", "-")),
                    }
                }
                out.push('\n');
            }
        }

        if self.marks.len() >= 2 {
            out.push_str("  -- phase timeline (wall-clock, non-overlapping) --\n");
            for w in self.marks.windows(2) {
                out.push_str(&format!(
                    "  {:<18} {:>12.1} ms\n",
                    w[0].0,
                    ms(w[1].1.duration_since(w[0].1))
                ));
            }
            let span = self
                .marks
                .last()
                .unwrap()
                .1
                .duration_since(self.marks[0].1);
            out.push_str(&format!(
                "  {:<18} {:>12.1} ms  (first->last mark)\n",
                "TOTAL", ms(span)
            ));
        }
        log::info!("{}", out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Force-enable regardless of the ambient env var, for deterministic tests.
    fn enabled_profiler() -> PhaseProfiler {
        let mut p = PhaseProfiler::new("test");
        p.enabled = true;
        p
    }

    #[test]
    fn disabled_by_default_is_noop() {
        // Not relying on env: a fresh profiler with no VELOX_PROFILE set records nothing.
        let mut p = PhaseProfiler::new("test");
        if std::env::var_os("VELOX_PROFILE").is_none() {
            assert!(!p.enabled());
            p.start("CTRBC", 1, 0);
            p.stop("CTRBC", 1, 0);
            p.mark("phase");
            assert!(p.agg.is_empty());
            assert!(p.per_instance.is_empty());
            assert!(p.marks.is_empty());
        }
    }

    #[test]
    fn spans_aggregate_per_label() {
        let mut p = enabled_profiler();
        for i in 0..3 {
            p.start("CTRBC", i, 0);
            std::thread::sleep(Duration::from_millis(2));
            p.stop("CTRBC", i, 0);
        }
        let a = p.agg.get("CTRBC").expect("label recorded");
        assert_eq!(a.count, 3);
        assert!(a.max >= a.min);
        assert!(a.total >= a.max);
    }

    #[test]
    fn per_instance_breaks_out_each_instance() {
        let mut p = enabled_profiler();
        // Two ACSS instances, each with its own compute + a couple RA spans.
        for inst in [0usize, 1] {
            p.start("ACSS_total", inst, 9);
            p.start("ACSS_dealer_compute", inst, 9);
            std::thread::sleep(Duration::from_millis(1));
            p.stop("ACSS_dealer_compute", inst, 9);
            for sender in 0..2 {
                p.start("RA", inst, sender);
                p.stop("RA", inst, sender);
            }
            p.stop("ACSS_total", inst, 9);
        }
        // Global label aggregate still sums across instances.
        assert_eq!(p.agg.get("RA").unwrap().count, 4);
        // Per-instance view keeps them separate.
        assert_eq!(p.per_instance.len(), 2);
        assert_eq!(p.per_instance[&0].get("RA").unwrap().count, 2);
        assert!(p.per_instance[&0].contains_key("ACSS_total"));
        assert!(p.per_instance[&1].contains_key("ACSS_dealer_compute"));
        // Pivot rendering must not panic.
        p.report();
    }

    #[test]
    fn unclosed_and_mismatched_spans_are_ignored() {
        let mut p = enabled_profiler();
        p.start("RA", 7, 2);
        // stop with a different key -> no aggregation, open span stays.
        p.stop("RA", 7, 3);
        assert!(p.agg.get("RA").is_none());
        assert!(p.per_instance.is_empty());
        assert_eq!(p.spans_open.len(), 1);
    }

    #[test]
    fn marks_record_first_crossing_only() {
        let mut p = enabled_profiler();
        p.mark("mixing");
        p.mark("mixing"); // repeat ignored (per-level loops are safe)
        p.mark("verification");
        assert_eq!(p.marks.len(), 2);
        assert_eq!(p.marks[0].0, "mixing");
        assert_eq!(p.marks[1].0, "verification");
        // report must not panic with spans + marks present.
        p.start("AVID", 1, 0);
        p.stop("AVID", 1, 0);
        p.report();
    }
}
