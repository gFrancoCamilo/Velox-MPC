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

pub struct PhaseProfiler {
    enabled: bool,
    tag: String,
    spans_open: HashMap<(&'static str, u64), Instant>,
    agg: HashMap<&'static str, Agg>,
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
            self.agg.entry(label).or_insert_with(Agg::new).add(t0.elapsed());
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
    fn unclosed_and_mismatched_spans_are_ignored() {
        let mut p = enabled_profiler();
        p.start("RA", 7, 2);
        // stop with a different key -> no aggregation, open span stays.
        p.stop("RA", 7, 3);
        assert!(p.agg.get("RA").is_none());
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
