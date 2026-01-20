use std::collections::HashMap;

#[derive(Clone, Copy, Debug)]
pub enum TradeDirection {
    Long,
    Short,
}

#[derive(Clone, Debug)]
pub struct TradeRecord {
    pub direction: TradeDirection,
    pub entry_index: usize,
    pub exit_index: usize,
    pub entry_price: f64,
    pub exit_price: f64,
    pub rr: f64,
    pub period: i64,
    pub stop_price: f64,
    pub tp_price: f64,
}

/// Inputs for the generic backtest engine.
///
/// The engine is fully strategy-agnostic: callers must provide
/// precomputed entry signals and stop/take-profit levels per bar.
pub struct BacktestInputs {
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,

    pub entry_long: Vec<bool>,
    pub entry_short: Vec<bool>,

    pub stop_long: Vec<f64>,
    pub tp_long: Vec<f64>,
    pub stop_short: Vec<f64>,
    pub tp_short: Vec<f64>,

    /// Period identifier per bar (e.g. week id, month id).
    pub period_index: Vec<i64>,
    /// Last bar index for the period associated with each bar.
    pub period_end_index: Vec<usize>,

    /// Maximum number of trades per period for each direction.
    pub max_trades_per_period_long: u8,
    pub max_trades_per_period_short: u8,

    /// Optional rule: once a winning trade has been recorded in a
    /// period for the given direction, skip any further entries in
    /// that period even if the max_trades_per_period_* cap has not
    /// been reached.
    pub stop_after_first_winning_trade_long: bool,
    pub stop_after_first_winning_trade_short: bool,
}

/// Outputs from the generic backtest engine.
pub struct BacktestOutputs {
    pub target_long: Vec<bool>,
    pub rr_long: Vec<f64>,
    pub target_short: Vec<bool>,
    pub rr_short: Vec<f64>,
    pub trades: Vec<TradeRecord>,
}

const SMALL_DIVISOR: f64 = 1e-9;

/// Run a generic backtest for both long and short directions.
///
/// The engine assumes:
/// - At most one open trade per direction at any time.
/// - No overlapping trades within the same direction (the next
///   candidate index advances to the bar after the exit).
/// - Per-period caps on the number of trades per direction.
pub fn run_backtest(inputs: &BacktestInputs) -> BacktestOutputs {
    let len = inputs.close.len();

    assert_eq!(inputs.high.len(), len, "high length mismatch");
    assert_eq!(inputs.low.len(), len, "low length mismatch");
    assert_eq!(inputs.entry_long.len(), len, "entry_long length mismatch");
    assert_eq!(inputs.entry_short.len(), len, "entry_short length mismatch");
    assert_eq!(inputs.stop_long.len(), len, "stop_long length mismatch");
    assert_eq!(inputs.tp_long.len(), len, "tp_long length mismatch");
    assert_eq!(inputs.stop_short.len(), len, "stop_short length mismatch");
    assert_eq!(inputs.tp_short.len(), len, "tp_short length mismatch");
    assert_eq!(
        inputs.period_index.len(),
        len,
        "period_index length mismatch"
    );
    assert_eq!(
        inputs.period_end_index.len(),
        len,
        "period_end_index length mismatch"
    );

    let mut target_long = vec![false; len];
    let mut rr_long = vec![f64::NAN; len];
    let mut target_short = vec![false; len];
    let mut rr_short = vec![f64::NAN; len];
    let mut trades: Vec<TradeRecord> = Vec::new();

    // Long side
    if inputs.max_trades_per_period_long > 0 {
        run_direction_backtest_long(inputs, &mut target_long, &mut rr_long, &mut trades);
    }

    // Short side
    if inputs.max_trades_per_period_short > 0 {
        run_direction_backtest_short(inputs, &mut target_short, &mut rr_short, &mut trades);
    }

    // Ensure trades are returned in chronological order by entry index,
    // regardless of which direction produced them.
    trades.sort_by_key(|trade| trade.entry_index);

    BacktestOutputs {
        target_long,
        rr_long,
        target_short,
        rr_short,
        trades,
    }
}

fn run_direction_backtest_long(
    inputs: &BacktestInputs,
    target: &mut [bool],
    rr_out: &mut [f64],
    trades: &mut Vec<TradeRecord>,
) {
    let len = inputs.close.len();
    let mut trades_per_period: HashMap<i64, u8> = HashMap::new();
    let mut winners_per_period: HashMap<i64, u8> = HashMap::new();
    let mut i = 0usize;

    while i < len {
        let mut advanced = false;

        if inputs.entry_long[i] {
            let period = inputs.period_index[i];
            let used = trades_per_period.get(&period).copied().unwrap_or(0);
            let winners = winners_per_period.get(&period).copied().unwrap_or(0);
            let stop_after_win = inputs.stop_after_first_winning_trade_long;
            if used < inputs.max_trades_per_period_long && (!stop_after_win || winners == 0) {
                let entry = inputs.close[i];
                let stop = inputs.stop_long[i];
                let tp = inputs.tp_long[i];

                if entry.is_finite()
                    && stop.is_finite()
                    && tp.is_finite()
                    && stop < entry
                    && tp > entry
                {
                    let risk = entry - stop;
                    if risk.abs() > SMALL_DIVISOR {
                        let last_idx = inputs.period_end_index[i];
                        let mut exit_price = entry;
                        let mut exit_idx = i;

                        // Entry is assumed at the close of bar `i`,
                        // so start evaluating SL/TP on the *next* bar.
                        let mut j = i.saturating_add(1);
                        if j > last_idx {
                            // No future bars within this period: exit
                            // at entry.
                            exit_price = entry;
                            exit_idx = i;
                        }

                        while j <= last_idx {
                            let h = inputs.high[j];
                            let l = inputs.low[j];
                            let c = inputs.close[j];
                            if !h.is_finite() || !l.is_finite() || !c.is_finite() {
                                continue;
                            }

                            // Conservative ordering: treat SL hit as dominant when both
                            // SL and TP would be touched within the same bar.
                            if l <= stop {
                                exit_price = stop;
                                exit_idx = j;
                                break;
                            }
                            if h >= tp {
                                exit_price = tp;
                                exit_idx = j;
                                break;
                            }

                            if j == last_idx {
                                exit_price = c;
                                exit_idx = j;
                            }

                            j += 1;
                        }

                        let rr = (exit_price - entry) / risk;
                        rr_out[i] = rr;
                        target[i] = rr.is_finite() && rr > 0.0;

                        if stop_after_win && rr.is_finite() && rr > 0.0 {
                            winners_per_period.insert(period, winners.saturating_add(1));
                        }

                        trades.push(TradeRecord {
                            direction: TradeDirection::Long,
                            entry_index: i,
                            exit_index: exit_idx,
                            entry_price: entry,
                            exit_price,
                            rr,
                            period,
                            stop_price: stop,
                            tp_price: tp,
                        });

                        trades_per_period.insert(period, used.saturating_add(1));
                        i = exit_idx.saturating_add(1);
                        advanced = true;
                    }
                }
            }
        }

        if !advanced {
            i += 1;
        }
    }
}

fn run_direction_backtest_short(
    inputs: &BacktestInputs,
    target: &mut [bool],
    rr_out: &mut [f64],
    trades: &mut Vec<TradeRecord>,
) {
    let len = inputs.close.len();
    let mut trades_per_period: HashMap<i64, u8> = HashMap::new();
    let mut winners_per_period: HashMap<i64, u8> = HashMap::new();
    let mut i = 0usize;

    while i < len {
        let mut advanced = false;

        if inputs.entry_short[i] {
            let period = inputs.period_index[i];
            let used = trades_per_period.get(&period).copied().unwrap_or(0);
            let winners = winners_per_period.get(&period).copied().unwrap_or(0);
            let stop_after_win = inputs.stop_after_first_winning_trade_short;
            if used < inputs.max_trades_per_period_short && (!stop_after_win || winners == 0) {
                let entry = inputs.close[i];
                let stop = inputs.stop_short[i];
                let tp = inputs.tp_short[i];

                if entry.is_finite()
                    && stop.is_finite()
                    && tp.is_finite()
                    && stop > entry
                    && tp < entry
                {
                    let risk = stop - entry;
                    if risk.abs() > SMALL_DIVISOR {
                        let last_idx = inputs.period_end_index[i];
                        let mut exit_price = entry;
                        let mut exit_idx = i;

                        // Entry is assumed at the close of bar `i`,
                        // so start evaluating SL/TP on the *next* bar.
                        let mut j = i.saturating_add(1);
                        if j > last_idx {
                            exit_price = entry;
                            exit_idx = i;
                        }

                        while j <= last_idx {
                            let h = inputs.high[j];
                            let l = inputs.low[j];
                            let c = inputs.close[j];
                            if !h.is_finite() || !l.is_finite() || !c.is_finite() {
                                continue;
                            }

                            // Conservative ordering: treat SL hit as dominant when both
                            // SL and TP would be touched within the same bar.
                            if h >= stop {
                                exit_price = stop;
                                exit_idx = j;
                                break;
                            }
                            if l <= tp {
                                exit_price = tp;
                                exit_idx = j;
                                break;
                            }

                            if j == last_idx {
                                exit_price = c;
                                exit_idx = j;
                            }

                            j += 1;
                        }

                        let rr = (entry - exit_price) / risk;
                        rr_out[i] = rr;
                        target[i] = rr.is_finite() && rr > 0.0;

                        if stop_after_win && rr.is_finite() && rr > 0.0 {
                            winners_per_period.insert(period, winners.saturating_add(1));
                        }

                        trades.push(TradeRecord {
                            direction: TradeDirection::Short,
                            entry_index: i,
                            exit_index: exit_idx,
                            entry_price: entry,
                            exit_price,
                            rr,
                            period,
                            stop_price: stop,
                            tp_price: tp,
                        });

                        trades_per_period.insert(period, used.saturating_add(1));
                        i = exit_idx.saturating_add(1);
                        advanced = true;
                    }
                }
            }
        }

        if !advanced {
            i += 1;
        }
    }
}
