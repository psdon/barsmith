use serde::Serialize;
use std::collections::HashMap;

use once_cell::sync::Lazy;

#[derive(Debug, Clone, Serialize)]
pub struct AssetProfile {
    pub code: &'static str,
    pub description: &'static str,
    pub point_value: f64,
    pub tick_size: f64,
    pub tick_value: f64,
    pub ibkr_commission_per_side: f64,
    pub default_slippage_ticks: f64,
    /// Initial/overnight margin per contract in USD (used as a sizing cap).
    pub margin_per_contract_dollar: f64,
}

static ASSET_PROFILES: Lazy<HashMap<&'static str, AssetProfile>> = Lazy::new(|| {
    let mut m = HashMap::new();

    // NOTE: Commission values are indicative and should be calibrated
    // against the live IBKR schedule in production.
    m.insert(
        "ES",
        AssetProfile {
            code: "ES",
            description: "E-mini S&P 500",
            point_value: 50.0,
            tick_size: 0.25,
            tick_value: 12.50,
            // IBKR all-in ~ $4.50 per round turn ⇒ ~$2.25 per side.
            ibkr_commission_per_side: 2.25,
            // Default slippage set to 0 ticks; configure explicitly per run if desired.
            default_slippage_ticks: 0.0,
            // Typical initial margin varies by broker/exchange; keep as a conservative default.
            margin_per_contract_dollar: 25_000.0,
        },
    );

    m.insert(
        "MES",
        AssetProfile {
            code: "MES",
            description: "Micro E-mini S&P 500",
            point_value: 5.0,
            tick_size: 0.25,
            tick_value: 1.25,
            // IBKR all-in ~ $1.14 per round turn ⇒ ~$0.57 per side.
            ibkr_commission_per_side: 0.57,
            // Default slippage set to 0 ticks; configure explicitly per run if desired.
            default_slippage_ticks: 0.0,
            // User-provided current margin reference.
            margin_per_contract_dollar: 2_500.0,
        },
    );

    m
});

pub fn find_asset(code: &str) -> Option<&'static AssetProfile> {
    let upper = code.to_ascii_uppercase();
    ASSET_PROFILES.get(upper.as_str())
}
