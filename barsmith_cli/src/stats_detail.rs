use clap::ValueEnum;

use barsmith_rs::config::StatsDetail;

#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum StatsDetailValue {
    Core,
    Full,
}

impl StatsDetailValue {
    pub fn to_mode(self) -> StatsDetail {
        match self {
            StatsDetailValue::Core => StatsDetail::Core,
            StatsDetailValue::Full => StatsDetail::Full,
        }
    }
}
