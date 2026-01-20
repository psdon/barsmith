mod engineer;
pub mod features;
mod thresholds;

use anyhow::{Result, anyhow};
use barsmith_rs::{Config, PermutationPipeline};

use features::FeatureCatalog;

pub use engineer::{BacktestConfig, BacktestTargetKind, run_backtest_with_target};
pub use engineer::{PrepareDatasetOptions, prepare_dataset, prepare_dataset_with_options};

#[derive(Clone, Copy, Debug)]
pub struct CustomPipelineOptions {
    pub ack_new_df: bool,
    pub drop_nan_rows_in_core: bool,
}

impl Default for CustomPipelineOptions {
    fn default() -> Self {
        Self {
            ack_new_df: false,
            drop_nan_rows_in_core: true,
        }
    }
}

pub fn run_custom_pipeline(config: Config) -> Result<()> {
    run_custom_pipeline_with_options(config, CustomPipelineOptions::default())
}

pub fn run_custom_pipeline_with_options(
    config: Config,
    options: CustomPipelineOptions,
) -> Result<()> {
    let mut config = config;
    let prepared_csv = prepare_dataset_with_options(
        &config,
        PrepareDatasetOptions {
            ack_new_df: options.ack_new_df,
            drop_nan_rows_in_core: options.drop_nan_rows_in_core,
        },
    )?;
    config.input_csv = prepared_csv.clone();
    let catalog = FeatureCatalog::build_with_dataset(&config.input_csv, &config)?;

    // Run the heavy permutation pipeline on a dedicated thread with an
    // explicitly enlarged stack to avoid stack overflows in deeply nested
    // evaluation paths on large catalogs and long resumes.
    let pipeline_config = config;
    let descriptors = catalog.descriptors;
    let comparison_specs = catalog.comparison_specs;

    let builder = std::thread::Builder::new()
        .name("barsmith-pipeline".to_string())
        .stack_size(32 * 1024 * 1024);

    let handle = builder
        .spawn(move || -> Result<()> {
            let mut pipeline =
                PermutationPipeline::new(pipeline_config, descriptors, comparison_specs);
            pipeline.run()
        })
        .map_err(|err| anyhow!("failed to spawn barsmith pipeline thread: {err}"))?;

    handle
        .join()
        .map_err(|_| anyhow!("barsmith pipeline thread panicked"))?
}
