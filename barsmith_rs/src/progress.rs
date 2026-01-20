use crate::config::Config;

#[derive(Debug)]
pub struct ProgressTracker {
    processed: usize,
    limit: Option<usize>,
    start_offset: usize,
}

impl ProgressTracker {
    pub fn new(config: &Config) -> Self {
        let start_offset = config.resume_offset as usize;
        Self {
            processed: start_offset,
            limit: config.max_combos,
            start_offset,
        }
    }

    pub fn processed(&self) -> usize {
        self.processed
    }

    pub fn start_offset(&self) -> usize {
        self.start_offset
    }

    pub fn processed_since_start(&self) -> usize {
        self.processed.saturating_sub(self.start_offset)
    }

    pub fn record_batch(&mut self, batch_size: usize) -> bool {
        self.processed += batch_size;
        self.limit.is_none_or(|limit| self.processed < limit)
    }
}
