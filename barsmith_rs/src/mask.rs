use std::collections::HashMap;
use std::sync::{Arc, RwLock};

pub type MaskBuffer = Arc<Vec<bool>>;

#[derive(Clone, Debug)]
pub struct MaskCache {
    inner: Arc<RwLock<HashMap<String, MaskBuffer>>>,
    max_entries: usize,
}

impl Default for MaskCache {
    fn default() -> Self {
        Self::new()
    }
}

impl MaskCache {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
            max_entries: 8_192,
        }
    }

    pub fn with_max_entries(max_entries: usize) -> Self {
        Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
            max_entries: max_entries.max(1),
        }
    }

    pub fn get(&self, key: &str) -> Option<MaskBuffer> {
        self.inner
            .read()
            .ok()
            .and_then(|lock| lock.get(key).cloned())
    }

    pub fn get_or_insert(&self, key: &str, mask: Vec<bool>) -> MaskBuffer {
        if let Some(existing) = self.get(key) {
            return existing;
        }
        let arc = Arc::new(mask);
        if let Ok(mut guard) = self.inner.write() {
            if guard.len() >= self.max_entries {
                // Simple generational eviction: when the cache exceeds the bound,
                // clear all entries. This bounds memory while preserving correctness
                // (masks are recomputed on cache misses).
                guard.clear();
            }
            guard.entry(key.to_string()).or_insert_with(|| arc.clone());
        }
        arc
    }

    pub fn len(&self) -> usize {
        self.inner.read().map(|guard| guard.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn clear(&self) {
        if let Ok(mut guard) = self.inner.write() {
            guard.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_cache_respects_max_entries_via_clear() {
        let cache = MaskCache::with_max_entries(2);
        cache.get_or_insert("a", vec![true, false]);
        cache.get_or_insert("b", vec![false, true]);
        assert_eq!(cache.len(), 2);

        // Inserting a third entry forces a generational clear and leaves
        // only the new entry present.
        cache.get_or_insert("c", vec![true, true]);
        assert_eq!(cache.len(), 1);
        assert!(cache.get("c").is_some());
    }
}
