use itertools::Itertools;

use crate::feature::FeatureDescriptor;

/// Human-readable combination used in reporting, tests, and storage.
/// This remains descriptor-based so existing callers that need feature
/// names can continue to use it.
pub type Combination = Vec<FeatureDescriptor>;

/// Index-based combination used in the hot evaluation path. Each entry
/// is an index into the feature catalog held by `FeaturePools`.
pub type IndexCombination = Vec<usize>;

/// Compute C(n, k) - the number of k-combinations from n items.
/// Returns 0 for invalid inputs (k > n). Returns 1 for k == 0.
pub fn combinations_for_depth(feature_count: usize, depth: usize) -> u128 {
    if depth > feature_count {
        return 0;
    }
    if depth == 0 {
        return 1; // C(n, 0) = 1 for all n >= 0
    }
    // Use symmetry: C(n, k) = C(n, n-k) to minimize iterations
    let k = depth.min(feature_count - depth);
    let mut numerator = 1u128;
    let mut denominator = 1u128;
    for i in 0..k {
        numerator *= (feature_count - i) as u128;
        denominator *= (i + 1) as u128;
    }
    numerator / denominator
}

/// Compute the total number of combinations across depths 1..=max_depth.
pub fn total_combinations(feature_count: usize, max_depth: usize) -> u128 {
    (1..=max_depth)
        .map(|depth| combinations_for_depth(feature_count, depth))
        .sum()
}

// =============================================================================
// Combinatorial Unranking
// =============================================================================

/// Convert a global combination index to (depth, local_index_within_depth).
///
/// The global index orders combinations as:
/// - All depth-1 combinations (indices 0 to C(n,1)-1)
/// - All depth-2 combinations (indices C(n,1) to C(n,1)+C(n,2)-1)
/// - ...and so on up to max_depth
///
/// Returns None if the index exceeds total combinations.
pub fn global_to_depth_and_local(
    global_index: u128,
    n: usize,
    max_depth: usize,
) -> Option<(usize, u128)> {
    let mut remaining = global_index;

    for depth in 1..=max_depth {
        let count_at_depth = combinations_for_depth(n, depth);
        if remaining < count_at_depth {
            return Some((depth, remaining));
        }
        remaining -= count_at_depth;
    }

    None // Index exceeds total combinations
}

/// Unrank a local index within a specific depth to get the combination indices.
///
/// Given a 0-based rank within C(n, k) combinations, returns the k indices
/// that form that combination in lexicographic order.
///
/// Uses the combinatorial number system for O(k) complexity.
///
/// # Algorithm
///
/// For lexicographic ordering, we find elements from left to right:
/// - The first element c₀ is the smallest value where
///   C(n-1-c₀, k-1) <= remaining_rank
/// - Subtract C(n-1-c₀, k-1) from remaining and repeat for k-1 elements
pub fn unrank_combination(rank: u128, n: usize, k: usize) -> Vec<usize> {
    if k == 0 || k > n || rank >= combinations_for_depth(n, k) {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(k);
    let mut remaining = rank;
    let mut start = 0usize; // Minimum valid index for next element

    for i in 0..k {
        let elements_remaining = k - i;
        // Find the smallest c where the combinations after choosing c fit
        // We need: combinations of (elements_remaining - 1) from indices > c
        let c = find_lex_element(remaining, n, start, elements_remaining);
        result.push(c);

        // Subtract the count of combinations that start with indices < c
        let combos_skipped = count_combos_before(c, n, start, elements_remaining);
        remaining -= combos_skipped;

        start = c + 1;
    }

    result
}

/// Find the element at position `pos` (0-indexed from left) in a k-combination
/// ranked by `remaining` within the space starting from `start`.
///
/// Uses binary search for O(log n) complexity instead of O(n) linear scan.
fn find_lex_element(remaining: u128, n: usize, start: usize, elements_remaining: usize) -> usize {
    // We're looking for the smallest index c >= start such that
    // remaining < combos_after(c) + count_combos_before(c)
    //
    // Key insight: count_combos_before(c) is monotonically increasing in c,
    // and combos_after(c) is monotonically decreasing. Their sum at index c
    // represents the threshold where combinations starting with c begin.
    // We want the first c where remaining falls within combinations starting at c.

    let max_valid = n - elements_remaining; // Maximum valid index for this position

    if start > max_valid {
        return max_valid;
    }

    // Binary search: find smallest c in [start, max_valid] where condition holds
    let mut lo = start;
    let mut hi = max_valid;

    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let combos_after = combinations_for_depth(n - mid - 1, elements_remaining - 1);
        let combos_before = count_combos_before(mid, n, start, elements_remaining);

        if remaining < combos_after + combos_before {
            // Condition satisfied at mid, but there might be a smaller c
            hi = mid;
        } else {
            // Condition not satisfied, need larger c
            lo = mid + 1;
        }
    }

    lo
}

/// Count combinations that start with an index less than `c`.
fn count_combos_before(c: usize, n: usize, start: usize, elements_remaining: usize) -> u128 {
    let mut total = 0u128;
    for idx in start..c {
        total += combinations_for_depth(n - idx - 1, elements_remaining - 1);
    }
    total
}

/// Compute the rank of a sorted combination (inverse of unrank).
/// Useful for testing roundtrip correctness.
pub fn rank_combination(combo: &[usize], n: usize) -> u128 {
    let k = combo.len();
    if k == 0 || k > n {
        return 0;
    }

    let mut rank = 0u128;
    let mut start = 0usize;

    for (i, &c) in combo.iter().enumerate() {
        let elements_remaining = k - i;
        // Add combinations that come before this choice
        rank += count_combos_before(c, n, start, elements_remaining);
        start = c + 1;
    }

    rank
}

/// Unrank a global index across all depths to feature indices.
/// Returns None if the index exceeds total combinations.
pub fn unrank_global(global_index: u128, n: usize, max_depth: usize) -> Option<Vec<usize>> {
    let (depth, local_index) = global_to_depth_and_local(global_index, n, max_depth)?;
    Some(unrank_combination(local_index, n, depth))
}

// =============================================================================
// Seekable Iterator (O(k) startup instead of O(n))
// =============================================================================

/// A combination iterator that can start from any global index in O(k) time.
///
/// Unlike `CombinationIterator` which must iterate through all combinations
/// to reach a specific offset, this iterator uses combinatorial unranking
/// to jump directly to any position.
pub struct SeekableCombinationIterator<'a> {
    features: &'a [FeatureDescriptor],
    n: usize,
    max_depth: usize,
    current_depth: usize,
    current_indices: Vec<usize>, // Current combination as feature indices
    exhausted: bool,
}

impl<'a> SeekableCombinationIterator<'a> {
    /// Create an iterator starting at global_index (0-based).
    ///
    /// This operation is O(k) where k is the depth of the starting combination,
    /// compared to O(n) for the naive skip approach.
    pub fn starting_at(
        features: &'a [FeatureDescriptor],
        max_depth: usize,
        start_index: u128,
    ) -> Self {
        let n = features.len();
        let max_depth = max_depth.min(n).max(1);

        if start_index == 0 {
            // Special case: start from the beginning
            return Self {
                features,
                n,
                max_depth,
                current_depth: 1,
                current_indices: vec![0],
                exhausted: n == 0,
            };
        }

        if let Some((depth, local_index)) = global_to_depth_and_local(start_index, n, max_depth) {
            let indices = unrank_combination(local_index, n, depth);
            if indices.is_empty() {
                // Invalid unranking result
                Self {
                    features,
                    n,
                    max_depth,
                    current_depth: max_depth + 1,
                    current_indices: Vec::new(),
                    exhausted: true,
                }
            } else {
                Self {
                    features,
                    n,
                    max_depth,
                    current_depth: depth,
                    current_indices: indices,
                    exhausted: false,
                }
            }
        } else {
            // Start index beyond total - iterator is exhausted
            Self {
                features,
                n,
                max_depth,
                current_depth: max_depth + 1,
                current_indices: Vec::new(),
                exhausted: true,
            }
        }
    }

    /// Create an iterator starting from the beginning.
    pub fn new(features: &'a [FeatureDescriptor], max_depth: usize) -> Self {
        Self::starting_at(features, max_depth, 0)
    }

    /// Advance to the next combination in lexicographic order.
    fn advance(&mut self) {
        if self.exhausted {
            return;
        }

        let k = self.current_indices.len();

        // Try to increment from rightmost position
        for i in (0..k).rev() {
            let max_val = self.n - (k - i);
            if self.current_indices[i] < max_val {
                self.current_indices[i] += 1;
                // Reset all positions to the right
                for j in (i + 1)..k {
                    self.current_indices[j] = self.current_indices[j - 1] + 1;
                }
                return;
            }
        }

        // Exhausted current depth, move to next
        self.current_depth += 1;
        if self.current_depth > self.max_depth {
            self.exhausted = true;
            return;
        }

        // Start at first combination of new depth: [0, 1, 2, ..., depth-1]
        self.current_indices = (0..self.current_depth).collect();
    }

    /// Get the current global index (for debugging/verification).
    pub fn current_global_index(&self) -> Option<u128> {
        if self.exhausted {
            return None;
        }

        // Sum combinations from all previous depths
        let mut index = 0u128;
        for d in 1..self.current_depth {
            index += combinations_for_depth(self.n, d);
        }

        // Add local rank within current depth
        index += rank_combination(&self.current_indices, self.n);

        Some(index)
    }
}

impl<'a> Iterator for SeekableCombinationIterator<'a> {
    type Item = Combination;

    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }

        // Build combination from current indices
        let combo: Combination = self
            .current_indices
            .iter()
            .map(|&i| self.features[i].clone())
            .collect();

        self.advance();
        Some(combo)
    }
}

// =============================================================================
// Index-based Seekable Iterator and Batcher (evaluation hot path)
// =============================================================================

/// Seekable iterator that yields index-based combinations only. This avoids
/// cloning feature descriptors in the hot evaluation path; indices are later
/// mapped back to names only for reporting/storage.
pub struct SeekableIndexIterator {
    n: usize,
    max_depth: usize,
    current_depth: usize,
    current_indices: Vec<usize>,
    exhausted: bool,
}

impl SeekableIndexIterator {
    pub fn starting_at(n: usize, max_depth: usize, start_index: u128) -> Self {
        let max_depth = max_depth.min(n).max(1);

        if n == 0 {
            return Self {
                n,
                max_depth,
                current_depth: max_depth + 1,
                current_indices: Vec::new(),
                exhausted: true,
            };
        }

        if let Some(indices) = unrank_global(start_index, n, max_depth) {
            let depth = indices.len();
            Self {
                n,
                max_depth,
                current_depth: depth,
                current_indices: indices,
                exhausted: false,
            }
        } else {
            Self {
                n,
                max_depth,
                current_depth: max_depth + 1,
                current_indices: Vec::new(),
                exhausted: true,
            }
        }
    }

    fn advance(&mut self) {
        if self.exhausted {
            return;
        }

        let k = self.current_indices.len();

        // Try to increment from rightmost position
        for i in (0..k).rev() {
            let max_val = self.n - (k - i);
            if self.current_indices[i] < max_val {
                self.current_indices[i] += 1;
                // Reset all positions to the right
                for j in (i + 1)..k {
                    self.current_indices[j] = self.current_indices[j - 1] + 1;
                }
                return;
            }
        }

        // Exhausted current depth, move to next
        self.current_depth += 1;
        if self.current_depth > self.max_depth {
            self.exhausted = true;
            return;
        }

        // Start at first combination of new depth: [0, 1, 2, ..., depth-1]
        self.current_indices = (0..self.current_depth).collect();
    }
}

impl Iterator for SeekableIndexIterator {
    type Item = IndexCombination;

    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }

        let combo = self.current_indices.clone();
        self.advance();
        Some(combo)
    }
}

// =============================================================================
// Original Types (kept for compatibility)
// =============================================================================

pub struct FeaturePools {
    features: Vec<FeatureDescriptor>,
}

impl FeaturePools {
    pub fn new(features: Vec<FeatureDescriptor>) -> Self {
        Self { features }
    }

    pub fn descriptors(&self) -> &[FeatureDescriptor] {
        &self.features
    }
}

/// Original combination iterator using itertools (O(n) skip for resume).
/// Kept for compatibility and testing.
pub struct CombinationIterator<'a> {
    features: &'a [FeatureDescriptor],
    current_depth: usize,
    max_depth: usize,
    inner: Option<Box<dyn Iterator<Item = Combination> + 'a>>,
}

impl<'a> CombinationIterator<'a> {
    pub fn new(features: &'a [FeatureDescriptor], max_depth: usize) -> Self {
        Self {
            features,
            current_depth: 1,
            max_depth: max_depth.max(1),
            inner: None,
        }
    }
}

impl<'a> Iterator for CombinationIterator<'a> {
    type Item = Combination;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(iter) = self.inner.as_mut() {
                if let Some(combo) = iter.next() {
                    return Some(combo);
                }
            }

            if self.current_depth > self.max_depth {
                return None;
            }

            let depth = self.current_depth;
            self.current_depth += 1;
            self.inner = Some(Box::new(
                self.features
                    .iter()
                    .cloned()
                    .combinations(depth)
                    .map(|combo| combo.into_iter().collect()),
            ));
        }
    }
}

// =============================================================================
// CombinationBatcher (now uses O(1) seek via SeekableCombinationIterator)
// =============================================================================

pub struct CombinationBatcher<'a> {
    iter: SeekableCombinationIterator<'a>,
}

impl<'a> CombinationBatcher<'a> {
    /// Create a new batcher starting at the given offset.
    ///
    /// This now uses O(k) combinatorial unranking instead of O(n) iteration,
    /// where k is the depth at the starting position.
    pub fn new(features: &'a FeaturePools, max_depth: usize, start_offset: u64) -> Self {
        let iter = SeekableCombinationIterator::starting_at(
            features.descriptors(),
            max_depth,
            start_offset as u128,
        );
        Self { iter }
    }

    pub fn next_batch(&mut self, batch_size: usize) -> Option<Vec<Combination>> {
        let mut batch = Vec::with_capacity(batch_size);
        while batch.len() < batch_size {
            match self.iter.next() {
                Some(combo) => batch.push(combo),
                None => break,
            }
        }
        if batch.is_empty() { None } else { Some(batch) }
    }
}

/// Index-based batcher used by the evaluation pipeline to avoid cloning
/// feature descriptors for every enumerated combination.
pub struct IndexCombinationBatcher {
    iter: SeekableIndexIterator,
}

impl IndexCombinationBatcher {
    pub fn new(features: &FeaturePools, max_depth: usize, start_offset: u64) -> Self {
        let n = features.descriptors().len();
        let iter = SeekableIndexIterator::starting_at(n, max_depth, start_offset as u128);
        Self { iter }
    }

    pub fn next_batch(&mut self, batch_size: usize) -> Option<Vec<IndexCombination>> {
        let mut batch = Vec::with_capacity(batch_size);
        while batch.len() < batch_size {
            match self.iter.next() {
                Some(combo) => batch.push(combo),
                None => break,
            }
        }
        if batch.is_empty() { None } else { Some(batch) }
    }
}

// =============================================================================
// Unit Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature::FeatureCategory;

    fn make_features(n: usize) -> Vec<FeatureDescriptor> {
        (0..n)
            .map(|i| FeatureDescriptor {
                name: format!("f{}", i),
                category: FeatureCategory::Boolean,
                note: "test".to_string(),
            })
            .collect()
    }

    // -------------------------------------------------------------------------
    // combinations_for_depth tests
    // -------------------------------------------------------------------------

    #[test]
    fn combinations_for_depth_basic() {
        // C(5, 1) = 5
        assert_eq!(combinations_for_depth(5, 1), 5);
        // C(5, 2) = 10
        assert_eq!(combinations_for_depth(5, 2), 10);
        // C(5, 3) = 10
        assert_eq!(combinations_for_depth(5, 3), 10);
        // C(5, 4) = 5
        assert_eq!(combinations_for_depth(5, 4), 5);
        // C(5, 5) = 1
        assert_eq!(combinations_for_depth(5, 5), 1);
    }

    #[test]
    fn combinations_for_depth_edge_cases() {
        // depth = 0 returns 1 (C(n, 0) = 1)
        assert_eq!(combinations_for_depth(5, 0), 1);
        assert_eq!(combinations_for_depth(0, 0), 1);
        // depth > n returns 0
        assert_eq!(combinations_for_depth(5, 6), 0);
        // n = 0, k > 0 returns 0
        assert_eq!(combinations_for_depth(0, 1), 0);
        // C(1, 1) = 1
        assert_eq!(combinations_for_depth(1, 1), 1);
    }

    #[test]
    fn combinations_for_depth_large() {
        // C(100, 2) = 4950
        assert_eq!(combinations_for_depth(100, 2), 4950);
        // C(100, 3) = 161700
        assert_eq!(combinations_for_depth(100, 3), 161700);
        // C(1000, 2) = 499500
        assert_eq!(combinations_for_depth(1000, 2), 499500);
    }

    #[test]
    fn combinations_for_depth_symmetry() {
        // C(n, k) = C(n, n-k)
        for n in 1..=20 {
            for k in 0..=n {
                assert_eq!(
                    combinations_for_depth(n, k),
                    combinations_for_depth(n, n - k),
                    "Symmetry failed for C({}, {})",
                    n,
                    k
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // total_combinations tests
    // -------------------------------------------------------------------------

    #[test]
    fn total_combinations_basic() {
        // n=5, max_depth=1: C(5,1) = 5
        assert_eq!(total_combinations(5, 1), 5);
        // n=5, max_depth=2: C(5,1) + C(5,2) = 5 + 10 = 15
        assert_eq!(total_combinations(5, 2), 15);
        // n=5, max_depth=3: 5 + 10 + 10 = 25
        assert_eq!(total_combinations(5, 3), 25);
    }

    #[test]
    fn total_combinations_all_depths() {
        // Sum of all depths = 2^n - 1
        for n in 1..=10 {
            let total = total_combinations(n, n);
            let expected = (1u128 << n) - 1;
            assert_eq!(total, expected, "Failed for n={}", n);
        }
    }

    // -------------------------------------------------------------------------
    // global_to_depth_and_local tests
    // -------------------------------------------------------------------------

    #[test]
    fn global_to_depth_basic() {
        // n=5, max_depth=3
        // Depth 1: indices 0-4 (5 combos)
        // Depth 2: indices 5-14 (10 combos)
        // Depth 3: indices 15-24 (10 combos)

        assert_eq!(global_to_depth_and_local(0, 5, 3), Some((1, 0)));
        assert_eq!(global_to_depth_and_local(4, 5, 3), Some((1, 4)));
        assert_eq!(global_to_depth_and_local(5, 5, 3), Some((2, 0)));
        assert_eq!(global_to_depth_and_local(14, 5, 3), Some((2, 9)));
        assert_eq!(global_to_depth_and_local(15, 5, 3), Some((3, 0)));
        assert_eq!(global_to_depth_and_local(24, 5, 3), Some((3, 9)));
    }

    #[test]
    fn global_to_depth_beyond_total() {
        // Index exceeds total should return None
        assert_eq!(global_to_depth_and_local(25, 5, 3), None);
        assert_eq!(global_to_depth_and_local(100, 5, 3), None);
    }

    #[test]
    fn global_to_depth_single_depth() {
        // max_depth=1: only depth-1 combinations
        for i in 0..5 {
            assert_eq!(global_to_depth_and_local(i, 5, 1), Some((1, i)));
        }
        assert_eq!(global_to_depth_and_local(5, 5, 1), None);
    }

    // -------------------------------------------------------------------------
    // unrank_combination tests
    // -------------------------------------------------------------------------

    #[test]
    fn unrank_depth_1() {
        // C(5, 1) = 5 combinations: {0}, {1}, {2}, {3}, {4}
        assert_eq!(unrank_combination(0, 5, 1), vec![0]);
        assert_eq!(unrank_combination(1, 5, 1), vec![1]);
        assert_eq!(unrank_combination(2, 5, 1), vec![2]);
        assert_eq!(unrank_combination(3, 5, 1), vec![3]);
        assert_eq!(unrank_combination(4, 5, 1), vec![4]);
    }

    #[test]
    fn unrank_depth_2() {
        // C(5, 2) = 10 combinations in lex order:
        // {0,1}, {0,2}, {0,3}, {0,4}, {1,2}, {1,3}, {1,4}, {2,3}, {2,4}, {3,4}
        assert_eq!(unrank_combination(0, 5, 2), vec![0, 1]);
        assert_eq!(unrank_combination(1, 5, 2), vec![0, 2]);
        assert_eq!(unrank_combination(2, 5, 2), vec![0, 3]);
        assert_eq!(unrank_combination(3, 5, 2), vec![0, 4]);
        assert_eq!(unrank_combination(4, 5, 2), vec![1, 2]);
        assert_eq!(unrank_combination(5, 5, 2), vec![1, 3]);
        assert_eq!(unrank_combination(6, 5, 2), vec![1, 4]);
        assert_eq!(unrank_combination(7, 5, 2), vec![2, 3]);
        assert_eq!(unrank_combination(8, 5, 2), vec![2, 4]);
        assert_eq!(unrank_combination(9, 5, 2), vec![3, 4]);
    }

    #[test]
    fn unrank_depth_3() {
        // C(5, 3) = 10 combinations in lex order:
        // {0,1,2}, {0,1,3}, {0,1,4}, {0,2,3}, {0,2,4}, {0,3,4}, {1,2,3}, {1,2,4}, {1,3,4}, {2,3,4}
        assert_eq!(unrank_combination(0, 5, 3), vec![0, 1, 2]);
        assert_eq!(unrank_combination(1, 5, 3), vec![0, 1, 3]);
        assert_eq!(unrank_combination(2, 5, 3), vec![0, 1, 4]);
        assert_eq!(unrank_combination(3, 5, 3), vec![0, 2, 3]);
        assert_eq!(unrank_combination(4, 5, 3), vec![0, 2, 4]);
        assert_eq!(unrank_combination(5, 5, 3), vec![0, 3, 4]);
        assert_eq!(unrank_combination(6, 5, 3), vec![1, 2, 3]);
        assert_eq!(unrank_combination(7, 5, 3), vec![1, 2, 4]);
        assert_eq!(unrank_combination(8, 5, 3), vec![1, 3, 4]);
        assert_eq!(unrank_combination(9, 5, 3), vec![2, 3, 4]);
    }

    #[test]
    fn unrank_invalid_inputs() {
        let empty: Vec<usize> = vec![];
        // k = 0
        assert_eq!(unrank_combination(0, 5, 0), empty);
        // k > n
        assert_eq!(unrank_combination(0, 5, 6), empty);
        // rank exceeds C(n, k)
        assert_eq!(unrank_combination(10, 5, 2), empty);
        assert_eq!(unrank_combination(100, 5, 2), empty);
    }

    #[test]
    fn unrank_single_element() {
        let empty: Vec<usize> = vec![];
        // C(1, 1) = 1
        assert_eq!(unrank_combination(0, 1, 1), vec![0]);
        assert_eq!(unrank_combination(1, 1, 1), empty);
    }

    #[test]
    fn unrank_all_elements() {
        let empty: Vec<usize> = vec![];
        // C(5, 5) = 1: only {0, 1, 2, 3, 4}
        assert_eq!(unrank_combination(0, 5, 5), vec![0, 1, 2, 3, 4]);
        assert_eq!(unrank_combination(1, 5, 5), empty);
    }

    // -------------------------------------------------------------------------
    // rank_combination tests
    // -------------------------------------------------------------------------

    #[test]
    fn rank_depth_1() {
        assert_eq!(rank_combination(&[0], 5), 0);
        assert_eq!(rank_combination(&[1], 5), 1);
        assert_eq!(rank_combination(&[4], 5), 4);
    }

    #[test]
    fn rank_depth_2() {
        assert_eq!(rank_combination(&[0, 1], 5), 0);
        assert_eq!(rank_combination(&[0, 4], 5), 3);
        assert_eq!(rank_combination(&[1, 2], 5), 4);
        assert_eq!(rank_combination(&[3, 4], 5), 9);
    }

    #[test]
    fn rank_depth_3() {
        assert_eq!(rank_combination(&[0, 1, 2], 5), 0);
        assert_eq!(rank_combination(&[2, 3, 4], 5), 9);
    }

    // -------------------------------------------------------------------------
    // Rank/Unrank roundtrip tests
    // -------------------------------------------------------------------------

    #[test]
    fn rank_unrank_roundtrip_small() {
        for n in 1..=8 {
            for k in 1..=n.min(4) {
                let count = combinations_for_depth(n, k);
                for rank in 0..count {
                    let combo = unrank_combination(rank, n, k);
                    let reranked = rank_combination(&combo, n);
                    assert_eq!(
                        rank, reranked,
                        "Roundtrip failed: n={}, k={}, rank={}, combo={:?}",
                        n, k, rank, combo
                    );
                }
            }
        }
    }

    #[test]
    fn rank_unrank_roundtrip_large() {
        // Test specific large cases
        let test_cases = [
            (100, 2, 0),
            (100, 2, 4949),
            (100, 3, 0),
            (100, 3, 161699),
            (1000, 2, 250000),
        ];

        for (n, k, rank) in test_cases {
            let combo = unrank_combination(rank, n, k);
            let reranked = rank_combination(&combo, n);
            assert_eq!(
                rank, reranked,
                "Roundtrip failed: n={}, k={}, rank={}",
                n, k, rank
            );
        }
    }

    #[test]
    fn unrank_rank_roundtrip_exhaustive_small() {
        // For small n/k, verify all combinations
        for n in 2..=6 {
            for k in 1..=n.min(3) {
                let count = combinations_for_depth(n, k) as usize;
                for rank in 0..count {
                    let combo = unrank_combination(rank as u128, n, k);
                    assert_eq!(
                        combo.len(),
                        k,
                        "Wrong length for n={}, k={}, rank={}",
                        n,
                        k,
                        rank
                    );

                    // Verify sorted and in bounds
                    for i in 0..k {
                        assert!(combo[i] < n, "Out of bounds: {:?}", combo);
                        if i > 0 {
                            assert!(combo[i] > combo[i - 1], "Not sorted: {:?}", combo);
                        }
                    }
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // unrank_global tests
    // -------------------------------------------------------------------------

    #[test]
    fn unrank_global_basic() {
        // n=5, max_depth=2
        // Global 0 -> depth 1, local 0 -> [0]
        assert_eq!(unrank_global(0, 5, 2), Some(vec![0]));
        // Global 4 -> depth 1, local 4 -> [4]
        assert_eq!(unrank_global(4, 5, 2), Some(vec![4]));
        // Global 5 -> depth 2, local 0 -> [0, 1]
        assert_eq!(unrank_global(5, 5, 2), Some(vec![0, 1]));
        // Global 14 -> depth 2, local 9 -> [3, 4]
        assert_eq!(unrank_global(14, 5, 2), Some(vec![3, 4]));
    }

    #[test]
    fn unrank_global_beyond_total() {
        assert_eq!(unrank_global(15, 5, 2), None);
        assert_eq!(unrank_global(100, 5, 2), None);
    }

    // -------------------------------------------------------------------------
    // SeekableCombinationIterator tests
    // -------------------------------------------------------------------------

    #[test]
    fn seekable_from_start() {
        let features = make_features(5);
        let mut iter = SeekableCombinationIterator::starting_at(&features, 2, 0);

        // First 5 should be depth-1
        for i in 0..5 {
            let combo = iter.next().expect("Should have combo");
            assert_eq!(combo.len(), 1);
            assert_eq!(combo[0].name, format!("f{}", i));
        }

        // Next should be depth-2
        let combo = iter.next().expect("Should have combo");
        assert_eq!(combo.len(), 2);
        assert_eq!(combo[0].name, "f0");
        assert_eq!(combo[1].name, "f1");
    }

    #[test]
    fn seekable_from_middle() {
        let features = make_features(5);

        // Skip to index 5 (first depth-2 combination)
        let mut iter = SeekableCombinationIterator::starting_at(&features, 2, 5);

        let combo = iter.next().expect("Should have combo");
        assert_eq!(combo.len(), 2);
        assert_eq!(combo[0].name, "f0");
        assert_eq!(combo[1].name, "f1");
    }

    #[test]
    fn seekable_from_end() {
        let features = make_features(5);

        // Total for depth 1-2: 5 + 10 = 15
        // Skip to last combo (index 14)
        let mut iter = SeekableCombinationIterator::starting_at(&features, 2, 14);

        let combo = iter.next().expect("Should have combo");
        assert_eq!(combo.len(), 2);
        assert_eq!(combo[0].name, "f3");
        assert_eq!(combo[1].name, "f4");

        // Next should be None
        assert!(iter.next().is_none());
    }

    #[test]
    fn seekable_beyond_end() {
        let features = make_features(5);

        // Beyond total (15 for depth 1-2)
        let mut iter = SeekableCombinationIterator::starting_at(&features, 2, 100);
        assert!(iter.next().is_none());
    }

    #[test]
    fn seekable_matches_sequential() {
        let features = make_features(6);

        // Collect all combinations sequentially
        let sequential: Vec<_> = CombinationIterator::new(&features, 3).collect();

        // Verify seekable iterator at various offsets
        for offset in [0u128, 1, 5, 6, 10, 15, 20, 25] {
            if offset as usize >= sequential.len() {
                continue;
            }
            let seekable: Vec<_> =
                SeekableCombinationIterator::starting_at(&features, 3, offset).collect();
            assert_eq!(
                seekable,
                sequential[offset as usize..].to_vec(),
                "Mismatch at offset {}",
                offset
            );
        }
    }

    #[test]
    fn seekable_all_offsets_match() {
        let features = make_features(5);
        let sequential: Vec<_> = CombinationIterator::new(&features, 2).collect();
        let total = sequential.len();

        // Test every single offset
        for offset in 0..=total {
            let seekable: Vec<_> =
                SeekableCombinationIterator::starting_at(&features, 2, offset as u128).collect();

            if offset < total {
                assert_eq!(
                    seekable,
                    sequential[offset..].to_vec(),
                    "Mismatch at offset {}",
                    offset
                );
            } else {
                assert!(seekable.is_empty(), "Should be empty at offset {}", offset);
            }
        }
    }

    #[test]
    fn seekable_current_global_index() {
        let features = make_features(5);
        let mut iter = SeekableCombinationIterator::starting_at(&features, 2, 0);

        for expected_index in 0..15u128 {
            assert_eq!(
                iter.current_global_index(),
                Some(expected_index),
                "Wrong index at step {}",
                expected_index
            );
            iter.next();
        }
        assert_eq!(iter.current_global_index(), None);
    }

    #[test]
    fn seekable_empty_features() {
        let features: Vec<FeatureDescriptor> = vec![];
        let mut iter = SeekableCombinationIterator::starting_at(&features, 2, 0);
        assert!(iter.next().is_none());
    }

    #[test]
    fn seekable_single_feature() {
        let features = make_features(1);
        let mut iter = SeekableCombinationIterator::starting_at(&features, 2, 0);

        // Only one depth-1 combination
        let combo = iter.next().expect("Should have one combo");
        assert_eq!(combo.len(), 1);
        assert_eq!(combo[0].name, "f0");

        // No more
        assert!(iter.next().is_none());
    }

    #[test]
    fn seekable_depth_transition() {
        let features = make_features(4);

        // Start at last depth-1 combo (index 3)
        let mut iter = SeekableCombinationIterator::starting_at(&features, 3, 3);

        // Should get [3]
        let combo = iter.next().expect("Should have combo");
        assert_eq!(combo.len(), 1);
        assert_eq!(combo[0].name, "f3");

        // Should transition to depth-2: [0, 1]
        let combo = iter.next().expect("Should have combo");
        assert_eq!(combo.len(), 2);
        assert_eq!(combo[0].name, "f0");
        assert_eq!(combo[1].name, "f1");
    }

    // -------------------------------------------------------------------------
    // CombinationBatcher tests
    // -------------------------------------------------------------------------

    #[test]
    fn batcher_from_start() {
        let features = make_features(5);
        let pools = FeaturePools::new(features);
        let mut batcher = CombinationBatcher::new(&pools, 2, 0);

        let batch = batcher.next_batch(3).expect("Should have batch");
        assert_eq!(batch.len(), 3);
        assert_eq!(batch[0].len(), 1);
        assert_eq!(batch[0][0].name, "f0");
    }

    #[test]
    fn batcher_with_offset() {
        let features = make_features(5);
        let pools = FeaturePools::new(features);

        // Skip first 5 (all depth-1)
        let mut batcher = CombinationBatcher::new(&pools, 2, 5);

        let batch = batcher.next_batch(2).expect("Should have batch");
        assert_eq!(batch.len(), 2);
        // First depth-2 combo
        assert_eq!(batch[0].len(), 2);
        assert_eq!(batch[0][0].name, "f0");
        assert_eq!(batch[0][1].name, "f1");
    }

    #[test]
    fn batcher_exhaustion() {
        let features = make_features(3);
        let pools = FeaturePools::new(features);
        let mut batcher = CombinationBatcher::new(&pools, 1, 0);

        // Should get all 3 depth-1 combos
        let batch = batcher.next_batch(10).expect("Should have batch");
        assert_eq!(batch.len(), 3);

        // Next should be None
        assert!(batcher.next_batch(10).is_none());
    }

    #[test]
    fn batcher_matches_sequential_collection() {
        let features = make_features(6);
        let pools = FeaturePools::new(features.clone());

        // Collect via batcher
        let mut batcher = CombinationBatcher::new(&pools, 2, 0);
        let mut batcher_combos = Vec::new();
        while let Some(batch) = batcher.next_batch(5) {
            batcher_combos.extend(batch);
        }

        // Collect via sequential iterator
        let sequential: Vec<_> = CombinationIterator::new(&features, 2).collect();

        assert_eq!(batcher_combos, sequential);
    }

    #[test]
    fn batcher_with_large_offset() {
        let features = make_features(10);
        let pools = FeaturePools::new(features);

        // C(10, 1) = 10, C(10, 2) = 45, total = 55
        // Skip to index 50
        let mut batcher = CombinationBatcher::new(&pools, 2, 50);

        let batch = batcher.next_batch(10).expect("Should have batch");
        // Should have 5 remaining combos
        assert_eq!(batch.len(), 5);
    }

    // -------------------------------------------------------------------------
    // Performance verification (not timing, just correctness at scale)
    // -------------------------------------------------------------------------

    #[test]
    fn seekable_handles_larger_feature_set() {
        let features = make_features(100);

        // Skip to middle of depth-2: C(100, 1) + C(100, 2)/2 = 100 + 2475 = 2575
        let offset = 2575u128;
        let mut iter = SeekableCombinationIterator::starting_at(&features, 2, offset);

        let combo = iter.next().expect("Should have combo");
        assert_eq!(combo.len(), 2);

        // Verify it's a valid combination
        assert!(combo[0].name.starts_with("f"));
        assert!(combo[1].name.starts_with("f"));
    }

    #[test]
    fn unrank_handles_depth_4() {
        // C(20, 4) = 4845
        let combo = unrank_combination(2422, 20, 4); // Middle
        assert_eq!(combo.len(), 4);

        // Verify roundtrip
        let rank = rank_combination(&combo, 20);
        assert_eq!(rank, 2422);
    }

    #[test]
    fn unrank_handles_depth_5() {
        // C(30, 5) = 142506
        let combo = unrank_combination(71253, 30, 5); // Middle
        assert_eq!(combo.len(), 5);

        // Verify roundtrip
        let rank = rank_combination(&combo, 30);
        assert_eq!(rank, 71253);
    }

    // -------------------------------------------------------------------------
    // Edge case tests
    // -------------------------------------------------------------------------

    #[test]
    fn seekable_max_depth_exceeds_features() {
        let features = make_features(3);
        // max_depth=10, but only 3 features
        let combos: Vec<_> = SeekableCombinationIterator::starting_at(&features, 10, 0).collect();

        // Should get C(3,1) + C(3,2) + C(3,3) = 3 + 3 + 1 = 7 combos
        assert_eq!(combos.len(), 7);
    }

    #[test]
    fn seekable_max_depth_zero_treated_as_one() {
        let features = make_features(3);
        // max_depth=0 should be treated as 1
        let combos: Vec<_> = SeekableCombinationIterator::new(&features, 0).collect();

        // Should get 3 depth-1 combos
        assert_eq!(combos.len(), 3);
    }

    #[test]
    fn combinations_boundary_values() {
        // Test near u128 overflow won't happen for reasonable inputs
        // C(10000, 6) is very large but should not overflow u128
        let large = combinations_for_depth(10000, 6);
        assert!(large > 0);
        assert!(large < u128::MAX / 2); // Sanity check
    }
}
