//! Integration tests for combinatorial unranking and seekable iteration.
//!
//! These tests verify that the O(k) unranking implementation produces
//! identical results to the original O(n) iteration approach.

use std::collections::HashSet;
use std::time::Instant;

use barsmith_rs::combinator::{
    CombinationBatcher, CombinationIterator, FeaturePools, SeekableCombinationIterator,
    combinations_for_depth, global_to_depth_and_local, rank_combination, total_combinations,
    unrank_combination, unrank_global,
};
use barsmith_rs::{FeatureCategory, FeatureDescriptor};

fn make_features(n: usize) -> Vec<FeatureDescriptor> {
    (0..n)
        .map(|i| FeatureDescriptor {
            name: format!("f{}", i),
            category: FeatureCategory::Boolean,
            note: "test".to_string(),
        })
        .collect()
}

// =============================================================================
// Exhaustive parity tests (small n)
// =============================================================================

/// Verify that SeekableCombinationIterator produces identical output to
/// CombinationIterator for all offsets.
#[test]
fn seekable_matches_original_exhaustive_n5_depth3() {
    let features = make_features(5);
    let sequential: Vec<_> = CombinationIterator::new(&features, 3).collect();
    let total = sequential.len();

    // Test every possible starting offset
    for offset in 0..=total {
        let seekable: Vec<_> =
            SeekableCombinationIterator::starting_at(&features, 3, offset as u128).collect();

        if offset < total {
            assert_eq!(
                seekable.len(),
                total - offset,
                "Wrong count at offset {}",
                offset
            );
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
fn seekable_matches_original_exhaustive_n7_depth2() {
    let features = make_features(7);
    let sequential: Vec<_> = CombinationIterator::new(&features, 2).collect();
    let total = sequential.len();

    for offset in 0..=total {
        let seekable: Vec<_> =
            SeekableCombinationIterator::starting_at(&features, 2, offset as u128).collect();

        if offset < total {
            assert_eq!(seekable, sequential[offset..].to_vec());
        } else {
            assert!(seekable.is_empty());
        }
    }
}

#[test]
fn seekable_matches_original_exhaustive_n4_depth4() {
    let features = make_features(4);
    // All depths: C(4,1) + C(4,2) + C(4,3) + C(4,4) = 4 + 6 + 4 + 1 = 15
    let sequential: Vec<_> = CombinationIterator::new(&features, 4).collect();
    assert_eq!(sequential.len(), 15);

    for offset in 0..=15 {
        let seekable: Vec<_> =
            SeekableCombinationIterator::starting_at(&features, 4, offset as u128).collect();

        if offset < 15 {
            assert_eq!(seekable, sequential[offset..].to_vec());
        } else {
            assert!(seekable.is_empty());
        }
    }
}

// =============================================================================
// Rank/Unrank roundtrip at scale
// =============================================================================

#[test]
fn rank_unrank_roundtrip_n50_k3_all() {
    // C(50, 3) = 19600
    let n = 50;
    let k = 3;
    let count = combinations_for_depth(n, k);
    assert_eq!(count, 19600);

    for rank in 0..count {
        let combo = unrank_combination(rank, n, k);
        assert_eq!(combo.len(), k);

        let reranked = rank_combination(&combo, n);
        assert_eq!(rank, reranked, "Roundtrip failed at rank {}", rank);

        // Verify strictly increasing
        for i in 1..k {
            assert!(combo[i] > combo[i - 1]);
        }
        // Verify bounds
        assert!(combo[k - 1] < n);
    }
}

#[test]
fn rank_unrank_roundtrip_n100_k2_all() {
    // C(100, 2) = 4950
    let n = 100;
    let k = 2;
    let count = combinations_for_depth(n, k);
    assert_eq!(count, 4950);

    for rank in 0..count {
        let combo = unrank_combination(rank, n, k);
        let reranked = rank_combination(&combo, n);
        assert_eq!(rank, reranked);
    }
}

#[test]
fn rank_unrank_roundtrip_boundaries() {
    // Test at boundary values for various n/k
    let test_cases = [(10, 3), (20, 4), (30, 5), (50, 2), (100, 3), (200, 2)];

    for (n, k) in test_cases {
        let count = combinations_for_depth(n, k);

        // Test first, middle, last
        for rank in [0, count / 2, count - 1] {
            let combo = unrank_combination(rank, n, k);
            let reranked = rank_combination(&combo, n);
            assert_eq!(rank, reranked, "Failed at n={}, k={}, rank={}", n, k, rank);
        }
    }
}

// =============================================================================
// Global index tests
// =============================================================================

#[test]
fn global_index_covers_all_depths() {
    let n = 6;
    let max_depth = 3;

    // Total: C(6,1) + C(6,2) + C(6,3) = 6 + 15 + 20 = 41
    let total = total_combinations(n, max_depth);
    assert_eq!(total, 41);

    let mut seen_depths = HashSet::new();

    for global in 0..total {
        let (depth, local) = global_to_depth_and_local(global, n, max_depth).unwrap();
        seen_depths.insert(depth);

        // Verify local index is valid for this depth
        assert!(local < combinations_for_depth(n, depth));

        // Verify unrank_global works
        let combo = unrank_global(global, n, max_depth).unwrap();
        assert_eq!(combo.len(), depth);
    }

    // Should have seen all depths
    assert_eq!(seen_depths, HashSet::from([1, 2, 3]));
}

#[test]
fn global_index_depth_boundaries() {
    let n = 10;
    let max_depth = 4;

    // Depth 1: 0-9 (10 combos)
    // Depth 2: 10-54 (45 combos)
    // Depth 3: 55-174 (120 combos)
    // Depth 4: 175-384 (210 combos)

    assert_eq!(global_to_depth_and_local(0, n, max_depth), Some((1, 0)));
    assert_eq!(global_to_depth_and_local(9, n, max_depth), Some((1, 9)));
    assert_eq!(global_to_depth_and_local(10, n, max_depth), Some((2, 0)));
    assert_eq!(global_to_depth_and_local(54, n, max_depth), Some((2, 44)));
    assert_eq!(global_to_depth_and_local(55, n, max_depth), Some((3, 0)));
    assert_eq!(global_to_depth_and_local(174, n, max_depth), Some((3, 119)));
    assert_eq!(global_to_depth_and_local(175, n, max_depth), Some((4, 0)));
    assert_eq!(global_to_depth_and_local(384, n, max_depth), Some((4, 209)));
    assert_eq!(global_to_depth_and_local(385, n, max_depth), None);
}

// =============================================================================
// Uniqueness tests
// =============================================================================

#[test]
fn all_combinations_unique_n8_k3() {
    let n = 8;
    let k = 3;
    let count = combinations_for_depth(n, k) as usize;

    let mut seen = HashSet::new();
    for rank in 0..count {
        let combo = unrank_combination(rank as u128, n, k);
        let key = format!("{:?}", combo);
        assert!(
            seen.insert(key.clone()),
            "Duplicate combination at rank {}: {}",
            rank,
            key
        );
    }

    assert_eq!(seen.len(), count);
}

#[test]
fn seekable_iterator_yields_unique_combinations() {
    let features = make_features(8);
    let combos: Vec<_> = SeekableCombinationIterator::starting_at(&features, 3, 0).collect();

    let mut seen = HashSet::new();
    for combo in &combos {
        let names: Vec<_> = combo.iter().map(|f| &f.name).collect();
        let key = format!("{:?}", names);
        assert!(seen.insert(key), "Duplicate combination found");
    }

    // C(8,1) + C(8,2) + C(8,3) = 8 + 28 + 56 = 92
    assert_eq!(seen.len(), 92);
}

// =============================================================================
// Batcher tests
// =============================================================================

#[test]
fn batcher_collects_all_combinations() {
    let features = make_features(8);
    let pools = FeaturePools::new(features.clone());

    let mut batcher = CombinationBatcher::new(&pools, 2, 0);
    let mut all_combos = Vec::new();
    while let Some(batch) = batcher.next_batch(10) {
        all_combos.extend(batch);
    }

    // C(8,1) + C(8,2) = 8 + 28 = 36
    assert_eq!(all_combos.len(), 36);

    // Compare with original iterator
    let original: Vec<_> = CombinationIterator::new(&features, 2).collect();
    assert_eq!(all_combos, original);
}

#[test]
fn batcher_with_various_offsets() {
    let features = make_features(10);
    let pools = FeaturePools::new(features.clone());
    let original: Vec<_> = CombinationIterator::new(&features, 2).collect();
    let total = original.len();

    for offset in [0, 1, 5, 10, 25, 50, total - 1] {
        if offset >= total {
            continue;
        }

        let mut batcher = CombinationBatcher::new(&pools, 2, offset as u64);
        let mut collected = Vec::new();
        while let Some(batch) = batcher.next_batch(7) {
            collected.extend(batch);
        }

        assert_eq!(
            collected,
            original[offset..].to_vec(),
            "Mismatch at offset {}",
            offset
        );
    }
}

#[test]
fn batcher_with_various_batch_sizes() {
    let features = make_features(6);
    let pools = FeaturePools::new(features.clone());
    let original: Vec<_> = CombinationIterator::new(&features, 3).collect();

    for batch_size in [1, 2, 3, 5, 10, 20, 50, 100] {
        let mut batcher = CombinationBatcher::new(&pools, 3, 0);
        let mut collected = Vec::new();
        while let Some(batch) = batcher.next_batch(batch_size) {
            assert!(batch.len() <= batch_size);
            collected.extend(batch);
        }

        assert_eq!(
            collected, original,
            "Mismatch with batch_size {}",
            batch_size
        );
    }
}

// =============================================================================
// Performance verification (timing-based)
// =============================================================================

#[test]
fn seekable_startup_is_fast_for_large_offset() {
    let features = make_features(1000);

    // C(1000, 1) + C(1000, 2) = 1000 + 499500 = 500500
    // Skip to near the end
    let offset = 400_000u128;

    let start = Instant::now();
    let mut iter = SeekableCombinationIterator::starting_at(&features, 2, offset);
    let creation_time = start.elapsed();

    // Should be very fast (sub-millisecond for O(k) unranking)
    assert!(
        creation_time.as_millis() < 50,
        "Iterator creation took {}ms, expected <50ms",
        creation_time.as_millis()
    );

    // Verify we can still iterate
    let combo = iter.next().expect("Should have combo");
    assert_eq!(combo.len(), 2);
}

#[test]
fn unrank_is_fast_for_large_n() {
    // C(10000, 3) = 166,616,670,000
    let n = 10000;
    let k = 3;
    let rank = 83_308_335_000u128; // Middle

    let start = Instant::now();
    let combo = unrank_combination(rank, n, k);
    let elapsed = start.elapsed();

    assert_eq!(combo.len(), k);
    // Allow more time in debug builds (release builds are much faster)
    assert!(
        elapsed.as_millis() < 1000,
        "Unrank took {}ms, expected <1000ms",
        elapsed.as_millis()
    );

    // Verify roundtrip
    let reranked = rank_combination(&combo, n);
    assert_eq!(rank, reranked);
}

// =============================================================================
// Edge cases
// =============================================================================

#[test]
fn empty_features_handling() {
    let features: Vec<FeatureDescriptor> = vec![];
    let pools = FeaturePools::new(features.clone());

    // Iterator should be immediately exhausted
    let mut iter = SeekableCombinationIterator::starting_at(&features, 2, 0);
    assert!(iter.next().is_none());

    // Batcher should return None
    let mut batcher = CombinationBatcher::new(&pools, 2, 0);
    assert!(batcher.next_batch(10).is_none());
}

#[test]
fn single_feature_handling() {
    let features = make_features(1);
    let pools = FeaturePools::new(features.clone());

    // Should only have one depth-1 combination
    let combos: Vec<_> = SeekableCombinationIterator::starting_at(&features, 5, 0).collect();
    assert_eq!(combos.len(), 1);
    assert_eq!(combos[0].len(), 1);
    assert_eq!(combos[0][0].name, "f0");

    // Batcher should work too
    let mut batcher = CombinationBatcher::new(&pools, 5, 0);
    let batch = batcher.next_batch(10).expect("Should have batch");
    assert_eq!(batch.len(), 1);
    assert!(batcher.next_batch(10).is_none());
}

#[test]
fn max_depth_larger_than_n() {
    let features = make_features(3);

    // max_depth = 10, but n = 3, so should cap at depth 3
    let combos: Vec<_> = SeekableCombinationIterator::starting_at(&features, 10, 0).collect();

    // C(3,1) + C(3,2) + C(3,3) = 3 + 3 + 1 = 7
    assert_eq!(combos.len(), 7);

    // Verify depths present
    let depth_1: Vec<_> = combos.iter().filter(|c| c.len() == 1).collect();
    let depth_2: Vec<_> = combos.iter().filter(|c| c.len() == 2).collect();
    let depth_3: Vec<_> = combos.iter().filter(|c| c.len() == 3).collect();

    assert_eq!(depth_1.len(), 3);
    assert_eq!(depth_2.len(), 3);
    assert_eq!(depth_3.len(), 1);
}

#[test]
fn offset_at_exact_total() {
    let features = make_features(5);
    let total = total_combinations(5, 2); // 5 + 10 = 15

    // At exact total: should be empty
    let mut iter = SeekableCombinationIterator::starting_at(&features, 2, total);
    assert!(iter.next().is_none());

    // One before total: should have one combo
    let mut iter = SeekableCombinationIterator::starting_at(&features, 2, total - 1);
    let combo = iter.next().expect("Should have one combo");
    assert_eq!(combo.len(), 2);
    assert!(iter.next().is_none());
}

#[test]
fn offset_way_beyond_total() {
    let features = make_features(5);

    // Way beyond total
    let mut iter = SeekableCombinationIterator::starting_at(&features, 2, 1_000_000);
    assert!(iter.next().is_none());

    // Current index should be None
    assert_eq!(iter.current_global_index(), None);
}

// =============================================================================
// Depth transition tests
// =============================================================================

#[test]
fn depth_transition_at_boundary() {
    let features = make_features(5);

    // Start at last depth-1 (index 4)
    let mut iter = SeekableCombinationIterator::starting_at(&features, 3, 4);

    let combo = iter.next().expect("Should have combo");
    assert_eq!(combo.len(), 1);
    assert_eq!(combo[0].name, "f4");

    // Next should be first depth-2
    let combo = iter.next().expect("Should have combo");
    assert_eq!(combo.len(), 2);
    assert_eq!(combo[0].name, "f0");
    assert_eq!(combo[1].name, "f1");
}

#[test]
fn depth_transition_multiple() {
    let features = make_features(3);

    // Collect all and verify depth transitions
    let combos: Vec<_> = SeekableCombinationIterator::starting_at(&features, 3, 0).collect();

    // Should be: [0], [1], [2], [0,1], [0,2], [1,2], [0,1,2]
    assert_eq!(combos.len(), 7);

    // Verify structure
    assert_eq!(combos[0].len(), 1);
    assert_eq!(combos[1].len(), 1);
    assert_eq!(combos[2].len(), 1);
    assert_eq!(combos[3].len(), 2);
    assert_eq!(combos[4].len(), 2);
    assert_eq!(combos[5].len(), 2);
    assert_eq!(combos[6].len(), 3);
}

// =============================================================================
// Consistency with different iteration patterns
// =============================================================================

#[test]
fn seekable_at_every_position_matches_skip() {
    let features = make_features(6);
    let all_combos: Vec<_> = CombinationIterator::new(&features, 2).collect();

    for (i, combo) in all_combos.iter().enumerate() {
        // Get combo by seeking
        let mut iter = SeekableCombinationIterator::starting_at(&features, 2, i as u128);
        let seeked = iter.next().expect("Should have combo");

        // Should match
        assert_eq!(seeked, *combo, "Mismatch at position {}", i);
    }
}

#[test]
fn current_global_index_is_accurate() {
    let features = make_features(5);
    let mut iter = SeekableCombinationIterator::starting_at(&features, 2, 0);

    let total = total_combinations(5, 2);
    for expected in 0..total {
        let actual = iter.current_global_index();
        assert_eq!(actual, Some(expected), "Wrong index at step {}", expected);
        iter.next();
    }

    assert_eq!(iter.current_global_index(), None);
}

#[test]
fn current_global_index_after_seek() {
    let features = make_features(10);

    for start in [0u128, 5, 10, 25, 50] {
        let iter = SeekableCombinationIterator::starting_at(&features, 2, start);

        let idx = iter.current_global_index();
        assert_eq!(
            idx,
            Some(start),
            "Wrong initial index after seeking to {}",
            start
        );
    }
}

// =============================================================================
// Stress tests
// =============================================================================

#[test]
fn stress_many_small_seeks() {
    let features = make_features(20);
    let total = total_combinations(20, 2) as usize; // 20 + 190 = 210

    // Seek to many random-ish positions and verify
    for offset in (0..total).step_by(7) {
        let mut iter = SeekableCombinationIterator::starting_at(&features, 2, offset as u128);

        // Get a few combos and verify they're valid
        for _ in 0..3 {
            if let Some(combo) = iter.next() {
                assert!(combo.len() <= 2);
                for f in &combo {
                    assert!(f.name.starts_with("f"));
                }
            }
        }
    }
}

#[test]
fn stress_depth_3_medium_n() {
    let features = make_features(15);

    // C(15,1) + C(15,2) + C(15,3) = 15 + 105 + 455 = 575
    let total = total_combinations(15, 3);
    assert_eq!(total, 575);

    // Seek to various positions
    for offset in [0, 14, 15, 119, 120, 500, 574] {
        let combos: Vec<_> =
            SeekableCombinationIterator::starting_at(&features, 3, offset).collect();

        assert_eq!(combos.len(), (total - offset) as usize);
    }
}
