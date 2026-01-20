use custom_rs::features::FeatureCatalog;

#[test]
fn boolean_catalog_includes_structural_and_regime_flags() {
    let names = FeatureCatalog::boolean_names();

    // Candle / pattern structure
    assert!(
        names.contains(&"is_tribar"),
        "boolean catalog should include is_tribar"
    );
    assert!(
        names.contains(&"consecutive_green_3"),
        "boolean catalog should include consecutive_green_3"
    );
    assert!(
        names.contains(&"bullish_engulfing"),
        "boolean catalog should include bullish_engulfing"
    );

    // KF regime flags
    assert!(
        names.contains(&"kf_trending_volatile"),
        "boolean catalog should include kf_trending_volatile"
    );
    assert!(
        names.contains(&"is_kf_smooth_trend"),
        "boolean catalog should include is_kf_smooth_trend"
    );
}
