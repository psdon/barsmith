use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeatureCategory {
    Boolean,
    Continuous,
    /// Feature-vs-constant comparison (scalar threshold, e.g. rsi_14>20).
    FeatureVsConstant,
    /// Feature-vs-feature comparison (pairwise numeric predicate, e.g. 9ema>200sma).
    FeatureVsFeature,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FeatureDescriptor {
    pub name: String,
    pub category: FeatureCategory,
    pub note: String,
}

impl FeatureDescriptor {
    pub fn new(
        name: impl Into<String>,
        category: FeatureCategory,
        note: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            category,
            note: note.into(),
        }
    }

    pub fn boolean(name: &str, note: &str) -> Self {
        Self::new(name, FeatureCategory::Boolean, note)
    }

    pub fn comparison(name: impl Into<String>, note: impl Into<String>) -> Self {
        Self::new(name, FeatureCategory::FeatureVsConstant, note)
    }

    /// Descriptor for feature-vs-constant scalar comparisons (e.g. rsi_14>20).
    pub fn feature_vs_constant(name: impl Into<String>, note: impl Into<String>) -> Self {
        Self::new(name, FeatureCategory::FeatureVsConstant, note)
    }

    /// Descriptor for feature-vs-feature numeric comparisons (e.g. 9ema>200sma).
    pub fn feature_vs_feature(name: impl Into<String>, note: impl Into<String>) -> Self {
        Self::new(name, FeatureCategory::FeatureVsFeature, note)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    GreaterEqual,
    LessEqual,
}

#[derive(Clone, Debug)]
pub struct ComparisonSpec {
    pub base_feature: String,
    pub operator: ComparisonOperator,
    /// Threshold for feature-to-constant comparisons. When `rhs_feature` is
    /// `Some`, this may be `None`.
    pub threshold: Option<f64>,
    /// Optional right-hand side feature for feature-to-feature comparisons.
    /// When `rhs_feature` is `Some`, the comparison is evaluated as
    /// `base_feature (op) rhs_feature`.
    pub rhs_feature: Option<String>,
}

impl ComparisonSpec {
    pub fn threshold(
        base_feature: impl Into<String>,
        operator: ComparisonOperator,
        threshold: f64,
    ) -> Self {
        Self {
            base_feature: base_feature.into(),
            operator,
            threshold: Some(threshold),
            rhs_feature: None,
        }
    }

    pub fn pair(
        left_feature: impl Into<String>,
        operator: ComparisonOperator,
        right_feature: impl Into<String>,
    ) -> Self {
        Self {
            base_feature: left_feature.into(),
            operator,
            threshold: None,
            rhs_feature: Some(right_feature.into()),
        }
    }
}

fn operator_symbol(op: ComparisonOperator) -> &'static str {
    match op {
        ComparisonOperator::GreaterThan => ">",
        ComparisonOperator::LessThan => "<",
        ComparisonOperator::GreaterEqual => ">=",
        ComparisonOperator::LessEqual => "<=",
    }
}

/// Generate generic feature-to-feature comparison descriptors and specs.
///
/// Given a list of numeric feature names and comparison operators, this helper
/// produces virtual boolean features such as `sma200>close` or `close>9ema`.
/// These can be treated like any other boolean feature in the permutation
/// engine.
///
/// `max_pairs` limits the total number of comparison conditions emitted
/// (counting each (left, op, right) triple as one condition); when `None`,
/// all ordered pairs are included.
pub fn generate_feature_comparisons(
    features: &[&str],
    operators: &[ComparisonOperator],
    max_pairs: Option<usize>,
    note: &str,
) -> (
    Vec<FeatureDescriptor>,
    std::collections::HashMap<String, ComparisonSpec>,
) {
    use std::collections::HashMap;

    let mut descriptors = Vec::new();
    let mut specs = HashMap::new();
    let mut emitted = 0usize;
    let limit = max_pairs.unwrap_or(usize::MAX);

    for &left in features {
        for &right in features {
            if left == right {
                continue;
            }
            for &op in operators {
                if emitted >= limit {
                    return (descriptors, specs);
                }
                let symbol = operator_symbol(op);
                let name = format!("{left}{symbol}{right}");
                if specs.contains_key(&name) {
                    continue;
                }
                let descriptor = FeatureDescriptor::feature_vs_feature(name.clone(), note);
                let spec = ComparisonSpec::pair(left, op, right);
                descriptors.push(descriptor);
                specs.insert(name, spec);
                emitted += 1;
            }
        }
    }

    (descriptors, specs)
}

/// Generate unordered feature-to-feature comparison descriptors and specs.
///
/// This variant treats each feature pair as an *unordered* set, emitting
/// comparisons only for the canonical (left, right) ordering to avoid
/// redundant inverted counterparts. For example, given features
/// ["9ema", "200sma"] and operators [">", "<="], this helper will emit
/// "9ema>200sma" and "9ema<=200sma" but not "200sma<9ema" / "200sma>=9ema",
/// since those are logically equivalent and would just bloat the search
/// space.
pub fn generate_unordered_feature_comparisons(
    features: &[&str],
    operators: &[ComparisonOperator],
    max_pairs: Option<usize>,
    note: &str,
) -> (Vec<FeatureDescriptor>, HashMap<String, ComparisonSpec>) {
    let mut descriptors = Vec::new();
    let mut specs = HashMap::new();
    let mut emitted = 0usize;
    let limit = max_pairs.unwrap_or(usize::MAX);

    for (i, &left) in features.iter().enumerate() {
        for &right in features.iter().skip(i + 1) {
            for &op in operators {
                if emitted >= limit {
                    return (descriptors, specs);
                }
                let symbol = operator_symbol(op);
                let name = format!("{left}{symbol}{right}");
                if specs.contains_key(&name) {
                    continue;
                }
                let descriptor = FeatureDescriptor::feature_vs_feature(name.clone(), note);
                let spec = ComparisonSpec::pair(left, op, right);
                descriptors.push(descriptor);
                specs.insert(name, spec);
                emitted += 1;
            }
        }
    }

    (descriptors, specs)
}
