//! Maximal Marginal Relevance (MMR) candidate diversification for RAG retrieval.
//!
//! Literature Provenance:
//! - Carbonell & Goldstein (1998): "The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries".
//! - Polzer: "RAG with Python Cookbook", Ch. 7, Recipe 7.6.
//! - Documented in `rag-wiki/features/dartboard-retrieval-and-mmr.md`.
//!
//! MMR balances relevance to the query with diversity among selected candidates:
//!   MMR(d) = argmax_{d in D \ R} [ lambda * rel(d) - (1 - lambda) * max_{r in R} sim(d, r) ]

/// Compute cosine similarity between two float slices.
/// Assumes vectors are already L2 normalized (common in ONNX/embedding pipelines).
/// Falls back safely to normalized dot product if magnitudes differ.
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom > 1e-9 {
        (dot / denom).clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

/// A candidate item with relevance score and optional embedding vector for MMR diversification.
#[derive(Debug, Clone)]
pub struct MmrCandidate<T> {
    pub item: T,
    pub score: f32,
    pub embedding: Option<Vec<f32>>,
}

/// Diversify candidate pool using Maximal Marginal Relevance.
///
/// Parameters:
/// - `candidates`: Pool of items ordered or scored by initial retrieval relevance.
/// - `target_k`: Maximum number of diversified results to select.
/// - `lambda`: Diversity parameter in [0.0, 1.0].
///   - `1.0`: Pure relevance ranking (no diversity penalty).
///   - `0.0`: Maximal diversity (penalizes similarity to already selected items maximally).
///   - `0.7`: Recommended balanced default from RAG literature.
pub fn mmr_diversify<T: Clone>(
    candidates: &[MmrCandidate<T>],
    target_k: usize,
    lambda: f32,
) -> Vec<T> {
    if candidates.is_empty() || target_k == 0 {
        return Vec::new();
    }
    if target_k >= candidates.len() && lambda >= 0.999 {
        return candidates.iter().map(|c| c.item.clone()).collect();
    }

    let lambda = lambda.clamp(0.0, 1.0);
    let mut unselected: Vec<usize> = (0..candidates.len()).collect();
    let mut selected_indices: Vec<usize> = Vec::with_capacity(target_k.min(candidates.len()));

    // Normalize relevance scores to [0.0, 1.0] for fair balancing with cosine similarity [-1, 1] -> [0, 1]
    let max_score = candidates
        .iter()
        .map(|c| c.score)
        .fold(f32::NEG_INFINITY, f32::max);
    let min_score = candidates
        .iter()
        .map(|c| c.score)
        .fold(f32::INFINITY, f32::min);
    let score_range = if (max_score - min_score).abs() > 1e-6 {
        max_score - min_score
    } else {
        1.0
    };

    let norm_rel = |score: f32| -> f32 {
        if score_range > 0.0 {
            ((score - min_score) / score_range).clamp(0.0, 1.0)
        } else {
            1.0
        }
    };

    while selected_indices.len() < target_k && !unselected.is_empty() {
        let mut best_idx_in_unselected = 0;
        let mut best_mmr_score = f32::NEG_INFINITY;

        for (pos, &cand_idx) in unselected.iter().enumerate() {
            let cand = &candidates[cand_idx];
            let rel = norm_rel(cand.score);

            let max_sim = if selected_indices.is_empty() {
                0.0
            } else {
                let mut max_s = 0.0f32;
                if let Some(cand_emb) = &cand.embedding {
                    for &sel_idx in &selected_indices {
                        if let Some(sel_emb) = &candidates[sel_idx].embedding {
                            let sim = cosine_similarity(cand_emb, sel_emb).max(0.0);
                            if sim > max_s {
                                max_s = sim;
                            }
                        }
                    }
                }
                max_s
            };

            let mmr_val = lambda * rel - (1.0 - lambda) * max_sim;
            if mmr_val > best_mmr_score {
                best_mmr_score = mmr_val;
                best_idx_in_unselected = pos;
            }
        }

        let chosen_cand_idx = unselected.remove(best_idx_in_unselected);
        selected_indices.push(chosen_cand_idx);
    }

    selected_indices
        .into_iter()
        .map(|idx| candidates[idx].item.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-5);

        let orthogonal = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &orthogonal) - 0.0).abs() < 1e-5);

        let opposite = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &opposite) - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_mmr_diversify_empty_and_zero_k() {
        let empty: Vec<MmrCandidate<String>> = vec![];
        let res = mmr_diversify(&empty, 5, 0.7);
        assert!(res.is_empty());

        let cands = vec![MmrCandidate {
            item: "a".to_string(),
            score: 1.0,
            embedding: Some(vec![1.0, 0.0]),
        }];
        let res0 = mmr_diversify(&cands, 0, 0.7);
        assert!(res0.is_empty());
    }

    #[test]
    fn test_mmr_diversifies_away_from_duplicate_chunks() {
        // Candidate 0: query match A (score 1.0, direction [1, 0])
        // Candidate 1: duplicate of A with high score (score 0.95, direction [0.99, 0.01])
        // Candidate 2: distinct code B (score 0.80, direction [0, 1])
        let cands = vec![
            MmrCandidate {
                item: "doc_A1".to_string(),
                score: 1.0,
                embedding: Some(vec![1.0, 0.0]),
            },
            MmrCandidate {
                item: "doc_A2_duplicate".to_string(),
                score: 0.95,
                embedding: Some(vec![0.999, 0.001]),
            },
            MmrCandidate {
                item: "doc_B_distinct".to_string(),
                score: 0.80,
                embedding: Some(vec![0.0, 1.0]),
            },
        ];

        // With pure relevance (lambda = 1.0), order is A1, A2, B
        let pure_rel = mmr_diversify(&cands, 2, 1.0);
        assert_eq!(pure_rel, vec!["doc_A1", "doc_A2_duplicate"]);

        // With diversity (lambda = 0.5), B is picked over duplicate A2
        let diversified = mmr_diversify(&cands, 2, 0.5);
        assert_eq!(diversified, vec!["doc_A1", "doc_B_distinct"]);
    }
}
