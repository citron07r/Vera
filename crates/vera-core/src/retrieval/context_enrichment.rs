//! Small-to-big context enrichment for RAG retrieval.
//!
//! Literature Provenance:
//! - Mendelevitch: "Hands-On RAG for Production", Ch. 2 ("The Base RAG Stack" - Parent-Child Chunking).
//! - Nolan: "RAG in Practice", Ch. 5 ("Chunking Strategies: How to Split Documents Without Losing Meaning").
//! - Polzer: "RAG with Python Cookbook", Ch. 7, Recipe 7.5 ("Auto-Merging & Sentence Window Retrieval").
//! - Documented in `rag-wiki/features/context-enrichment-and-hierarchical-indices.md`.
//!
//! Enriches leaf-level search results with enclosing architectural context (class names,
//! trait/struct declarations, parent scope paths, or module definitions) to eliminate
//! fragmented understanding in LLM and coding agent prompts.

use crate::types::SearchResult;

/// Search result enriched with parent scope and structural declaration context.
#[derive(Debug, Clone)]
pub struct EnrichedSearchResult {
    /// The primary matched leaf chunk result.
    pub result: SearchResult,
    /// Extracted or resolved parent scope (e.g. `crate::retrieval::hybrid::Searcher`).
    pub parent_scope: Option<String>,
    /// Enclosing signature or definition (e.g. `impl SearchService for ...`).
    pub enclosing_signature: Option<String>,
}

impl EnrichedSearchResult {
    /// Format the chunk content for agent prompt consumption, attaching parent breadcrumbs.
    pub fn formatted_content(&self) -> String {
        let mut header = String::new();
        if let Some(scope) = &self.parent_scope {
            header.push_str(&format!("// Scope: {}\n", scope));
        }
        if let Some(sig) = &self.enclosing_signature {
            header.push_str(&format!("// Enclosing: {}\n", sig));
        }

        if header.is_empty() {
            self.result.content.clone()
        } else {
            format!("{}{}", header, self.result.content)
        }
    }
}

/// Heuristically extract parent scope from symbol name, file path, or content prefixes.
pub fn infer_parent_scope(result: &SearchResult) -> Option<String> {
    if let Some(name) = &result.symbol_name {
        if let Some((parent, _child)) = name.rsplit_once("::") {
            return Some(parent.to_string());
        }
        if let Some((parent, _child)) = name.rsplit_once('.') {
            return Some(parent.to_string());
        }
    }

    // Derive module scope from file path (e.g., "crates/vera-core/src/retrieval/hybrid.rs" -> "retrieval::hybrid")
    let path = &result.file_path;
    if let Some(src_idx) = path.find("src/") {
        let after_src = &path[src_idx + 4..];
        let module_path = after_src
            .trim_end_matches(".rs")
            .trim_end_matches("/mod")
            .replace('/', "::");
        if !module_path.is_empty() {
            return Some(module_path);
        }
    }

    None
}

/// Enrich a collection of search results with structural parent context.
pub fn enrich_search_results(
    results: Vec<SearchResult>,
    default_enclosing_sig: Option<&str>,
) -> Vec<EnrichedSearchResult> {
    results
        .into_iter()
        .map(|r| {
            let parent_scope = infer_parent_scope(&r);
            let enclosing_signature = default_enclosing_sig.map(|s| s.to_string());
            EnrichedSearchResult {
                result: r,
                parent_scope,
                enclosing_signature,
            }
        })
        .collect()
}

/// Auto-merge adjacent or co-scoped sibling methods into a unified enclosing context.
///
/// Provenance: Polzer Ch. 7, Recipe 7.5 ("Auto-Merging Retriever").
/// When multiple retrieved chunks share the exact same `(file_path, parent_scope)`,
/// returning them separately fragments the agent's view. This function groups sibling
/// chunks occurring in the same container.
///
/// If `results` contains $\ge 2$ entries sharing `(file_path, parent_scope)`, they are
/// coalesced:
/// - `line_start` becomes the minimum of the siblings.
/// - `line_end` becomes the maximum of the siblings.
/// - `score` becomes the maximum score among the siblings.
/// - `content` joins the sibling contents separated by double newlines with a structural boundary marker.
pub fn auto_merge_hierarchical_results(
    results: Vec<EnrichedSearchResult>,
    min_siblings_to_merge: usize,
) -> Vec<EnrichedSearchResult> {
    if results.len() < 2 || min_siblings_to_merge <= 1 {
        return results;
    }

    use std::collections::HashMap;

    // Count occurrences of each (file_path, parent_scope) key
    let mut scope_counts: HashMap<(String, String), usize> = HashMap::new();
    for r in &results {
        if let Some(scope) = &r.parent_scope {
            *scope_counts
                .entry((r.result.file_path.clone(), scope.clone()))
                .or_insert(0) += 1;
        }
    }

    let mut merged: Vec<EnrichedSearchResult> = Vec::new();
    let mut seen_merged_scopes: HashMap<(String, String), usize> = HashMap::new();

    for item in results {
        let key = item
            .parent_scope
            .as_ref()
            .map(|s| (item.result.file_path.clone(), s.clone()));

        if let Some(ref k) = key
            && scope_counts.get(k).copied().unwrap_or(0) >= min_siblings_to_merge
        {
            if let Some(&target_idx) = seen_merged_scopes.get(k) {
                // Merge into the already created container
                let existing = &mut merged[target_idx];
                existing.result.line_start = existing.result.line_start.min(item.result.line_start);
                existing.result.line_end = existing.result.line_end.max(item.result.line_end);
                if item.result.score > existing.result.score {
                    existing.result.score = item.result.score;
                }
                existing
                    .result
                    .content
                    .push_str("\n\n// ── Sibling Chunk ──\n");
                existing.result.content.push_str(&item.result.content);
            } else {
                let idx = merged.len();
                seen_merged_scopes.insert(k.clone(), idx);
                merged.push(item);
            }
            continue;
        }

        merged.push(item);
    }

    merged
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Language, SymbolType};

    #[test]
    fn test_infer_parent_scope_from_symbol_hierarchy() {
        let r = SearchResult {
            file_path: "crates/vera-core/src/retrieval/hybrid.rs".to_string(),
            line_start: 10,
            line_end: 20,
            content: "fn fuse() {}".to_string(),
            language: Language::Rust,
            score: 1.0,
            symbol_name: Some("HybridSearcher::fuse".to_string()),
            symbol_type: Some(SymbolType::Method),
            part_index: None,
        };

        let scope = infer_parent_scope(&r);
        assert_eq!(scope, Some("HybridSearcher".to_string()));
    }

    #[test]
    fn test_infer_parent_scope_from_file_path() {
        let r = SearchResult {
            file_path: "crates/vera-core/src/retrieval/hybrid.rs".to_string(),
            line_start: 10,
            line_end: 20,
            content: "fn fuse() {}".to_string(),
            language: Language::Rust,
            score: 1.0,
            symbol_name: Some("fuse".to_string()),
            symbol_type: Some(SymbolType::Function),
            part_index: None,
        };

        let scope = infer_parent_scope(&r);
        assert_eq!(scope, Some("retrieval::hybrid".to_string()));
    }

    #[test]
    fn test_formatted_content_with_breadcrumbs() {
        let r = SearchResult {
            file_path: "src/service.rs".to_string(),
            line_start: 15,
            line_end: 25,
            content: "pub fn handle_request() {}".to_string(),
            language: Language::Rust,
            score: 0.9,
            symbol_name: Some("service::handle_request".to_string()),
            symbol_type: Some(SymbolType::Function),
            part_index: None,
        };

        let enriched = EnrichedSearchResult {
            result: r,
            parent_scope: Some("service".to_string()),
            enclosing_signature: Some("impl ApiServer".to_string()),
        };

        let formatted = enriched.formatted_content();
        assert!(formatted.starts_with("// Scope: service\n// Enclosing: impl ApiServer\n"));
        assert!(formatted.contains("pub fn handle_request() {}"));
    }

    #[test]
    fn test_auto_merge_hierarchical_results() {
        let r1 = SearchResult {
            file_path: "src/engine.rs".to_string(),
            line_start: 10,
            line_end: 25,
            content: "fn start() {}".to_string(),
            language: Language::Rust,
            score: 0.8,
            symbol_name: Some("Engine::start".to_string()),
            symbol_type: Some(SymbolType::Method),
            part_index: None,
        };
        let r2 = SearchResult {
            file_path: "src/engine.rs".to_string(),
            line_start: 30,
            line_end: 45,
            content: "fn stop() {}".to_string(),
            language: Language::Rust,
            score: 0.95,
            symbol_name: Some("Engine::stop".to_string()),
            symbol_type: Some(SymbolType::Method),
            part_index: None,
        };
        let r3 = SearchResult {
            file_path: "src/other.rs".to_string(),
            line_start: 1,
            line_end: 5,
            content: "fn helper() {}".to_string(),
            language: Language::Rust,
            score: 0.7,
            symbol_name: Some("helper".to_string()),
            symbol_type: Some(SymbolType::Function),
            part_index: None,
        };

        let enriched = enrich_search_results(vec![r1, r2, r3], None);
        assert_eq!(enriched.len(), 3);

        let merged = auto_merge_hierarchical_results(enriched, 2);
        assert_eq!(merged.len(), 2);

        // The Engine container was merged
        let engine_item = merged
            .iter()
            .find(|m| m.parent_scope.as_deref() == Some("Engine"))
            .unwrap();
        assert_eq!(engine_item.result.line_start, 10);
        assert_eq!(engine_item.result.line_end, 45);
        assert_eq!(engine_item.result.score, 0.95);
        assert!(engine_item.result.content.contains("fn start() {}"));
        assert!(engine_item.result.content.contains("fn stop() {}"));
        assert!(
            engine_item
                .result
                .content
                .contains("// ── Sibling Chunk ──")
        );
    }
}
