//! BM25-lite scoring for tool_search.

pub fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|t| !t.is_empty())
        .map(str::to_string)
        .collect()
}

pub fn bm25_scores(query: &str, documents: &[String]) -> Vec<f64> {
    let q_terms = tokenize(query);
    if q_terms.is_empty() || documents.is_empty() {
        return vec![0.0; documents.len()];
    }

    let doc_terms: Vec<Vec<String>> = documents.iter().map(|d| tokenize(d)).collect();
    let n = documents.len() as f64;
    let avgdl = doc_terms.iter().map(|t| t.len() as f64).sum::<f64>() / n.max(1.0);

    let mut df = std::collections::HashMap::new();
    for terms in &doc_terms {
        let mut seen = std::collections::HashSet::new();
        for t in terms {
            if seen.insert(t.clone()) {
                *df.entry(t.clone()).or_insert(0usize) += 1;
            }
        }
    }

    let k1 = 1.5;
    let b = 0.75;

    doc_terms
        .iter()
        .map(|terms| {
            let dl = terms.len() as f64;
            let mut score = 0.0;
            for qt in &q_terms {
                let tf = terms.iter().filter(|t| *t == qt).count() as f64;
                if tf == 0.0 {
                    continue;
                }
                let df_q = *df.get(qt).unwrap_or(&0) as f64;
                let idf = ((n - df_q + 0.5) / (df_q + 0.5) + 1.0).ln();
                score += idf * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * dl / avgdl.max(1.0)));
            }
            score
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranks_github_higher() {
        let docs = vec![
            "filesystem read write grep".into(),
            "github pull request mcp server".into(),
        ];
        let scores = bm25_scores("github mcp", &docs);
        assert!(scores[1] > scores[0]);
    }

    #[test]
    fn empty_query_scores_everything_zero() {
        let docs = vec!["github mcp server".into(), "filesystem tools".into()];
        assert_eq!(bm25_scores("", &docs), vec![0.0, 0.0]);
        assert_eq!(bm25_scores("   ", &docs), vec![0.0, 0.0]);
    }

    #[test]
    fn punctuation_only_query_has_no_terms() {
        // Tokenize drops non-alphanumeric characters, so a query made only of
        // punctuation/symbols reduces to zero terms — matches the empty-query case.
        let docs = vec!["github mcp server".into()];
        assert_eq!(bm25_scores("!!! ??? ---", &docs), vec![0.0]);
    }

    #[test]
    fn no_documents_returns_empty_scores() {
        assert_eq!(bm25_scores("github", &[]), Vec::<f64>::new());
    }
}
