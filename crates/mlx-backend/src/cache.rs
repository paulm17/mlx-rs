use std::collections::HashMap;

use crate::llama::LayerCache;

#[derive(Clone)]
struct TrieNode {
    children: HashMap<i32, TrieNode>,
    caches: Option<Vec<LayerCache>>,
    token_count: usize,
}

impl TrieNode {
    fn new() -> Self {
        Self {
            children: HashMap::new(),
            caches: None,
            token_count: 0,
        }
    }
}

pub struct PrefixCache {
    root: TrieNode,
    #[allow(dead_code)]
    num_layers: usize,
    #[allow(dead_code)]
    max_entries: usize,
}

impl PrefixCache {
    pub fn new(num_layers: usize, max_entries: usize) -> Self {
        Self {
            root: TrieNode::new(),
            num_layers,
            max_entries,
        }
    }

    pub fn find(&self, tokens: &[i32]) -> (usize, Option<&Vec<LayerCache>>) {
        let mut node = &self.root;
        let mut matched = 0;
        let mut last_cache: Option<&Vec<LayerCache>> = None;

        for &tok in tokens {
            if let Some(child) = node.children.get(&tok) {
                node = child;
                matched += 1;
                if node.caches.is_some() {
                    last_cache = node.caches.as_ref();
                }
            } else {
                break;
            }
        }

        (matched, last_cache)
    }

    pub fn insert(&mut self, tokens: &[i32], caches: Vec<LayerCache>) {
        let mut node = &mut self.root;
        for &tok in tokens {
            node = node.children.entry(tok).or_insert_with(TrieNode::new);
        }
        node.caches = Some(caches);
        node.token_count = tokens.len();
    }

    pub fn clear(&mut self) {
        self.root = TrieNode::new();
    }

    pub fn entry_count(&self) -> usize {
        fn count_nodes(node: &TrieNode) -> usize {
            let mut count = if node.caches.is_some() { 1 } else { 0 };
            for child in node.children.values() {
                count += count_nodes(child);
            }
            count
        }
        count_nodes(&self.root)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llama::KvCache;

    fn make_test_caches(n: usize) -> Vec<LayerCache> {
        (0..n).map(|_| LayerCache::Attention(KvCache::new())).collect()
    }

    #[test]
    fn test_prefix_cache_basic() {
        let mut cache = PrefixCache::new(2, 10);
        let tokens = vec![1, 2, 3];
        cache.insert(&tokens, make_test_caches(2));

        let (matched, _) = cache.find(&[1, 2, 3, 4]);
        assert_eq!(matched, 3);

        let (matched, _) = cache.find(&[1, 2]);
        assert_eq!(matched, 2);

        let (matched, _) = cache.find(&[9, 10]);
        assert_eq!(matched, 0);
    }

    #[test]
    fn test_prefix_cache_no_match() {
        let cache = PrefixCache::new(2, 10);
        let (matched, kv) = cache.find(&[1, 2, 3]);
        assert_eq!(matched, 0);
        assert!(kv.is_none());
    }

    #[test]
    fn test_prefix_cache_partial_match() {
        let mut cache = PrefixCache::new(2, 10);
        cache.insert(&[1, 2, 3], make_test_caches(2));

        let (matched, _) = cache.find(&[1, 2, 9]);
        assert_eq!(matched, 2);
    }

    #[test]
    fn test_prefix_cache_multiple_entries() {
        let mut cache = PrefixCache::new(2, 10);
        cache.insert(&[1, 2, 3], make_test_caches(2));
        cache.insert(&[1, 2, 4], make_test_caches(2));

        assert_eq!(cache.entry_count(), 2);

        let (m1, _) = cache.find(&[1, 2, 3]);
        assert_eq!(m1, 3);
        let (m2, _) = cache.find(&[1, 2, 4]);
        assert_eq!(m2, 3);
    }

    #[test]
    fn test_prefix_cache_clear() {
        let mut cache = PrefixCache::new(2, 10);
        cache.insert(&[1, 2, 3], make_test_caches(2));
        assert_eq!(cache.entry_count(), 1);

        cache.clear();
        assert_eq!(cache.entry_count(), 0);
    }
}
