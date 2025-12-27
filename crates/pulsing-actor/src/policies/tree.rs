//! Thread-safe multi-tenant radix tree for cache-aware routing
//!
//! This module provides a radix tree implementation that:
//! - Stores data for multiple tenants (workers) with overlapping prefixes
//! - Uses node-level locking for concurrent access
//! - Supports LRU eviction based on tenant access time
//!
//! The tree is used by CacheAwarePolicy to track request history and
//! make routing decisions based on prefix matching.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::debug;

type NodeRef = Arc<Node>;

#[derive(Debug)]
struct Node {
    children: RwLock<HashMap<char, NodeRef>>,
    text: RwLock<String>,
    tenant_last_access_time: RwLock<HashMap<String, u128>>,
    parent: RwLock<Option<NodeRef>>,
}

/// Thread-safe multi-tenant radix tree
///
/// Stores request history for multiple workers (tenants) and supports
/// prefix matching for cache-aware routing decisions.
#[derive(Debug)]
pub struct Tree {
    root: NodeRef,
    pub tenant_char_count: RwLock<HashMap<String, usize>>,
}

// Eviction entry for priority queue
struct EvictionEntry {
    timestamp: u128,
    tenant: String,
    node: NodeRef,
}

impl Eq for EvictionEntry {}

impl PartialOrd for EvictionEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for EvictionEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.timestamp.cmp(&other.timestamp)
    }
}

impl PartialEq for EvictionEntry {
    fn eq(&self, other: &Self) -> bool {
        self.timestamp == other.timestamp
    }
}

// UTF-8 character utilities
fn shared_prefix_count(a: &str, b: &str) -> usize {
    let mut i = 0;
    let mut a_iter = a.chars();
    let mut b_iter = b.chars();

    loop {
        match (a_iter.next(), b_iter.next()) {
            (Some(a_char), Some(b_char)) if a_char == b_char => {
                i += 1;
            }
            _ => break,
        }
    }

    i
}

fn slice_by_chars(s: &str, start: usize, end: usize) -> String {
    s.chars().skip(start).take(end - start).collect()
}

impl Default for Tree {
    fn default() -> Self {
        Self::new()
    }
}

impl Tree {
    /// Create a new empty tree
    pub fn new() -> Self {
        Tree {
            root: Arc::new(Node {
                children: RwLock::new(HashMap::new()),
                text: RwLock::new(String::new()),
                tenant_last_access_time: RwLock::new(HashMap::new()),
                parent: RwLock::new(None),
            }),
            tenant_char_count: RwLock::new(HashMap::new()),
        }
    }

    /// Insert text into tree with given tenant (worker URL)
    pub fn insert(&self, text: &str, tenant: &str) {
        let mut curr = Arc::clone(&self.root);
        let mut curr_idx = 0;

        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();

        // Update root's tenant access time
        {
            let mut access_times = curr.tenant_last_access_time.write().unwrap();
            access_times.insert(tenant.to_string(), timestamp_ms);
        }

        // Initialize tenant char count if not exists
        {
            let mut counts = self.tenant_char_count.write().unwrap();
            counts.entry(tenant.to_string()).or_insert(0);
        }

        let mut prev = Arc::clone(&self.root);
        let text_count = text.chars().count();

        while curr_idx < text_count {
            let first_char = text.chars().nth(curr_idx).unwrap();
            curr = prev;

            let child = {
                let children = curr.children.read().unwrap();
                children.get(&first_char).cloned()
            };

            match child {
                None => {
                    // No match - create new node
                    let curr_text = slice_by_chars(text, curr_idx, text_count);
                    let curr_text_count = curr_text.chars().count();

                    let new_node = Arc::new(Node {
                        children: RwLock::new(HashMap::new()),
                        text: RwLock::new(curr_text),
                        tenant_last_access_time: RwLock::new(HashMap::new()),
                        parent: RwLock::new(Some(Arc::clone(&curr))),
                    });

                    // Update tenant char count
                    {
                        let mut counts = self.tenant_char_count.write().unwrap();
                        counts
                            .entry(tenant.to_string())
                            .and_modify(|count| *count += curr_text_count)
                            .or_insert(curr_text_count);
                    }

                    // Set tenant access time on new node
                    {
                        let mut access_times = new_node.tenant_last_access_time.write().unwrap();
                        access_times.insert(tenant.to_string(), timestamp_ms);
                    }

                    // Add to parent's children
                    {
                        let mut children = curr.children.write().unwrap();
                        children.insert(first_char, Arc::clone(&new_node));
                    }

                    prev = new_node;
                    curr_idx = text_count;
                }
                Some(matched_node) => {
                    let matched_node_text = matched_node.text.read().unwrap().clone();
                    let matched_node_text_count = matched_node_text.chars().count();

                    let curr_text = slice_by_chars(text, curr_idx, text_count);
                    let shared_count = shared_prefix_count(&matched_node_text, &curr_text);

                    if shared_count < matched_node_text_count {
                        // Split the matched node
                        let matched_text = slice_by_chars(&matched_node_text, 0, shared_count);
                        let contracted_text = slice_by_chars(
                            &matched_node_text,
                            shared_count,
                            matched_node_text_count,
                        );
                        let matched_text_count = matched_text.chars().count();

                        // Clone tenant access times from matched node
                        let cloned_access_times = {
                            let times = matched_node.tenant_last_access_time.read().unwrap();
                            times.clone()
                        };

                        let new_node = Arc::new(Node {
                            text: RwLock::new(matched_text),
                            children: RwLock::new(HashMap::new()),
                            parent: RwLock::new(Some(Arc::clone(&curr))),
                            tenant_last_access_time: RwLock::new(cloned_access_times),
                        });

                        // Add matched node as child of new node
                        let first_new_char = contracted_text.chars().next().unwrap();
                        {
                            let mut children = new_node.children.write().unwrap();
                            children.insert(first_new_char, Arc::clone(&matched_node));
                        }

                        // Update parent's children
                        {
                            let mut children = curr.children.write().unwrap();
                            children.insert(first_char, Arc::clone(&new_node));
                        }

                        // Update matched node
                        {
                            let mut text = matched_node.text.write().unwrap();
                            *text = contracted_text;
                        }
                        {
                            let mut parent = matched_node.parent.write().unwrap();
                            *parent = Some(Arc::clone(&new_node));
                        }

                        prev = Arc::clone(&new_node);

                        // Update tenant access time on new node
                        {
                            let mut access_times = prev.tenant_last_access_time.write().unwrap();
                            let is_new = !access_times.contains_key(tenant);
                            access_times.insert(tenant.to_string(), timestamp_ms);

                            if is_new {
                                let mut counts = self.tenant_char_count.write().unwrap();
                                counts
                                    .entry(tenant.to_string())
                                    .and_modify(|count| *count += matched_text_count)
                                    .or_insert(matched_text_count);
                            }
                        }

                        curr_idx += shared_count;
                    } else {
                        // Full match - move to next node
                        prev = Arc::clone(&matched_node);

                        // Update tenant access time
                        {
                            let mut access_times = prev.tenant_last_access_time.write().unwrap();
                            let is_new = !access_times.contains_key(tenant);
                            access_times.insert(tenant.to_string(), timestamp_ms);

                            if is_new {
                                let mut counts = self.tenant_char_count.write().unwrap();
                                counts
                                    .entry(tenant.to_string())
                                    .and_modify(|count| *count += matched_node_text_count)
                                    .or_insert(matched_node_text_count);
                            }
                        }

                        curr_idx += shared_count;
                    }
                }
            }
        }
    }

    /// Find the longest prefix match and return (matched_text, tenant)
    pub fn prefix_match(&self, text: &str) -> (String, String) {
        let mut curr_idx = 0;

        let mut prev = Arc::clone(&self.root);
        let text_count = text.chars().count();

        while curr_idx < text_count {
            let first_char = text.chars().nth(curr_idx).unwrap();
            let curr_text = slice_by_chars(text, curr_idx, text_count);

            let curr = prev.clone();

            let child = {
                let children = curr.children.read().unwrap();
                children.get(&first_char).cloned()
            };

            if let Some(matched_node) = child {
                let matched_text = matched_node.text.read().unwrap().clone();
                let shared_count = shared_prefix_count(&matched_text, &curr_text);
                let matched_node_text_count = matched_text.chars().count();

                if shared_count == matched_node_text_count {
                    // Full match, continue to next node
                    curr_idx += shared_count;
                    prev = Arc::clone(&matched_node);
                } else {
                    // Partial match, stop here
                    curr_idx += shared_count;
                    prev = Arc::clone(&matched_node);
                    break;
                }
            } else {
                // No match found
                break;
            }
        }

        let curr = prev;

        // Select the first tenant (or most recently accessed)
        let tenant = {
            let access_times = curr.tenant_last_access_time.read().unwrap();
            access_times
                .iter()
                .max_by_key(|(_, &time)| time)
                .map(|(t, _)| t.clone())
                .unwrap_or_else(|| "empty".to_string())
        };

        // Update access times along the path
        if tenant != "empty" {
            let timestamp_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis();

            let mut current_node = Some(curr);
            while let Some(node) = current_node {
                {
                    let mut access_times = node.tenant_last_access_time.write().unwrap();
                    access_times.insert(tenant.clone(), timestamp_ms);
                }
                current_node = node.parent.read().unwrap().clone();
            }
        }

        let ret_text = slice_by_chars(text, 0, curr_idx);
        (ret_text, tenant)
    }

    /// Get leaf tenants for a node (tenants that don't have children)
    fn leaf_of(node: &NodeRef) -> Vec<String> {
        let mut candidates: HashMap<String, bool> = {
            let access_times = node.tenant_last_access_time.read().unwrap();
            access_times.keys().map(|k| (k.clone(), true)).collect()
        };

        let children = node.children.read().unwrap();
        for child in children.values() {
            let child_times = child.tenant_last_access_time.read().unwrap();
            for tenant in child_times.keys() {
                candidates.insert(tenant.clone(), false);
            }
        }

        candidates
            .into_iter()
            .filter(|(_, is_leaf)| *is_leaf)
            .map(|(tenant, _)| tenant)
            .collect()
    }

    /// Evict tenant data by size limit using LRU
    pub fn evict_tenant_by_size(&self, max_size: usize) {
        // Collect leaves into priority queue
        let mut stack = vec![Arc::clone(&self.root)];
        let mut pq = BinaryHeap::new();

        while let Some(curr) = stack.pop() {
            let children = curr.children.read().unwrap();
            for child in children.values() {
                stack.push(Arc::clone(child));
            }
            drop(children);

            for tenant in Tree::leaf_of(&curr) {
                let access_times = curr.tenant_last_access_time.read().unwrap();
                if let Some(&timestamp) = access_times.get(&tenant) {
                    pq.push(Reverse(EvictionEntry {
                        timestamp,
                        tenant: tenant.clone(),
                        node: Arc::clone(&curr),
                    }));
                }
            }
        }

        debug!(
            "Before eviction - tenant char counts: {:?}",
            self.tenant_char_count.read().unwrap()
        );

        // Process eviction
        while let Some(Reverse(entry)) = pq.pop() {
            let EvictionEntry { tenant, node, .. } = entry;

            let used_size = {
                let counts = self.tenant_char_count.read().unwrap();
                counts.get(&tenant).copied().unwrap_or(0)
            };

            if used_size <= max_size {
                continue;
            }

            // Decrement count when removing tenant from node
            {
                let access_times = node.tenant_last_access_time.read().unwrap();
                if access_times.contains_key(&tenant) {
                    let node_len = node.text.read().unwrap().chars().count();
                    let mut counts = self.tenant_char_count.write().unwrap();
                    counts.entry(tenant.clone()).and_modify(|count| {
                        *count = count.saturating_sub(node_len);
                    });
                }
            }

            // Remove tenant from node
            {
                let mut access_times = node.tenant_last_access_time.write().unwrap();
                access_times.remove(&tenant);
            }

            // Get parent reference before checking if node should be removed
            let parent_ref = node.parent.read().unwrap().clone();

            // Remove empty nodes
            {
                let children = node.children.read().unwrap();
                let access_times = node.tenant_last_access_time.read().unwrap();
                if children.is_empty() && access_times.is_empty() {
                    drop(children);
                    drop(access_times);

                    if let Some(ref parent) = parent_ref {
                        let text = node.text.read().unwrap();
                        if let Some(first_char) = text.chars().next() {
                            let mut parent_children = parent.children.write().unwrap();
                            parent_children.remove(&first_char);
                        }
                    }
                }
            }

            // Add parent to queue if it becomes a leaf
            if let Some(ref parent) = parent_ref {
                if Tree::leaf_of(parent).contains(&tenant) {
                    let access_times = parent.tenant_last_access_time.read().unwrap();
                    if let Some(&timestamp) = access_times.get(&tenant) {
                        pq.push(Reverse(EvictionEntry {
                            timestamp,
                            tenant: tenant.clone(),
                            node: Arc::clone(parent),
                        }));
                    }
                }
            }
        }

        debug!(
            "After eviction - tenant char counts: {:?}",
            self.tenant_char_count.read().unwrap()
        );
    }

    /// Remove all data for a tenant
    pub fn remove_tenant(&self, tenant: &str) {
        // Find all leaves for the tenant
        let mut stack = vec![Arc::clone(&self.root)];
        let mut queue = VecDeque::new();

        while let Some(curr) = stack.pop() {
            let children = curr.children.read().unwrap();
            for child in children.values() {
                stack.push(Arc::clone(child));
            }
            drop(children);

            if Tree::leaf_of(&curr).contains(&tenant.to_string()) {
                queue.push_back(Arc::clone(&curr));
            }
        }

        // Traverse up from leaves, removing tenant
        while let Some(curr) = queue.pop_front() {
            {
                let mut access_times = curr.tenant_last_access_time.write().unwrap();
                access_times.remove(tenant);
            }

            // Remove empty nodes
            let children = curr.children.read().unwrap();
            let access_times = curr.tenant_last_access_time.read().unwrap();
            if children.is_empty() && access_times.is_empty() {
                drop(children);
                drop(access_times);

                if let Some(parent) = curr.parent.read().unwrap().as_ref() {
                    let text = curr.text.read().unwrap();
                    if let Some(first_char) = text.chars().next() {
                        let mut parent_children = parent.children.write().unwrap();
                        parent_children.remove(&first_char);
                    }
                }
            } else {
                drop(children);
                drop(access_times);
            }

            // Add parent to queue if it becomes a leaf
            if let Some(parent) = curr.parent.read().unwrap().as_ref() {
                if Tree::leaf_of(parent).contains(&tenant.to_string()) {
                    queue.push_back(Arc::clone(parent));
                }
            }
        }

        // Remove from char count
        {
            let mut counts = self.tenant_char_count.write().unwrap();
            counts.remove(tenant);
        }
    }

    /// Get the tenant with smallest tree size
    pub fn get_smallest_tenant(&self) -> String {
        let counts = self.tenant_char_count.read().unwrap();

        if counts.is_empty() {
            return "empty".to_string();
        }

        counts
            .iter()
            .min_by_key(|(_, &count)| count)
            .map(|(tenant, _)| tenant.clone())
            .unwrap_or_else(|| "empty".to_string())
    }

    /// Get the used size per tenant
    pub fn get_used_size_per_tenant(&self) -> HashMap<String, usize> {
        self.tenant_char_count.read().unwrap().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cold_start() {
        let tree = Tree::new();
        let (matched_text, tenant) = tree.prefix_match("hello");
        assert_eq!(matched_text, "");
        assert_eq!(tenant, "empty");
    }

    #[test]
    fn test_exact_match() {
        let tree = Tree::new();
        tree.insert("hello", "tenant1");
        tree.insert("apple", "tenant2");
        tree.insert("banana", "tenant3");

        let (matched_text, tenant) = tree.prefix_match("hello");
        assert_eq!(matched_text, "hello");
        assert_eq!(tenant, "tenant1");

        let (matched_text, tenant) = tree.prefix_match("apple");
        assert_eq!(matched_text, "apple");
        assert_eq!(tenant, "tenant2");

        let (matched_text, tenant) = tree.prefix_match("banana");
        assert_eq!(matched_text, "banana");
        assert_eq!(tenant, "tenant3");
    }

    #[test]
    fn test_prefix_match() {
        let tree = Tree::new();
        tree.insert("hello world", "tenant1");

        let (matched_text, tenant) = tree.prefix_match("hello");
        assert_eq!(matched_text, "hello");
        assert_eq!(tenant, "tenant1");

        let (matched_text, tenant) = tree.prefix_match("hello world!");
        assert_eq!(matched_text, "hello world");
        assert_eq!(tenant, "tenant1");
    }

    #[test]
    fn test_shared_prefix() {
        let tree = Tree::new();
        tree.insert("apple", "tenant1");
        tree.insert("application", "tenant2");

        let (matched_text, tenant) = tree.prefix_match("app");
        assert_eq!(matched_text, "app");
        // Should return one of the tenants
        assert!(tenant == "tenant1" || tenant == "tenant2");
    }

    #[test]
    fn test_tenant_removal() {
        let tree = Tree::new();
        tree.insert("hello", "tenant1");
        tree.insert("hello", "tenant2");

        tree.remove_tenant("tenant1");

        let (matched_text, tenant) = tree.prefix_match("hello");
        assert_eq!(matched_text, "hello");
        assert_eq!(tenant, "tenant2");
    }

    #[test]
    fn test_get_smallest_tenant() {
        let tree = Tree::new();

        assert_eq!(tree.get_smallest_tenant(), "empty");

        tree.insert("hello", "tenant1");
        tree.insert("world", "tenant1");
        tree.insert("hi", "tenant2");

        // tenant2 has 2 chars, tenant1 has 10 chars
        assert_eq!(tree.get_smallest_tenant(), "tenant2");
    }

    #[test]
    fn test_utf8_support() {
        let tree = Tree::new();
        tree.insert("你好", "tenant1");
        tree.insert("你好世界", "tenant2");

        let (matched_text, _tenant) = tree.prefix_match("你好");
        assert_eq!(matched_text, "你好");

        let (matched_text, tenant) = tree.prefix_match("你好世界");
        assert_eq!(matched_text, "你好世界");
        assert_eq!(tenant, "tenant2");
    }
}
