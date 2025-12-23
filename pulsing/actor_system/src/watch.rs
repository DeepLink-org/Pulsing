//! Actor lifecycle monitoring and fault tolerance
//!
//! This module provides:
//! - Watch mechanism for monitoring actor lifecycle
//! - Termination handling (logging, cleanup, notification)
//! - Cluster broadcast for named actor failures

use crate::actor::{ActorId, ActorPath, Envelope, Message, StopReason, Terminated};
use crate::cluster::GossipCluster;
use dashmap::DashMap;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};

/// Actor lifecycle manager
///
/// Centralized management of actor lifecycle events including:
/// - Watch/unwatch relationships
/// - Termination handling
/// - Cluster broadcast
/// - Routing table cleanup
pub struct ActorLifecycle {
    /// Watch registry: target_actor_name -> set of watcher_actor_names
    watchers: RwLock<HashMap<String, HashSet<String>>>,
}

impl ActorLifecycle {
    /// Create a new actor lifecycle manager
    pub fn new() -> Self {
        Self {
            watchers: RwLock::new(HashMap::new()),
        }
    }

    // ==================== Watch API ====================

    /// Register a watch: watcher will be notified when target stops
    pub async fn watch(&self, watcher_name: &str, target_name: &str) {
        let mut watchers = self.watchers.write().await;
        watchers
            .entry(target_name.to_string())
            .or_default()
            .insert(watcher_name.to_string());

        tracing::debug!(
            watcher = watcher_name,
            target = target_name,
            "Watch registered"
        );
    }

    /// Remove a watch relationship
    pub async fn unwatch(&self, watcher_name: &str, target_name: &str) {
        let mut watchers = self.watchers.write().await;
        if let Some(watcher_set) = watchers.get_mut(target_name) {
            watcher_set.remove(watcher_name);
            if watcher_set.is_empty() {
                watchers.remove(target_name);
            }
        }

        tracing::debug!(
            watcher = watcher_name,
            target = target_name,
            "Watch removed"
        );
    }

    // ==================== Termination Handling ====================

    /// Handle actor termination
    ///
    /// This is the main entry point for processing actor termination events.
    /// It handles:
    /// 1. Logging the termination
    /// 2. Cleaning up routing tables
    /// 3. Broadcasting to cluster (for named actors)
    /// 4. Notifying all watchers
    ///
    /// # Arguments
    /// * `actor_id` - The terminated actor's ID
    /// * `actor_name` - The actor's local name
    /// * `named_path` - Optional named actor path
    /// * `reason` - Why the actor stopped
    /// * `named_actor_paths` - Routing table to clean up
    /// * `cluster` - Cluster reference for broadcasting
    /// * `get_sender` - Function to get sender for notifying watchers
    pub async fn handle_termination<F>(
        &self,
        actor_id: &ActorId,
        actor_name: &str,
        named_path: Option<ActorPath>,
        reason: StopReason,
        named_actor_paths: &DashMap<String, String>,
        cluster: &RwLock<Option<Arc<GossipCluster>>>,
        get_sender: F,
    ) where
        F: Fn(&str) -> Option<mpsc::Sender<Envelope>>,
    {
        // 1. Log termination
        self.log_termination(actor_id, named_path.as_ref(), &reason);

        // 2. Cleanup routing table and broadcast to cluster
        self.cleanup_and_broadcast(
            actor_id,
            named_path.as_ref(),
            &reason,
            named_actor_paths,
            cluster,
        )
        .await;

        // 3. Notify all watchers
        self.notify_watchers(actor_id, actor_name, reason, get_sender)
            .await;
    }

    /// Log actor termination event
    fn log_termination(
        &self,
        actor_id: &ActorId,
        named_path: Option<&ActorPath>,
        reason: &StopReason,
    ) {
        if let Some(path) = named_path {
            tracing::info!(
                actor_id = %actor_id,
                path = %path,
                reason = %reason,
                "Named actor terminated"
            );
        } else {
            tracing::info!(
                actor_id = %actor_id,
                reason = %reason,
                "Actor terminated"
            );
        }
    }

    /// Clean up routing table and broadcast failure to cluster
    async fn cleanup_and_broadcast(
        &self,
        actor_id: &ActorId,
        named_path: Option<&ActorPath>,
        reason: &StopReason,
        named_actor_paths: &DashMap<String, String>,
        cluster: &RwLock<Option<Arc<GossipCluster>>>,
    ) {
        if let Some(path) = named_path {
            // Remove from routing table
            let path_key = path.as_str();
            named_actor_paths.remove(&path_key);

            // Broadcast failure to cluster
            let cluster_guard = cluster.read().await;
            if let Some(cluster) = cluster_guard.as_ref() {
                cluster.broadcast_named_actor_failed(path, reason).await;
                cluster.unregister_named_actor(path).await;
            }
        } else {
            // Legacy actor - just unregister from cluster
            let cluster_guard = cluster.read().await;
            if let Some(cluster) = cluster_guard.as_ref() {
                cluster.unregister_actor(actor_id).await;
            }
        }
    }

    /// Notify all watchers that an actor has terminated
    async fn notify_watchers<F>(
        &self,
        actor_id: &ActorId,
        actor_name: &str,
        reason: StopReason,
        get_sender: F,
    ) where
        F: Fn(&str) -> Option<mpsc::Sender<Envelope>>,
    {
        // Get and remove watchers for this actor
        let watcher_names = {
            let mut watchers = self.watchers.write().await;
            watchers.remove(actor_name).unwrap_or_default()
        };

        if watcher_names.is_empty() {
            return;
        }

        tracing::info!(
            actor_id = %actor_id,
            reason = %reason,
            watcher_count = watcher_names.len(),
            "Notifying watchers of actor termination"
        );

        // Create Terminated message
        let terminated = Terminated {
            actor_id: actor_id.clone(),
            reason,
        };

        let msg = match Message::pack(&terminated) {
            Ok(msg) => msg,
            Err(e) => {
                tracing::error!(error = %e, "Failed to serialize Terminated message");
                return;
            }
        };

        // Get payload bytes (Terminated is always single message)
        let (msg_type, payload_bytes) = match msg {
            Message::Single { msg_type, data } => (msg_type, data),
            Message::Stream { msg_type, .. } => (msg_type, Vec::new()),
        };

        // Send to all watchers
        for watcher_name in watcher_names {
            if let Some(sender) = get_sender(&watcher_name) {
                let envelope = Envelope::tell(msg_type.clone(), payload_bytes.clone());
                if let Err(e) = sender.try_send(envelope) {
                    tracing::warn!(
                        watcher = watcher_name,
                        error = %e,
                        "Failed to send Terminated message to watcher"
                    );
                }
            }
        }
    }

    // ==================== Batch Operations ====================

    /// Remove an actor from all watch relationships
    ///
    /// Call this when an actor is being removed from the system.
    /// It removes the actor both as a target and as a watcher.
    pub async fn remove_actor(&self, actor_name: &str) {
        let mut watchers = self.watchers.write().await;

        // Remove as target
        watchers.remove(actor_name);

        // Remove as watcher from all targets, and clean up empty entries
        let mut empty_targets = Vec::new();
        for (target, watcher_set) in watchers.iter_mut() {
            watcher_set.remove(actor_name);
            if watcher_set.is_empty() {
                empty_targets.push(target.clone());
            }
        }

        for target in empty_targets {
            watchers.remove(&target);
        }
    }

    /// Clear all watch relationships
    ///
    /// Call this during system shutdown.
    pub async fn clear(&self) {
        let mut watchers = self.watchers.write().await;
        watchers.clear();
    }

    // ==================== Query API ====================

    /// Get the number of actors being watched
    pub async fn watched_count(&self) -> usize {
        self.watchers.read().await.len()
    }

    /// Get watchers for a specific actor
    pub async fn get_watchers(&self, target_name: &str) -> HashSet<String> {
        self.watchers
            .read()
            .await
            .get(target_name)
            .cloned()
            .unwrap_or_default()
    }

    /// Check if an actor is being watched
    pub async fn is_watched(&self, target_name: &str) -> bool {
        self.watchers.read().await.contains_key(target_name)
    }
}

impl Default for ActorLifecycle {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actor::NodeId;

    #[tokio::test]
    async fn test_watch_unwatch() {
        let lifecycle = ActorLifecycle::new();

        // Add watches
        lifecycle.watch("watcher1", "target1").await;
        lifecycle.watch("watcher2", "target1").await;

        assert!(lifecycle.is_watched("target1").await);
        assert_eq!(lifecycle.get_watchers("target1").await.len(), 2);

        // Unwatch
        lifecycle.unwatch("watcher1", "target1").await;
        assert_eq!(lifecycle.get_watchers("target1").await.len(), 1);

        lifecycle.unwatch("watcher2", "target1").await;
        assert!(!lifecycle.is_watched("target1").await);
    }

    #[tokio::test]
    async fn test_remove_actor() {
        let lifecycle = ActorLifecycle::new();

        // Setup: watcher1 watches target1 and target2
        lifecycle.watch("watcher1", "target1").await;
        lifecycle.watch("watcher1", "target2").await;
        lifecycle.watch("watcher2", "target1").await;

        // Remove watcher1 from all relationships
        lifecycle.remove_actor("watcher1").await;

        // watcher1 should be removed as watcher
        let watchers = lifecycle.get_watchers("target1").await;
        assert!(!watchers.contains("watcher1"));
        assert!(watchers.contains("watcher2"));

        // target2 should have no watchers
        assert!(!lifecycle.is_watched("target2").await);
    }

    #[tokio::test]
    async fn test_clear() {
        let lifecycle = ActorLifecycle::new();

        lifecycle.watch("w1", "t1").await;
        lifecycle.watch("w2", "t2").await;

        assert_eq!(lifecycle.watched_count().await, 2);

        lifecycle.clear().await;

        assert_eq!(lifecycle.watched_count().await, 0);
    }

    #[tokio::test]
    async fn test_notify_watchers() {
        let lifecycle = ActorLifecycle::new();

        lifecycle.watch("watcher1", "target1").await;
        lifecycle.watch("watcher2", "target1").await;

        let actor_id = ActorId::new(NodeId::generate(), "target1");

        // Create a channel to receive notifications
        let (tx, mut rx) = mpsc::channel::<Envelope>(10);

        // Notify watchers
        lifecycle
            .notify_watchers(&actor_id, "target1", StopReason::Normal, |_name| {
                Some(tx.clone())
            })
            .await;

        // Should receive 2 notifications
        let mut count = 0;
        while rx.try_recv().is_ok() {
            count += 1;
        }
        assert_eq!(count, 2);

        // Watchers should be cleared after notification
        assert!(!lifecycle.is_watched("target1").await);
    }
}
