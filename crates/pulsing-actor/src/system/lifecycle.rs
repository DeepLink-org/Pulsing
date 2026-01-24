//! Actor lifecycle management
//!
//! This module contains the implementation of actor stop and shutdown methods
//! for graceful lifecycle management.

use crate::actor::{ActorPath, StopReason};
use crate::system::ActorSystem;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

impl ActorSystem {
    /// Default timeout for graceful actor shutdown (30 seconds)
    pub(crate) const GRACEFUL_STOP_TIMEOUT: Duration = Duration::from_secs(30);

    /// Stop an actor gracefully
    ///
    /// This method first signals the actor to stop via its cancellation token,
    /// waits for it to finish (with timeout), then performs cleanup.
    /// If the actor doesn't stop within the timeout, it will be forcefully aborted.
    pub async fn stop(&self, name: impl AsRef<str>) -> anyhow::Result<()> {
        self.stop_with_reason(name, StopReason::Killed).await
    }

    /// Stop an actor with a specific reason
    ///
    /// Note: If the name doesn't contain a "/" and no actor is found with the exact name,
    /// it will try with the "actors/" prefix (for Python compatibility).
    pub async fn stop_with_reason(
        &self,
        name: impl AsRef<str>,
        reason: StopReason,
    ) -> anyhow::Result<()> {
        let name = name.as_ref();

        let actual_name = if self.actor_names.contains_key(name) {
            name.to_string()
        } else if !name.contains('/') {
            let prefixed = format!("actors/{}", name);
            if self.actor_names.contains_key(&prefixed) {
                prefixed
            } else {
                name.to_string()
            }
        } else {
            name.to_string()
        };

        if let Some((_, local_id)) = self.actor_names.remove(&actual_name) {
            if let Some((_, handle)) = self.local_actors.remove(&local_id) {
                let named_path = handle.named_path.clone();
                self.stop_local_actor(
                    &actual_name,
                    handle,
                    named_path,
                    reason,
                    Self::GRACEFUL_STOP_TIMEOUT,
                )
                .await;
            }
        }

        Ok(())
    }

    /// Stop a named actor by path
    pub async fn stop_named(&self, path: &crate::actor::ActorPath) -> anyhow::Result<()> {
        self.stop_named_with_reason(path, StopReason::Killed).await
    }

    /// Stop a named actor by path with a specific reason
    pub async fn stop_named_with_reason(
        &self,
        path: &crate::actor::ActorPath,
        reason: StopReason,
    ) -> anyhow::Result<()> {
        let path_key = path.as_str();

        if let Some(actor_name_ref) = self.named_actor_paths.get(&path_key) {
            let actor_name = actor_name_ref.clone();
            drop(actor_name_ref);

            if let Some((_, local_id)) = self.actor_names.remove(&actor_name) {
                if let Some((_, handle)) = self.local_actors.remove(&local_id) {
                    self.stop_local_actor(
                        &actor_name,
                        handle,
                        Some(path.clone()),
                        reason,
                        Self::GRACEFUL_STOP_TIMEOUT,
                    )
                    .await;
                }
            }
        }

        Ok(())
    }

    /// Shutdown the entire actor system
    ///
    pub async fn shutdown(&self) -> anyhow::Result<()> {
        tracing::info!("Shutting down actor system");

        self.cancel_token.cancel();

        tokio::time::sleep(Duration::from_millis(100)).await;

        let actor_entries: Vec<_> = self
            .local_actors
            .iter()
            .map(|entry| {
                let local_id = *entry.key();
                let actor_id = entry.actor_id;
                let named_path = entry.named_path.clone();
                let name = self
                    .actor_names
                    .iter()
                    .find(|e| *e.value() == local_id)
                    .map(|e| e.key().clone())
                    .unwrap_or_else(|| actor_id.to_string());
                (local_id, actor_id, name, named_path)
            })
            .collect();

        for (local_id, _actor_id, actor_name, named_path) in actor_entries {
            self.actor_names.remove(&actor_name);

            if let Some((_, handle)) = self.local_actors.remove(&local_id) {
                self.stop_local_actor(
                    &actor_name,
                    handle,
                    named_path,
                    StopReason::SystemShutdown,
                    Duration::from_secs(5),
                )
                .await;
            }
        }

        self.local_actors.clear();
        self.actor_names.clear();

        self.node_load.clear();

        self.lifecycle.clear().await;

        {
            let cluster_guard = self.cluster.read().await;
            if let Some(cluster) = cluster_guard.as_ref() {
                cluster.leave().await?;
            }
        }

        tracing::info!("Actor system shutdown complete");
        Ok(())
    }

    /// Get cancellation token
    pub fn cancel_token(&self) -> CancellationToken {
        self.cancel_token.clone()
    }

    async fn stop_local_actor(
        &self,
        actor_name: &str,
        handle: super::handle::LocalActorHandle,
        named_path: Option<ActorPath>,
        reason: StopReason,
        timeout: Duration,
    ) {
        // 1. Signal the actor to stop gracefully
        handle.cancel_token.cancel();

        // 2. Wait for the actor to finish with timeout
        match tokio::time::timeout(timeout, handle.join_handle).await {
            Ok(_) => {
                if let Some(path) = named_path.as_ref() {
                    tracing::debug!(
                        actor = %actor_name,
                        path = %path,
                        "Actor stopped gracefully"
                    );
                } else {
                    tracing::debug!(actor = %actor_name, "Actor stopped gracefully");
                }
            }
            Err(_) => {
                if let Some(path) = named_path.as_ref() {
                    tracing::warn!(
                        actor = %actor_name,
                        path = %path,
                        "Actor didn't stop gracefully within timeout"
                    );
                } else {
                    tracing::warn!(
                        actor = %actor_name,
                        "Actor didn't stop gracefully within timeout"
                    );
                }
            }
        }

        // 3. Handle lifecycle cleanup
        let actor_names = self.actor_names.clone();
        let local_actors = self.local_actors.clone();
        self.lifecycle
            .handle_termination(
                &handle.actor_id,
                actor_name,
                named_path,
                reason,
                &self.named_actor_paths,
                &self.cluster,
                |name| {
                    actor_names
                        .get(name)
                        .and_then(|id| local_actors.get(id.value()).map(|h| h.sender.clone()))
                },
            )
            .await;
    }
}
