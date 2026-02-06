//! Actor Registry - manages local actor instances, name mappings, and lifecycle.
//!
//! This module extracts actor management concerns from ActorSystem into a
//! focused subsystem, reducing ActorSystem's responsibilities.

use crate::actor::{ActorId, ActorRef, Envelope};
use crate::watch::ActorLifecycle;
use dashmap::DashMap;
use tokio::sync::mpsc;

use super::handle::LocalActorHandle;

/// Actor Registry - manages local actor instances and name resolution.
///
/// Extracted from ActorSystem to separate actor management concerns from
/// transport, cluster, and system configuration.
///
/// Responsibilities:
/// - Local actor instance storage (ActorId → handle)
/// - Name-to-ActorId mapping (name → ActorId)
/// - Named actor path mapping (path → actor_name)
/// - Actor lifecycle management (watchers, termination)
pub struct ActorRegistry {
    /// Local actors indexed by ActorId (O(1) lookup)
    pub(crate) local_actors: DashMap<ActorId, LocalActorHandle>,

    /// Actor name to ActorId mapping (for name-based lookups)
    pub(crate) actor_names: DashMap<String, ActorId>,

    /// Named actor path to local actor name mapping (path_string -> actor_name)
    pub(crate) named_actor_paths: DashMap<String, String>,

    /// Actor lifecycle manager (watch, termination handling)
    pub(crate) lifecycle: ActorLifecycle,
}

impl ActorRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            local_actors: DashMap::new(),
            actor_names: DashMap::new(),
            named_actor_paths: DashMap::new(),
            lifecycle: ActorLifecycle::new(),
        }
    }

    // =========================================================================
    // Registration
    // =========================================================================

    /// Register a local actor handle
    pub(crate) fn register_actor(&self, actor_id: ActorId, handle: LocalActorHandle) {
        self.local_actors.insert(actor_id, handle);
    }

    /// Register a name-to-ActorId mapping
    pub fn register_name(&self, name: String, actor_id: ActorId) {
        self.actor_names.insert(name, actor_id);
    }

    /// Register a named actor path mapping
    pub fn register_named_path(&self, path: String, actor_name: String) {
        self.named_actor_paths.insert(path, actor_name);
    }

    /// Check if a name is already registered
    pub fn has_name(&self, name: &str) -> bool {
        self.actor_names.contains_key(name)
    }

    /// Check if a named path is already registered
    pub fn has_named_path(&self, path: &str) -> bool {
        self.named_actor_paths.contains_key(path)
    }

    // =========================================================================
    // Lookup
    // =========================================================================

    /// Get a local actor handle by ActorId
    pub(crate) fn get_handle(
        &self,
        actor_id: &ActorId,
    ) -> Option<dashmap::mapref::one::Ref<'_, ActorId, LocalActorHandle>> {
        self.local_actors.get(actor_id)
    }

    /// Get ActorId by name
    pub fn get_actor_id(&self, name: &str) -> Option<ActorId> {
        self.actor_names.get(name).map(|r| *r.value())
    }

    /// Get actor name from named path
    pub fn get_actor_name_by_path(&self, path: &str) -> Option<String> {
        self.named_actor_paths.get(path).map(|r| r.value().clone())
    }

    /// Get a local actor reference by name (O(1) via name → id → handle)
    pub fn local_actor_ref_by_name(&self, name: &str) -> Option<ActorRef> {
        self.actor_names.get(name).and_then(|actor_id| {
            self.local_actors
                .get(actor_id.value())
                .map(|handle| ActorRef::local(handle.actor_id, handle.sender.clone()))
        })
    }

    /// Find actor sender by name or ActorId string (for HTTP handler)
    pub fn find_actor_sender(&self, actor_name: &str) -> Option<mpsc::Sender<Envelope>> {
        // First try by name → ActorId → handle
        if let Some(actor_id) = self.actor_names.get(actor_name) {
            if let Some(handle) = self.local_actors.get(actor_id.value()) {
                return Some(handle.sender.clone());
            }
        }

        // Then try parsing as ActorId (UUID format)
        if let Ok(uuid) = uuid::Uuid::parse_str(actor_name) {
            let actor_id = ActorId::new(uuid.as_u128());
            if let Some(handle) = self.local_actors.get(&actor_id) {
                return Some(handle.sender.clone());
            }
        }

        None
    }

    /// Get list of all local actor names
    pub fn actor_names_list(&self) -> Vec<String> {
        self.actor_names.iter().map(|e| e.key().clone()).collect()
    }

    // =========================================================================
    // Removal
    // =========================================================================

    /// Remove an actor by name, returning (name, actor_id)
    pub fn remove_by_name(&self, name: &str) -> Option<(String, ActorId)> {
        self.actor_names.remove(name)
    }

    /// Remove a local actor handle, returning it
    pub(crate) fn remove_handle(&self, actor_id: &ActorId) -> Option<(ActorId, LocalActorHandle)> {
        self.local_actors.remove(actor_id)
    }

    // =========================================================================
    // Iteration (for shutdown, health checks, etc.)
    // =========================================================================

    /// Iterate over all local actors (for health checks, metrics, etc.)
    pub(crate) fn iter_actors(&self) -> dashmap::iter::Iter<'_, ActorId, LocalActorHandle> {
        self.local_actors.iter()
    }

    /// Iterate over named actor paths
    pub fn iter_named_paths(&self) -> dashmap::iter::Iter<'_, String, String> {
        self.named_actor_paths.iter()
    }

    /// Get count of local actors
    pub fn actor_count(&self) -> usize {
        self.local_actors.len()
    }

    // =========================================================================
    // Bulk operations (for shutdown)
    // =========================================================================

    /// Clear all registrations (for shutdown)
    pub fn clear(&self) {
        self.local_actors.clear();
        self.actor_names.clear();
        self.named_actor_paths.clear();
    }

    /// Clear lifecycle watchers
    pub async fn clear_lifecycle(&self) {
        self.lifecycle.clear().await;
    }
}

impl Default for ActorRegistry {
    fn default() -> Self {
        Self::new()
    }
}
