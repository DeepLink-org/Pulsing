//! Shared-memory control-plane primitives.
//!
//! This module deliberately owns *lifetime* and *capability* semantics before
//! an OS specific mapping backend is introduced.  The current backend is an
//! in-process, immutable `Bytes` region: cloning a mapping is zero-copy and
//! has the same lease and revocation rules a cross-process backend will need.
//!
//! A future POSIX/Windows shared-memory transport can replace the backing
//! implementation behind these descriptors without changing the rendezvous
//! (`offer`) or serving (`publish` / `open`) API.

use crate::error::{PulsingError, Result, RuntimeError};
use bytes::Bytes;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// The backing currently selected by a [`ShmManager`].
///
/// `InProcess` is intentionally explicit: it is a semantic foundation, not a
/// claim that an arbitrary peer process can already map this memory.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ShmBackend {
    /// `Bytes`-backed regions shared by handles in this ActorSystem process.
    #[default]
    InProcess,
}

impl ShmBackend {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InProcess => "in_process",
        }
    }
}

/// Opaque capability for a bounded part of a shared-memory region.
///
/// A descriptor is valid only while its lease is active.  `generation` is
/// included so future reusable OS allocations can reject stale descriptors.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShmRegionDescriptor {
    pub region_id: u64,
    pub generation: u64,
    pub offset: usize,
    pub len: usize,
    pub lease_id: u64,
}

/// Snapshot for metrics and control-plane inspection.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ShmStats {
    pub regions: usize,
    pub published_regions: usize,
    pub active_leases: usize,
    pub bytes: usize,
}

/// Node-scoped region manager.
///
/// The lock is only on metadata operations.  A successful [`Self::map`]
/// returns a `Bytes` handle and does not keep the manager locked while users
/// consume the payload.
#[derive(Default)]
pub struct ShmManager {
    state: Mutex<ShmState>,
}

#[derive(Default)]
struct ShmState {
    next_region_id: u64,
    next_lease_id: u64,
    regions: HashMap<u64, Region>,
    published_names: HashMap<String, u64>,
}

struct Region {
    generation: u64,
    bytes: Bytes,
    published_name: Option<String>,
    /// `None` represents a TTL that exceeds the platform `Instant` range and
    /// is therefore treated as non-expiring rather than immediately expired.
    leases: HashMap<u64, Option<Instant>>,
}

impl ShmManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn backend(&self) -> ShmBackend {
        ShmBackend::InProcess
    }

    /// Create a one-recipient region for the rendezvous/message-style API.
    ///
    /// Releasing (or expiring) the returned lease removes the region.
    pub fn offer(&self, bytes: Bytes, lease_ttl: Duration) -> ShmRegionDescriptor {
        let mut state = self.lock_state();
        reclaim_expired_locked(&mut state, Instant::now());
        let region_id = next_id(&mut state.next_region_id);
        let generation = 1;
        let lease_id = next_id(&mut state.next_lease_id);
        let descriptor = ShmRegionDescriptor {
            region_id,
            generation,
            offset: 0,
            len: bytes.len(),
            lease_id,
        };
        let mut leases = HashMap::new();
        leases.insert(lease_id, deadline(lease_ttl));
        state.regions.insert(
            region_id,
            Region {
                generation,
                bytes,
                published_name: None,
                leases,
            },
        );
        descriptor
    }

    /// Publish a named region for the serve-style API.
    ///
    /// The name remains available until [`Self::unpublish`] is called.  Each
    /// [`Self::open`] creates an independent lease, so one consumer cannot
    /// revoke another consumer's mapping.
    pub fn publish(&self, name: impl Into<String>, bytes: Bytes) -> Result<()> {
        let name = name.into();
        if name.is_empty() {
            return Err(shm_error("published region name must not be empty"));
        }

        let mut state = self.lock_state();
        reclaim_expired_locked(&mut state, Instant::now());
        if state.published_names.contains_key(&name) {
            return Err(shm_error(format!(
                "shared-memory region '{name}' already exists"
            )));
        }
        let region_id = next_id(&mut state.next_region_id);
        state.published_names.insert(name.clone(), region_id);
        state.regions.insert(
            region_id,
            Region {
                generation: 1,
                bytes,
                published_name: Some(name),
                leases: HashMap::new(),
            },
        );
        Ok(())
    }

    /// Open a published region and create a time-bounded consumer capability.
    pub fn open(&self, name: &str, lease_ttl: Duration) -> Result<ShmRegionDescriptor> {
        let mut state = self.lock_state();
        reclaim_expired_locked(&mut state, Instant::now());
        let region_id = *state
            .published_names
            .get(name)
            .ok_or_else(|| shm_error(format!("shared-memory region '{name}' was not found")))?;
        let lease_id = next_id(&mut state.next_lease_id);
        let region = state
            .regions
            .get_mut(&region_id)
            .expect("published region must exist");
        region.leases.insert(lease_id, deadline(lease_ttl));
        Ok(ShmRegionDescriptor {
            region_id,
            generation: region.generation,
            offset: 0,
            len: region.bytes.len(),
            lease_id,
        })
    }

    /// Resolve a descriptor to its zero-copy in-process mapping.
    pub fn map(&self, descriptor: &ShmRegionDescriptor) -> Result<Bytes> {
        let mut state = self.lock_state();
        let now = Instant::now();
        let remove = match state.regions.get_mut(&descriptor.region_id) {
            Some(region) => {
                if region.generation != descriptor.generation {
                    return Err(shm_error("stale shared-memory region descriptor"));
                }
                match region.leases.get(&descriptor.lease_id) {
                    Some(expires_at)
                        if expires_at
                            .as_ref()
                            .map(|expires_at| *expires_at > now)
                            .unwrap_or(true) =>
                    {
                        let end = descriptor
                            .offset
                            .checked_add(descriptor.len)
                            .ok_or_else(|| shm_error("shared-memory descriptor range overflow"))?;
                        if end > region.bytes.len() {
                            return Err(shm_error("shared-memory descriptor range is invalid"));
                        }
                        return Ok(region.bytes.slice(descriptor.offset..end));
                    }
                    Some(_) => {
                        region.leases.remove(&descriptor.lease_id);
                        region.leases.is_empty() && region.published_name.is_none()
                    }
                    None => return Err(shm_error("shared-memory lease is no longer active")),
                }
            }
            None => return Err(shm_error("shared-memory region is no longer available")),
        };
        if remove {
            state.regions.remove(&descriptor.region_id);
        }
        Err(shm_error("shared-memory lease has expired"))
    }

    /// Release a consumer capability.  This is idempotent, which lets actor
    /// teardown safely race with best-effort lease cleanup.
    pub fn release(&self, descriptor: &ShmRegionDescriptor) {
        let mut state = self.lock_state();
        let remove = if let Some(region) = state.regions.get_mut(&descriptor.region_id) {
            if region.generation != descriptor.generation {
                return;
            }
            region.leases.remove(&descriptor.lease_id);
            region.leases.is_empty() && region.published_name.is_none()
        } else {
            false
        };
        if remove {
            state.regions.remove(&descriptor.region_id);
        }
    }

    /// Stop accepting new opens.  Existing leases remain valid until released
    /// or expired, which gives serving and rendezvous the same drain rule.
    pub fn unpublish(&self, name: &str) -> Result<()> {
        let mut state = self.lock_state();
        let region_id = state
            .published_names
            .remove(name)
            .ok_or_else(|| shm_error(format!("shared-memory region '{name}' was not found")))?;
        let remove = if let Some(region) = state.regions.get_mut(&region_id) {
            region.published_name = None;
            region.leases.is_empty()
        } else {
            false
        };
        if remove {
            state.regions.remove(&region_id);
        }
        Ok(())
    }

    /// Reclaim expired leases and return how many leases were removed.
    pub fn reclaim_expired(&self) -> usize {
        let mut state = self.lock_state();
        reclaim_expired_locked(&mut state, Instant::now())
    }

    pub fn stats(&self) -> ShmStats {
        let mut state = self.lock_state();
        reclaim_expired_locked(&mut state, Instant::now());
        ShmStats {
            regions: state.regions.len(),
            published_regions: state.published_names.len(),
            active_leases: state
                .regions
                .values()
                .map(|region| region.leases.len())
                .sum(),
            bytes: state
                .regions
                .values()
                .map(|region| region.bytes.len())
                .sum(),
        }
    }

    /// Revoke every manager-owned descriptor during node shutdown.
    ///
    /// Existing cloned `Bytes` values remain memory-safe, but no descriptor can
    /// be mapped again after this call.
    pub fn clear(&self) {
        let mut state = self.lock_state();
        state.regions.clear();
        state.published_names.clear();
    }

    fn lock_state(&self) -> std::sync::MutexGuard<'_, ShmState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

fn next_id(counter: &mut u64) -> u64 {
    *counter = counter
        .checked_add(1)
        .expect("shared-memory identifier exhausted");
    *counter
}

fn deadline(ttl: Duration) -> Option<Instant> {
    Instant::now().checked_add(ttl)
}

fn reclaim_expired_locked(state: &mut ShmState, now: Instant) -> usize {
    let mut reclaimed = 0;
    let mut empty_unpublished = Vec::new();
    for (region_id, region) in &mut state.regions {
        let before = region.leases.len();
        region.leases.retain(|_, expires_at| {
            expires_at
                .as_ref()
                .map(|expires_at| *expires_at > now)
                .unwrap_or(true)
        });
        reclaimed += before - region.leases.len();
        if region.leases.is_empty() && region.published_name.is_none() {
            empty_unpublished.push(*region_id);
        }
    }
    for region_id in empty_unpublished {
        state.regions.remove(&region_id);
    }
    reclaimed
}

fn shm_error(message: impl Into<String>) -> PulsingError {
    PulsingError::from(RuntimeError::Other(format!(
        "Shared-memory error: {}",
        message.into()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offer_maps_zero_copy_and_release_revokes_it() {
        let manager = ShmManager::new();
        let source = Bytes::from_static(b"tensor");
        let descriptor = manager.offer(source.clone(), Duration::from_secs(1));

        let mapped = manager.map(&descriptor).unwrap();
        assert_eq!(mapped, source);
        assert_eq!(mapped.as_ptr(), source.as_ptr());

        manager.release(&descriptor);
        assert!(manager.map(&descriptor).is_err());
        assert_eq!(manager.stats().regions, 0);
    }

    #[test]
    fn publish_open_and_unpublish_drain_existing_leases() {
        let manager = ShmManager::new();
        manager
            .publish("models/current", Bytes::from_static(b"weights"))
            .unwrap();
        let descriptor = manager
            .open("models/current", Duration::from_secs(1))
            .unwrap();

        manager.unpublish("models/current").unwrap();
        assert!(manager
            .open("models/current", Duration::from_secs(1))
            .is_err());
        assert_eq!(&manager.map(&descriptor).unwrap()[..], b"weights");

        manager.release(&descriptor);
        assert_eq!(manager.stats().regions, 0);
    }

    #[test]
    fn expired_offer_is_reclaimed() {
        let manager = ShmManager::new();
        let descriptor = manager.offer(Bytes::from_static(b"x"), Duration::ZERO);
        assert_eq!(manager.reclaim_expired(), 1);
        assert!(manager.map(&descriptor).is_err());
    }

    #[test]
    fn new_offer_opportunistically_reclaims_expired_regions() {
        let manager = ShmManager::new();
        let expired = manager.offer(Bytes::from_static(b"old"), Duration::ZERO);
        let active = manager.offer(Bytes::from_static(b"new"), Duration::from_secs(1));

        assert!(manager.map(&expired).is_err());
        assert_eq!(&manager.map(&active).unwrap()[..], b"new");
        assert_eq!(manager.stats().regions, 1);
    }

    #[test]
    fn stats_excludes_expired_leases_but_retains_published_region() {
        let manager = ShmManager::new();
        manager
            .publish("models/current", Bytes::from_static(b"weights"))
            .unwrap();
        manager.open("models/current", Duration::ZERO).unwrap();

        assert_eq!(
            manager.stats(),
            ShmStats {
                regions: 1,
                published_regions: 1,
                active_leases: 0,
                bytes: 7,
            }
        );
    }

    #[test]
    fn overflowing_ttl_does_not_expire_immediately() {
        let manager = ShmManager::new();
        let descriptor = manager.offer(Bytes::from_static(b"x"), Duration::MAX);
        assert_eq!(&manager.map(&descriptor).unwrap()[..], b"x");
        assert_eq!(manager.reclaim_expired(), 0);
    }

    #[test]
    fn clear_revokes_all_descriptors() {
        let manager = ShmManager::new();
        let descriptor = manager.offer(Bytes::from_static(b"x"), Duration::from_secs(1));
        manager
            .publish("models/current", Bytes::from_static(b"weights"))
            .unwrap();
        manager.clear();
        assert_eq!(manager.stats(), ShmStats::default());
        assert!(manager.map(&descriptor).is_err());
    }
}
