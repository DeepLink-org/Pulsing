//! Actor spawning logic
//!
//! This module contains the implementation of actor spawning methods
//! that are used by the ActorSystem.
//!
//! The core spawn implementation is in `SpawnBuilder::spawn_factory()`.
//! All other spawn methods delegate to the builder.

use crate::actor::{Actor, ActorContext, ActorId, ActorPath, ActorRef, ActorSystemRef, Mailbox};
use crate::error::{PulsingError, RuntimeError};
use crate::system::config::SpawnOptions;
use crate::system::handle::{ActorStats, LocalActorHandle};
use crate::system::runtime::run_supervision_loop;
use crate::system::ActorSystem;
use std::sync::Arc;

impl ActorSystem {
    /// Internal spawn implementation - the actual core logic
    ///
    /// This is called by `SpawnBuilder::spawn_factory()` and handles both
    /// anonymous and named actor spawning.
    pub(crate) async fn spawn_internal<F, A>(
        self: &Arc<Self>,
        path: Option<ActorPath>,
        factory: F,
        options: SpawnOptions,
    ) -> anyhow::Result<ActorRef>
    where
        F: FnMut() -> anyhow::Result<A> + Send + 'static,
        A: Actor,
    {
        let name_str = path.as_ref().map(|p| p.as_str().to_string());

        // Check for name conflicts (only for named actors)
        if let Some(ref name) = name_str {
            if self.actor_names.contains_key(name) {
                return Err(anyhow::Error::from(PulsingError::from(
                    RuntimeError::actor_already_exists(name.clone()),
                )));
            }
            if self.named_actor_paths.contains_key(name) {
                return Err(anyhow::anyhow!("Named path already registered: {}", name));
            }
        }

        let actor_id = self.next_actor_id();

        let mailbox = Mailbox::with_capacity(self.mailbox_capacity(&options));
        let (sender, receiver) = mailbox.split();

        let stats = Arc::new(ActorStats::default());
        let metadata = options.metadata.clone();

        let actor_cancel = self.cancel_token.child_token();

        let ctx = Self::build_context(self, actor_id, &sender, &actor_cancel, name_str.clone());

        let stats_clone = stats.clone();
        let cancel = actor_cancel.clone();
        let actor_id_for_log = actor_id;
        let supervision = options.supervision.clone();

        let join_handle = tokio::spawn(async move {
            let reason =
                run_supervision_loop(factory, receiver, ctx, cancel, stats_clone, supervision)
                    .await;
            tracing::debug!(actor_id = ?actor_id_for_log, reason = ?reason, "Actor stopped");
        });

        let handle = LocalActorHandle {
            sender: sender.clone(),
            join_handle,
            cancel_token: actor_cancel,
            stats: stats.clone(),
            metadata: metadata.clone(),
            named_path: path.clone(),
            actor_id,
        };

        self.local_actors.insert(actor_id, handle);

        // Register in name maps
        if let Some(ref name) = name_str {
            self.actor_names.insert(name.clone(), actor_id);
            self.named_actor_paths.insert(name.clone(), name.clone());

            // Register with cluster if available
            if let Some(ref path) = path {
                if let Some(cluster) = self.cluster.read().await.as_ref() {
                    if metadata.is_empty() {
                        cluster.register_named_actor(path.clone()).await;
                    } else {
                        cluster
                            .register_named_actor_full(path.clone(), actor_id, metadata)
                            .await;
                    }
                }
            }
        } else {
            // Anonymous actor: use actor_id as key
            self.actor_names.insert(actor_id.to_string(), actor_id);
        }

        Ok(ActorRef::local(actor_id, sender))
    }

    /// Generate a new unique actor ID using UUID
    pub(crate) fn next_actor_id(&self) -> ActorId {
        ActorId::generate()
    }

    fn mailbox_capacity(&self, options: &SpawnOptions) -> usize {
        options
            .mailbox_capacity
            .unwrap_or(self.default_mailbox_capacity)
    }

    fn build_context(
        system: &Arc<Self>,
        actor_id: ActorId,
        sender: &tokio::sync::mpsc::Sender<crate::actor::Envelope>,
        cancel: &tokio_util::sync::CancellationToken,
        name: Option<String>,
    ) -> ActorContext {
        match name {
            Some(name) => ActorContext::with_system_and_name(
                actor_id,
                system.clone() as Arc<dyn ActorSystemRef>,
                cancel.clone(),
                sender.clone(),
                Some(name),
            ),
            None => ActorContext::with_system(
                actor_id,
                system.clone() as Arc<dyn ActorSystemRef>,
                cancel.clone(),
                sender.clone(),
            ),
        }
    }
}
