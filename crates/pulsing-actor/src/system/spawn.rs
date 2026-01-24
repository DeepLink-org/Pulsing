//! Actor spawning logic
//!
//! This module contains the implementation of actor spawning methods
//! that are used by the ActorSystem.

use crate::actor::{
    Actor, ActorContext, ActorId, ActorRef, ActorSystemRef, IntoActor, IntoActorPath, Mailbox,
};
use crate::system::config::SpawnOptions;
use crate::system::handle::{ActorStats, LocalActorHandle};
use crate::system::runtime::{run_actor_instance, run_supervision_loop};
use crate::system::ActorSystem;
use std::sync::atomic::Ordering;
use std::sync::Arc;

impl ActorSystem {
    /// Create a once-use factory from an actor instance
    pub(crate) fn once_factory<A: Actor>(actor: A) -> impl FnMut() -> anyhow::Result<A> {
        let mut actor_opt = Some(actor);
        move || {
            actor_opt
                .take()
                .ok_or_else(|| anyhow::anyhow!("Actor cannot be restarted (spawned as instance)"))
        }
    }

    /// Spawn an anonymous actor (no name, only accessible via ActorRef)
    ///
    /// Note: Anonymous actors do not support supervision/restart because they have
    /// no stable identity for re-resolution. Use `spawn_named_factory` for actors
    /// that need supervision.
    pub async fn spawn_anonymous<A>(self: &Arc<Self>, actor: A) -> anyhow::Result<ActorRef>
    where
        A: IntoActor,
    {
        self.spawn_anonymous_with_options(actor.into_actor(), SpawnOptions::default())
            .await
    }

    /// Spawn an anonymous actor with custom options
    pub async fn spawn_anonymous_with_options<A>(
        self: &Arc<Self>,
        actor: A,
        options: SpawnOptions,
    ) -> anyhow::Result<ActorRef>
    where
        A: IntoActor,
    {
        let actor = actor.into_actor();
        let actor_id = self.next_actor_id();

        let mailbox = Mailbox::with_capacity(self.mailbox_capacity(&options));
        let (sender, receiver) = mailbox.split();

        let stats = Arc::new(ActorStats::default());

        let actor_cancel = self.cancel_token.child_token();

        let ctx = Self::build_context(self, actor_id, &sender, &actor_cancel, None);

        let stats_clone = stats.clone();
        let cancel = actor_cancel.clone();
        let actor_id_for_log = actor_id;

        let join_handle = tokio::spawn(async move {
            let mut receiver = receiver;
            let mut ctx = ctx;
            let reason =
                run_actor_instance(actor, &mut receiver, &mut ctx, cancel, stats_clone).await;
            tracing::debug!(actor_id = ?actor_id_for_log, reason = ?reason, "Anonymous actor stopped");
        });

        let local_id = actor_id.local_id();
        let handle = LocalActorHandle {
            sender: sender.clone(),
            join_handle,
            cancel_token: actor_cancel,
            stats: stats.clone(),
            metadata: options.metadata.clone(),
            named_path: None,
            actor_id,
        };

        self.local_actors.insert(local_id, handle);
        self.actor_names.insert(actor_id.to_string(), local_id);

        Ok(ActorRef::local(actor_id, sender))
    }

    /// Spawn a named actor (resolvable by name across the cluster)
    ///
    /// # Example
    /// ```rust,ignore
    /// // Name is used as both path (for resolution) and local name
    /// system.spawn_named("services/echo", MyActor).await?;
    /// ```
    pub async fn spawn_named<P, A>(self: &Arc<Self>, name: P, actor: A) -> anyhow::Result<ActorRef>
    where
        P: IntoActorPath,
        A: IntoActor,
    {
        let path = name.into_actor_path()?;
        self.spawn_named_factory(
            path,
            Self::once_factory(actor.into_actor()),
            SpawnOptions::default(),
        )
        .await
    }

    /// Spawn a named actor with custom options
    pub async fn spawn_named_with_options<P, A>(
        self: &Arc<Self>,
        name: P,
        actor: A,
        options: SpawnOptions,
    ) -> anyhow::Result<ActorRef>
    where
        P: IntoActorPath,
        A: IntoActor,
    {
        let path = name.into_actor_path()?;
        self.spawn_named_factory(path, Self::once_factory(actor.into_actor()), options)
            .await
    }

    /// Spawn a named actor using a factory function
    pub async fn spawn_named_factory<P, F, A>(
        self: &Arc<Self>,
        name: P,
        factory: F,
        options: SpawnOptions,
    ) -> anyhow::Result<ActorRef>
    where
        P: IntoActorPath,
        F: FnMut() -> anyhow::Result<A> + Send + 'static,
        A: Actor,
    {
        let path = name.into_actor_path()?;
        let name_str = path.as_str();

        if self.actor_names.contains_key(&name_str.to_string()) {
            return Err(anyhow::anyhow!("Actor already exists: {}", name_str));
        }

        if self.named_actor_paths.contains_key(&name_str.to_string()) {
            return Err(anyhow::anyhow!(
                "Named path already registered: {}",
                name_str
            ));
        }

        let actor_id = self.next_actor_id();
        let local_id = actor_id.local_id();

        let mailbox = Mailbox::with_capacity(self.mailbox_capacity(&options));
        let (sender, receiver) = mailbox.split();

        let stats = Arc::new(ActorStats::default());
        let metadata = options.metadata.clone();

        let actor_cancel = self.cancel_token.child_token();

        let ctx = Self::build_context(
            self,
            actor_id,
            &sender,
            &actor_cancel,
            Some(name_str.to_string()),
        );

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
            named_path: Some(path.clone()),
            actor_id,
        };

        self.local_actors.insert(local_id, handle);
        self.actor_names.insert(name_str.to_string(), local_id);
        self.named_actor_paths
            .insert(name_str.to_string(), name_str.to_string());

        if let Some(cluster) = self.cluster.read().await.as_ref() {
            if metadata.is_empty() {
                cluster.register_named_actor(path.clone()).await;
            } else {
                cluster
                    .register_named_actor_full(path.clone(), actor_id, metadata)
                    .await;
            }
        }

        Ok(ActorRef::local(actor_id, sender))
    }

    /// Generate a new unique local actor ID
    pub(crate) fn next_actor_id(&self) -> ActorId {
        let local_id = self.actor_id_counter.fetch_add(1, Ordering::Relaxed);
        ActorId::new(self.node_id, local_id)
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
