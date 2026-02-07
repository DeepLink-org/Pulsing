//! Actor runtime loop and supervision

use super::handle::ActorStats;
use crate::actor::{Actor, ActorContext, Envelope, StopReason};
use crate::supervision::SupervisionSpec;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// Actor instance loop - runs a single instance of an actor
pub(crate) async fn run_actor_instance<A: Actor>(
    mut actor: A,
    receiver: &mut mpsc::Receiver<Envelope>,
    ctx: &mut ActorContext,
    cancel: CancellationToken,
    stats: Arc<ActorStats>,
) -> StopReason {
    // Call on_start
    if let Err(e) = actor.on_start(ctx).await {
        tracing::error!(actor_id = ?ctx.id(), error = %e, "Actor start error");
        stats.inc_stop();
        return StopReason::Failed(e.to_string());
    }

    let stop_reason = loop {
        tokio::select! {
            msg = receiver.recv() => {
                match msg {
                    Some(envelope) => {
                        stats.inc_message();
                        let (message, responder) = envelope.into_parts();

                        match actor.receive(message, ctx).await {
                            Ok(response) => {
                                responder.send(Ok(response));
                            }
                            Err(e) => {
                                // 业务错误：receive 返回 Err，只把错误返回给调用者，actor 继续处理下一条消息
                                tracing::warn!(actor_id = ?ctx.id(), error = %e, "Receive returned error (returned to caller)");
                                responder.send(Err(e));
                            }
                        }
                    }
                    None => {
                        // Mailbox closed (all senders dropped)
                        break StopReason::Normal;
                    }
                }
            }
            _ = cancel.cancelled() => {
                break StopReason::SystemShutdown;
            }
        }
    };

    // Cleanup
    stats.inc_stop();
    if let Err(e) = actor.on_stop(ctx).await {
        tracing::warn!(actor_id = ?ctx.id(), error = %e, "Actor stop error");
        // If on_stop fails, mark as failed
        if matches!(stop_reason, StopReason::Normal) {
            return StopReason::Failed(e.to_string());
        }
    }

    stop_reason
}

/// Supervision loop - manages actor restarts
pub(crate) async fn run_supervision_loop<F, A>(
    mut factory: F,
    mut receiver: mpsc::Receiver<Envelope>,
    mut ctx: ActorContext,
    cancel: CancellationToken,
    stats: Arc<ActorStats>,
    spec: SupervisionSpec,
) -> StopReason
where
    F: FnMut() -> crate::error::Result<A> + Send + 'static,
    A: Actor,
{
    let mut restarts = 0;
    // Track restarts for windowing if needed (timestamp of restart)
    let mut restart_timestamps: Vec<std::time::Instant> = Vec::new();

    loop {
        // Create actor instance
        let actor = match factory() {
            Ok(a) => a,
            Err(e) => {
                tracing::error!(actor_id = ?ctx.id(), error = %e, "Failed to create actor instance");
                return StopReason::Failed(format!("Factory error: {}", e));
            }
        };

        // Run actor instance
        let reason = run_actor_instance(
            actor,
            &mut receiver,
            &mut ctx,
            cancel.clone(),
            stats.clone(),
        )
        .await;

        // Check if we should restart
        let is_failure = matches!(reason, StopReason::Failed(_));
        if !spec.policy.should_restart(is_failure) {
            return reason;
        }

        if matches!(reason, StopReason::SystemShutdown | StopReason::Killed) {
            return reason;
        }

        // Check max restarts
        restarts += 1;

        // Prune old timestamps if window is set
        if let Some(window) = spec.restart_window {
            let now = std::time::Instant::now();
            restart_timestamps.push(now);
            restart_timestamps.retain(|&t| now.duration_since(t) <= window);

            if restart_timestamps.len() as u32 > spec.max_restarts {
                tracing::error!(actor_id = ?ctx.id(), "Max restarts ({}) exceeded within window {:?}", spec.max_restarts, window);
                return reason;
            }
        } else {
            // Absolute count
            if restarts > spec.max_restarts {
                tracing::error!(actor_id = ?ctx.id(), "Max restarts ({}) exceeded", spec.max_restarts);
                return reason;
            }
        }

        tracing::info!(
            actor_id = ?ctx.id(),
            reason = ?reason,
            restarts = restarts,
            "Restarting actor..."
        );

        // Backoff
        let backoff = spec.backoff.duration(restarts - 1);
        if !backoff.is_zero() {
            tokio::time::sleep(backoff).await;
        }
    }
}
