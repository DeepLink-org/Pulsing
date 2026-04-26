//! Actor mailbox - message envelope and queue.

use super::traits::Message;
use crate::error::{PulsingError, Result, RuntimeError};
use tokio::sync::{mpsc, oneshot};

/// Response channel type.
pub type ResponseChannel = oneshot::Sender<Result<Message>>;

/// Responder - sends response back to caller (no-op for tell pattern).
pub struct Responder(Option<ResponseChannel>);

impl Responder {
    pub fn send(self, result: Result<Message>) {
        if let Some(tx) = self.0 {
            let _ = tx.send(result);
        }
    }
}

/// Message envelope with optional response channel.
pub struct Envelope {
    message: Message,
    respond_to: Option<ResponseChannel>,
    /// W3C traceparent captured at send time so `actor.receive` can parent to the dispatcher span
    /// (e.g. `http.request`) on a different task.
    linked_traceparent: Option<String>,
    /// W3C `tracestate` paired with [`Self::linked_traceparent`].
    linked_tracestate: Option<String>,
}

impl Envelope {
    pub fn tell(message: Message) -> Self {
        Self {
            message,
            respond_to: None,
            linked_traceparent: None,
            linked_tracestate: None,
        }
    }

    pub fn ask(message: Message, respond_to: ResponseChannel) -> Self {
        Self {
            message,
            respond_to: Some(respond_to),
            linked_traceparent: None,
            linked_tracestate: None,
        }
    }

    /// Attach trace link from [`crate::tracing::capture_linked_traceparent_for_mailbox`].
    pub fn with_linked_traceparent(mut self, tp: Option<String>) -> Self {
        self.linked_traceparent = tp;
        self
    }

    /// Attach W3C `tracestate` from [`crate::tracing::capture_linked_tracestate_for_mailbox`].
    pub fn with_linked_tracestate(mut self, ts: Option<String>) -> Self {
        self.linked_tracestate = ts;
        self
    }

    pub fn msg_type(&self) -> &str {
        self.message.msg_type()
    }

    pub fn into_parts(self) -> (Message, Responder, Option<String>, Option<String>) {
        (
            self.message,
            Responder(self.respond_to),
            self.linked_traceparent,
            self.linked_tracestate,
        )
    }

    pub fn expects_response(&self) -> bool {
        self.respond_to.is_some()
    }
}

impl std::fmt::Debug for Envelope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Envelope")
            .field("msg_type", &self.message.msg_type())
            .field("expects_response", &self.respond_to.is_some())
            .field("linked_trace", &self.linked_traceparent.is_some())
            .field("linked_tracestate", &self.linked_tracestate.is_some())
            .finish()
    }
}

/// Mailbox capacity.
pub const DEFAULT_MAILBOX_SIZE: usize = 256;

/// Actor mailbox.
pub struct Mailbox {
    sender: mpsc::Sender<Envelope>,

    receiver: mpsc::Receiver<Envelope>,
}

impl Mailbox {
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_MAILBOX_SIZE)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        let (sender, receiver) = mpsc::channel(capacity);
        Self { sender, receiver }
    }

    pub fn sender(&self) -> mpsc::Sender<Envelope> {
        self.sender.clone()
    }

    pub fn take_receiver(&mut self) -> mpsc::Receiver<Envelope> {
        let (_, new_rx) = mpsc::channel(1);
        std::mem::replace(&mut self.receiver, new_rx)
    }

    pub fn split(self) -> (mpsc::Sender<Envelope>, mpsc::Receiver<Envelope>) {
        (self.sender, self.receiver)
    }
}

impl Default for Mailbox {
    fn default() -> Self {
        Self::new()
    }
}

/// Mailbox sender wrapper with backpressure handling.
#[derive(Clone)]
pub struct MailboxSender {
    inner: mpsc::Sender<Envelope>,
}

impl MailboxSender {
    pub fn new(sender: mpsc::Sender<Envelope>) -> Self {
        Self { inner: sender }
    }

    pub async fn send(&self, envelope: Envelope) -> Result<()> {
        self.inner
            .send(envelope)
            .await
            .map_err(|_| PulsingError::from(RuntimeError::Other("Mailbox closed".into())))
    }

    pub fn try_send(&self, envelope: Envelope) -> Result<()> {
        self.inner.try_send(envelope).map_err(|e| {
            PulsingError::from(RuntimeError::Other(format!("Mailbox send failed: {}", e)))
        })
    }

    pub fn is_closed(&self) -> bool {
        self.inner.is_closed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mailbox_send_receive() {
        let mut mailbox = Mailbox::new();
        let sender = mailbox.sender();

        let envelope = Envelope::tell(Message::single("test", vec![1, 2, 3]));
        sender.send(envelope).await.unwrap();

        let mut receiver = mailbox.take_receiver();
        let received = receiver.recv().await.unwrap();
        assert_eq!(received.msg_type(), "test");
    }

    #[tokio::test]
    async fn test_envelope_ask_response() {
        let (tx, rx) = oneshot::channel();
        let msg = Message::single("test", b"hello");
        let envelope = Envelope::ask(msg, tx);

        assert!(envelope.expects_response());
        let (_, responder, _, _) = envelope.into_parts();
        responder.send(Ok(Message::single("", b"world")));

        let result = rx.await.unwrap().unwrap();
        assert!(result.is_single());
        let Message::Single { data, .. } = result else {
            panic!("expected single")
        };
        assert_eq!(data, b"world");
    }
}
