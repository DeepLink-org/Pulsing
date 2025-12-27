//! Actor mailbox - message envelope and queue

use super::traits::Message;
use tokio::sync::{mpsc, oneshot};

/// Response channel type
pub type ResponseChannel = oneshot::Sender<anyhow::Result<Message>>;

/// Responder - sends response back to caller (no-op for tell pattern)
pub struct Responder(Option<ResponseChannel>);

impl Responder {
    /// Send response (no-op if this was a tell)
    pub fn send(self, result: anyhow::Result<Message>) {
        if let Some(tx) = self.0 {
            let _ = tx.send(result);
        }
    }
}

/// Message envelope with optional response channel
pub struct Envelope {
    message: Message,
    respond_to: Option<ResponseChannel>,
}

impl Envelope {
    /// Create envelope for fire-and-forget (tell pattern)
    pub fn tell(message: Message) -> Self {
        Self {
            message,
            respond_to: None,
        }
    }

    /// Create envelope for request-response (ask pattern)
    pub fn ask(message: Message, respond_to: ResponseChannel) -> Self {
        Self {
            message,
            respond_to: Some(respond_to),
        }
    }

    /// Get the message type
    pub fn msg_type(&self) -> &str {
        self.message.msg_type()
    }

    /// Decompose into message and responder
    pub fn into_parts(self) -> (Message, Responder) {
        (self.message, Responder(self.respond_to))
    }

    /// Check if this envelope expects a response
    pub fn expects_response(&self) -> bool {
        self.respond_to.is_some()
    }
}

impl std::fmt::Debug for Envelope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Envelope")
            .field("msg_type", &self.message.msg_type())
            .field("expects_response", &self.respond_to.is_some())
            .finish()
    }
}

/// Mailbox capacity
pub const DEFAULT_MAILBOX_SIZE: usize = 256;

/// Actor mailbox
pub struct Mailbox {
    /// Sender half (cloneable)
    sender: mpsc::Sender<Envelope>,

    /// Receiver half
    receiver: mpsc::Receiver<Envelope>,
}

impl Mailbox {
    /// Create a new mailbox with default capacity
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_MAILBOX_SIZE)
    }

    /// Create a new mailbox with specified capacity
    pub fn with_capacity(capacity: usize) -> Self {
        let (sender, receiver) = mpsc::channel(capacity);
        Self { sender, receiver }
    }

    /// Get a clone of the sender
    pub fn sender(&self) -> mpsc::Sender<Envelope> {
        self.sender.clone()
    }

    /// Take the receiver (consumes it)
    pub fn take_receiver(&mut self) -> mpsc::Receiver<Envelope> {
        let (_, new_rx) = mpsc::channel(1);
        std::mem::replace(&mut self.receiver, new_rx)
    }

    /// Split into sender and receiver
    pub fn split(self) -> (mpsc::Sender<Envelope>, mpsc::Receiver<Envelope>) {
        (self.sender, self.receiver)
    }
}

impl Default for Mailbox {
    fn default() -> Self {
        Self::new()
    }
}

/// Mailbox sender wrapper with backpressure handling
#[derive(Clone)]
pub struct MailboxSender {
    inner: mpsc::Sender<Envelope>,
}

impl MailboxSender {
    pub fn new(sender: mpsc::Sender<Envelope>) -> Self {
        Self { inner: sender }
    }

    /// Send a message (blocking if full)
    pub async fn send(&self, envelope: Envelope) -> anyhow::Result<()> {
        self.inner
            .send(envelope)
            .await
            .map_err(|_| anyhow::anyhow!("Mailbox closed"))
    }

    /// Try to send without blocking
    pub fn try_send(&self, envelope: Envelope) -> anyhow::Result<()> {
        self.inner
            .try_send(envelope)
            .map_err(|e| anyhow::anyhow!("Mailbox send failed: {}", e))
    }

    /// Check if mailbox is closed
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
        let (_, responder) = envelope.into_parts();
        responder.send(Ok(Message::single("", b"world")));

        let result = rx.await.unwrap().unwrap();
        assert!(result.is_single());
        let Message::Single { data, .. } = result else {
            panic!("expected single")
        };
        assert_eq!(data, b"world");
    }
}
