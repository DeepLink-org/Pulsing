//! Actor mailbox - message queue implementation

use super::traits::Message;
use tokio::sync::{mpsc, oneshot};

/// Response channel for envelope
pub type ResponseChannel = oneshot::Sender<anyhow::Result<Message>>;

/// Response type for envelope
pub enum EnvelopeResponse {
    /// Unified response channel
    Unified(ResponseChannel),
}

/// Message envelope with optional response channel
pub struct Envelope {
    /// The message
    pub message: Message,

    /// Correlation ID for request tracking
    pub correlation_id: Option<u64>,

    /// Response channel (for ask pattern)
    pub respond_to: Option<EnvelopeResponse>,
}

impl Envelope {
    /// Create a new envelope for tell (fire-and-forget)
    pub fn tell(msg_type: String, payload: Vec<u8>) -> Self {
        Self {
            message: Message::single(msg_type, payload),
            correlation_id: None,
            respond_to: None,
        }
    }

    /// Create a new envelope for tell with Message
    pub fn tell_msg(message: Message) -> Self {
        Self {
            message,
            correlation_id: None,
            respond_to: None,
        }
    }

    /// Create a new envelope for ask
    pub fn ask(message: Message, respond_to: ResponseChannel) -> Self {
        Self {
            message,
            correlation_id: Some(rand::random()),
            respond_to: Some(EnvelopeResponse::Unified(respond_to)),
        }
    }

    /// Get the message type
    pub fn msg_type(&self) -> &str {
        self.message.msg_type()
    }

    /// Get payload as bytes (only for single message)
    pub fn payload(&self) -> anyhow::Result<&[u8]> {
        match &self.message {
            Message::Single { data, .. } => Ok(data),
            Message::Stream { .. } => Err(anyhow::anyhow!("Cannot get bytes from stream message")),
        }
    }

    /// Send a response
    pub fn respond(self, result: anyhow::Result<Message>) {
        if let Some(EnvelopeResponse::Unified(tx)) = self.respond_to {
            let _ = tx.send(result);
        }
    }

    /// Check if this envelope expects a response
    pub fn expects_response(&self) -> bool {
        self.respond_to.is_some()
    }
}

impl std::fmt::Debug for Envelope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Envelope")
            .field("message", &self.message)
            .field("correlation_id", &self.correlation_id)
            .field("has_respond_to", &self.respond_to.is_some())
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

        let envelope = Envelope::tell("test".to_string(), vec![1, 2, 3]);
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
        envelope.respond(Ok(Message::single("", b"world")));

        let result = rx.await.unwrap().unwrap();
        assert!(result.is_single());
        let Message::Single { data, .. } = result else {
            panic!("expected single")
        };
        assert_eq!(data, b"world");
    }
}
