//! Mailbox stress and reliability tests

use pulsing_actor::actor::{Envelope, Mailbox, MailboxSender, Message, DEFAULT_MAILBOX_SIZE};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::oneshot;

// ============================================================================
// Basic Mailbox Tests
// ============================================================================

#[tokio::test]
async fn test_mailbox_creation() {
    let mailbox = Mailbox::new();
    let sender = mailbox.sender();
    assert!(!sender.is_closed());
}

#[tokio::test]
async fn test_mailbox_with_capacity() {
    let mailbox = Mailbox::with_capacity(100);
    let sender = mailbox.sender();
    assert!(!sender.is_closed());
}

#[tokio::test]
async fn test_mailbox_send_receive() {
    let mut mailbox = Mailbox::new();
    let sender = MailboxSender::new(mailbox.sender());
    let mut receiver = mailbox.take_receiver();

    // Send
    let envelope = Envelope::tell("test".to_string(), vec![1, 2, 3]);
    sender.send(envelope).await.unwrap();

    // Receive
    let received = receiver.recv().await.unwrap();
    assert_eq!(received.msg_type(), "test");

    // Extract payload from message
    if let Message::Single { data, .. } = received.message {
        assert_eq!(data, vec![1, 2, 3]);
    } else {
        panic!("Expected single message");
    }
}

#[tokio::test]
async fn test_mailbox_ask_pattern() {
    let mut mailbox = Mailbox::new();
    let sender = MailboxSender::new(mailbox.sender());
    let mut receiver = mailbox.take_receiver();

    // Spawn receiver task
    let receiver_task = tokio::spawn(async move {
        if let Some(envelope) = receiver.recv().await {
            assert!(envelope.expects_response());
            // Respond with a Message
            envelope.respond(Ok(Message::single("", vec![42, 43, 44])));
        }
    });

    // Send ask with Message
    let (tx, rx) = oneshot::channel();
    let msg = Message::single("query", vec![1]);
    let envelope = Envelope::ask(msg, tx);
    sender.send(envelope).await.unwrap();

    // Wait for response (now returns Message)
    let response = rx.await.unwrap().unwrap();
    let Message::Single { data, .. } = response else {
        panic!("expected single")
    };
    assert_eq!(data, vec![42, 43, 44]);

    receiver_task.await.unwrap();
}

#[tokio::test]
async fn test_mailbox_closed_detection() {
    let mut mailbox = Mailbox::new();
    let sender = MailboxSender::new(mailbox.sender());

    // Take and drop receiver
    let receiver = mailbox.take_receiver();
    drop(receiver);

    // Try to send - should fail
    let envelope = Envelope::tell("test".to_string(), vec![]);
    let result = sender.send(envelope).await;
    assert!(result.is_err());
    assert!(sender.is_closed());
}

// ============================================================================
// Stress Tests
// ============================================================================

#[tokio::test]
async fn test_mailbox_high_throughput() {
    // Use split() to consume mailbox and avoid dangling sender
    let mailbox = Mailbox::with_capacity(1000);
    let (raw_sender, mut receiver) = mailbox.split();
    let sender = MailboxSender::new(raw_sender);

    let message_count = 10_000;
    let received_count = Arc::new(AtomicUsize::new(0));
    let received_count_clone = received_count.clone();

    // Receiver task
    let receiver_task = tokio::spawn(async move {
        while let Some(_envelope) = receiver.recv().await {
            received_count_clone.fetch_add(1, Ordering::SeqCst);
        }
    });

    // Send many messages
    let start = std::time::Instant::now();
    for i in 0..message_count {
        let envelope = Envelope::tell(format!("msg-{}", i), vec![i as u8]);
        sender.send(envelope).await.unwrap();
    }
    let send_time = start.elapsed();

    // Close sender to signal completion
    drop(sender);

    // Wait for receiver
    receiver_task.await.unwrap();

    let received = received_count.load(Ordering::SeqCst);
    assert_eq!(received, message_count);

    println!(
        "High throughput test: sent {} messages in {:?} ({:.2} msg/sec)",
        message_count,
        send_time,
        message_count as f64 / send_time.as_secs_f64()
    );
}

#[tokio::test]
async fn test_mailbox_concurrent_senders() {
    // Use split() to consume mailbox and avoid dangling sender
    let mailbox = Mailbox::with_capacity(1000);
    let (sender, mut receiver) = mailbox.split();

    let sender_count = 10;
    let messages_per_sender = 1000;
    let total_messages = sender_count * messages_per_sender;

    let received_count = Arc::new(AtomicUsize::new(0));
    let received_count_clone = received_count.clone();

    // Receiver task
    let receiver_task = tokio::spawn(async move {
        while let Some(_envelope) = receiver.recv().await {
            received_count_clone.fetch_add(1, Ordering::SeqCst);
        }
    });

    // Spawn multiple senders
    let mut sender_handles = Vec::new();
    for s in 0..sender_count {
        let sender_clone = MailboxSender::new(sender.clone());
        let handle = tokio::spawn(async move {
            for i in 0..messages_per_sender {
                let envelope =
                    Envelope::tell(format!("sender-{}-msg-{}", s, i), vec![s as u8, i as u8]);
                sender_clone.send(envelope).await.unwrap();
            }
        });
        sender_handles.push(handle);
    }

    // Wait for all senders
    for handle in sender_handles {
        handle.await.unwrap();
    }

    // Close original sender
    drop(sender);

    // Wait for receiver
    receiver_task.await.unwrap();

    let received = received_count.load(Ordering::SeqCst);
    assert_eq!(received, total_messages);

    println!(
        "Concurrent senders test: {} senders x {} messages = {} total",
        sender_count, messages_per_sender, total_messages
    );
}

#[tokio::test]
async fn test_mailbox_backpressure() {
    // Small capacity to trigger backpressure
    // Use split() to consume mailbox and avoid dangling sender
    let mailbox = Mailbox::with_capacity(10);
    let (raw_sender, mut receiver) = mailbox.split();
    let sender = MailboxSender::new(raw_sender);

    let message_count = 100;
    let received_count = Arc::new(AtomicUsize::new(0));
    let received_count_clone = received_count.clone();

    // Slow receiver
    let receiver_task = tokio::spawn(async move {
        while let Some(_envelope) = receiver.recv().await {
            // Simulate slow processing
            tokio::time::sleep(Duration::from_micros(100)).await;
            received_count_clone.fetch_add(1, Ordering::SeqCst);
        }
    });

    // Fast sender (will hit backpressure)
    let start = std::time::Instant::now();
    for i in 0..message_count {
        let envelope = Envelope::tell(format!("msg-{}", i), vec![]);
        sender.send(envelope).await.unwrap();
    }
    let send_time = start.elapsed();

    drop(sender);
    receiver_task.await.unwrap();

    let received = received_count.load(Ordering::SeqCst);
    assert_eq!(received, message_count);

    // Send should have been slowed down by backpressure
    println!(
        "Backpressure test: {} messages sent in {:?}",
        message_count, send_time
    );
}

#[tokio::test]
async fn test_mailbox_try_send() {
    let mut mailbox = Mailbox::with_capacity(2);
    let sender = MailboxSender::new(mailbox.sender());
    let _receiver = mailbox.take_receiver();

    // Fill the mailbox
    sender
        .try_send(Envelope::tell("msg1".to_string(), vec![]))
        .unwrap();
    sender
        .try_send(Envelope::tell("msg2".to_string(), vec![]))
        .unwrap();

    // Third should fail (mailbox full)
    let result = sender.try_send(Envelope::tell("msg3".to_string(), vec![]));
    assert!(result.is_err());
}

// ============================================================================
// Reliability Tests
// ============================================================================

#[tokio::test]
async fn test_mailbox_no_message_loss() {
    // Use split() to consume mailbox and avoid dangling sender
    let mailbox = Mailbox::with_capacity(100);
    let (raw_sender, mut receiver) = mailbox.split();
    let sender = MailboxSender::new(raw_sender);

    let message_count = 5000;
    let mut received_ids = Vec::with_capacity(message_count);

    // Receiver collects all message IDs
    let receiver_task = tokio::spawn(async move {
        while let Some(envelope) = receiver.recv().await {
            // Extract ID from payload
            if let Message::Single { data, .. } = envelope.message {
                if data.len() >= 4 {
                    let id = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
                    received_ids.push(id);
                }
            }
        }
        received_ids
    });

    // Send messages with sequential IDs
    for i in 0..message_count {
        let payload = (i as u32).to_le_bytes().to_vec();
        let envelope = Envelope::tell("numbered".to_string(), payload);
        sender.send(envelope).await.unwrap();
    }

    drop(sender);

    let received_ids = receiver_task.await.unwrap();

    // Verify no loss
    assert_eq!(received_ids.len(), message_count);

    // Verify all IDs received (order may vary in concurrent scenarios)
    let mut sorted_ids = received_ids.clone();
    sorted_ids.sort();
    for (idx, id) in sorted_ids.iter().enumerate() {
        assert_eq!(*id as usize, idx, "Missing or duplicate message ID");
    }
}

#[tokio::test]
async fn test_mailbox_response_delivery() {
    // Use split() to consume mailbox and avoid dangling sender
    let mailbox = Mailbox::new();
    let (raw_sender, mut receiver) = mailbox.split();
    let sender = MailboxSender::new(raw_sender);

    let request_count = 100;

    // Receiver responds to all requests
    let receiver_task = tokio::spawn(async move {
        while let Some(envelope) = receiver.recv().await {
            if envelope.expects_response() {
                // Echo back the payload doubled
                if let Message::Single { ref data, .. } = envelope.message {
                    let response: Vec<u8> = data.iter().map(|b| b * 2).collect();
                    envelope.respond(Ok(Message::single("", response)));
                }
            }
        }
    });

    // Send requests and verify responses
    for i in 0..request_count {
        let (tx, rx) = oneshot::channel();
        let payload = vec![i as u8];
        let msg = Message::single("request", payload);
        let envelope = Envelope::ask(msg, tx);
        sender.send(envelope).await.unwrap();

        let response = rx.await.unwrap().unwrap();
        let Message::Single {
            data: response_payload,
            ..
        } = response
        else {
            panic!("expected single")
        };
        assert_eq!(response_payload, vec![(i as u8) * 2]);
    }

    drop(sender);
    receiver_task.await.unwrap();
}

// ============================================================================
// Edge Cases
// ============================================================================

#[tokio::test]
async fn test_mailbox_empty_payload() {
    let mut mailbox = Mailbox::new();
    let sender = MailboxSender::new(mailbox.sender());
    let mut receiver = mailbox.take_receiver();

    let envelope = Envelope::tell("empty".to_string(), vec![]);
    sender.send(envelope).await.unwrap();

    let received = receiver.recv().await.unwrap();
    assert_eq!(received.msg_type(), "empty");

    if let Message::Single { data, .. } = received.message {
        assert!(data.is_empty());
    } else {
        panic!("Expected single message");
    }
}

#[tokio::test]
async fn test_mailbox_large_payload() {
    let mut mailbox = Mailbox::new();
    let sender = MailboxSender::new(mailbox.sender());
    let mut receiver = mailbox.take_receiver();

    // 1MB payload
    let large_payload: Vec<u8> = (0..1_000_000).map(|i| (i % 256) as u8).collect();
    let envelope = Envelope::tell("large".to_string(), large_payload.clone());
    sender.send(envelope).await.unwrap();

    let received = receiver.recv().await.unwrap();

    if let Message::Single { data, .. } = received.message {
        assert_eq!(data.len(), 1_000_000);
        assert_eq!(data, large_payload);
    } else {
        panic!("Expected single message");
    }
}

#[tokio::test]
async fn test_mailbox_default_size() {
    // Verify the default mailbox size constant
    assert_eq!(DEFAULT_MAILBOX_SIZE, 256);

    let mailbox = Mailbox::new();
    let sender = mailbox.sender();

    // Should be able to queue up to DEFAULT_MAILBOX_SIZE without blocking
    // (Note: This is a simplified test, actual behavior depends on implementation)
    assert!(!sender.is_closed());
}
