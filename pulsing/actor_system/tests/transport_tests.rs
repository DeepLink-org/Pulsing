//! TCP Transport layer tests

use pulsing_actor::actor::ActorId;
use pulsing_actor::transport::TransportMessage;

// ============================================================================
// Transport Message Tests
// ============================================================================

#[test]
fn test_transport_message_request() {
    let msg = TransportMessage::Request {
        id: 123,
        actor_id: ActorId::local("test"),
        msg_type: "Ping".to_string(),
        payload: vec![1, 2, 3],
    };

    assert!(matches!(msg, TransportMessage::Request { id: 123, .. }));
}

#[test]
fn test_transport_message_response() {
    let msg = TransportMessage::Response {
        id: 456,
        result: Ok(vec![4, 5, 6]),
    };

    match msg {
        TransportMessage::Response { id, result } => {
            assert_eq!(id, 456);
            assert!(result.is_ok());
            assert_eq!(result.unwrap(), vec![4, 5, 6]);
        }
        _ => panic!("Expected Response"),
    }
}

#[test]
fn test_transport_message_oneway() {
    let msg = TransportMessage::OneWay {
        actor_id: ActorId::local("actor"),
        msg_type: "Fire".to_string(),
        payload: vec![7, 8, 9],
    };

    match msg {
        TransportMessage::OneWay {
            actor_id,
            msg_type,
            payload,
        } => {
            assert_eq!(actor_id.name, "actor");
            assert_eq!(msg_type, "Fire");
            assert_eq!(payload, vec![7, 8, 9]);
        }
        _ => panic!("Expected OneWay"),
    }
}

#[test]
fn test_transport_message_request_helper() {
    let actor_id = ActorId::local("test-actor");
    let (id, msg) = TransportMessage::request(actor_id, "TestType".to_string(), vec![1, 2]);

    match msg {
        TransportMessage::Request {
            id: req_id,
            actor_id: req_actor,
            msg_type,
            payload,
        } => {
            assert_eq!(req_id, id);
            assert_eq!(req_actor.name, "test-actor");
            assert_eq!(msg_type, "TestType");
            assert_eq!(payload, vec![1, 2]);
        }
        _ => panic!("Expected Request"),
    }
}

#[test]
fn test_transport_message_response_helper() {
    let msg = TransportMessage::response(999, Ok(vec![42]));

    match msg {
        TransportMessage::Response { id, result } => {
            assert_eq!(id, 999);
            assert_eq!(result.unwrap(), vec![42]);
        }
        _ => panic!("Expected Response"),
    }
}

#[test]
fn test_transport_message_oneway_helper() {
    let actor_id = ActorId::local("one-way-actor");
    let msg = TransportMessage::one_way(actor_id, "FireAndForget".to_string(), vec![100]);

    match msg {
        TransportMessage::OneWay {
            actor_id,
            msg_type,
            payload,
        } => {
            assert_eq!(actor_id.name, "one-way-actor");
            assert_eq!(msg_type, "FireAndForget");
            assert_eq!(payload, vec![100]);
        }
        _ => panic!("Expected OneWay"),
    }
}

#[test]
fn test_transport_message_request_id() {
    let actor_id = ActorId::local("test");
    let (id, msg) = TransportMessage::request(actor_id, "Test".to_string(), vec![]);

    assert_eq!(msg.request_id(), Some(id));

    let response = TransportMessage::response(456, Ok(vec![]));
    assert_eq!(response.request_id(), None);

    let oneway = TransportMessage::one_way(ActorId::local("test"), "Test".to_string(), vec![]);
    assert_eq!(oneway.request_id(), None);
}

#[test]
fn test_transport_message_ping_pong() {
    let ping = TransportMessage::Ping { seq: 12345 };
    let pong = TransportMessage::Pong { seq: 12345 };

    assert!(matches!(ping, TransportMessage::Ping { seq: 12345 }));
    assert!(matches!(pong, TransportMessage::Pong { seq: 12345 }));
}
