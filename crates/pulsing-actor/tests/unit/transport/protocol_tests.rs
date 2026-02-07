//! Stream frame, message mode, and request type tests

// ============================================================================
// Stream Frame Tests
// ============================================================================

#[test]
fn test_stream_frame_data() {
    use pulsing_actor::transport::StreamFrame;

    let frame = StreamFrame::data("token", b"hello");
    assert_eq!(frame.msg_type, "token");
    assert!(!frame.end);
    assert!(frame.error.is_none());
    assert_eq!(frame.get_data(), b"hello");
}

#[test]
fn test_stream_frame_end() {
    use pulsing_actor::transport::StreamFrame;

    let frame = StreamFrame::end();
    assert!(frame.end);
    assert!(frame.error.is_none());
}

#[test]
fn test_stream_frame_error() {
    use pulsing_actor::transport::StreamFrame;

    let frame = StreamFrame::error("something went wrong");
    assert!(frame.end);
    assert!(frame.is_error());
    assert_eq!(frame.error.as_ref().unwrap(), "something went wrong");
}

#[test]
fn test_stream_frame_binary_roundtrip() {
    use pulsing_actor::transport::StreamFrame;

    let original = StreamFrame::data("response", b"world");
    let bytes = original.to_binary();
    let parsed = StreamFrame::from_binary(&bytes).unwrap();

    assert_eq!(parsed.msg_type, "response");
    assert_eq!(parsed.get_data(), b"world");
}

// ============================================================================
// Message Mode Tests
// ============================================================================

#[test]
fn test_message_mode_conversion() {
    use pulsing_actor::transport::MessageMode;

    assert_eq!(MessageMode::Ask.as_str(), "ask");
    assert_eq!(MessageMode::Tell.as_str(), "tell");
    assert_eq!(MessageMode::Stream.as_str(), "stream");

    assert_eq!(MessageMode::parse("ask"), Some(MessageMode::Ask));
    assert_eq!(MessageMode::parse("TELL"), Some(MessageMode::Tell));
    assert_eq!(MessageMode::parse("Stream"), Some(MessageMode::Stream));
    assert_eq!(MessageMode::parse("invalid"), None);
}

#[test]
fn test_request_type_conversion() {
    use pulsing_actor::transport::RequestType;

    assert_eq!(RequestType::Single.as_str(), "single");
    assert_eq!(RequestType::Stream.as_str(), "stream");

    assert_eq!(RequestType::parse("single"), Some(RequestType::Single));
    assert_eq!(RequestType::parse("STREAM"), Some(RequestType::Stream));
    assert_eq!(RequestType::parse("invalid"), None);
}
