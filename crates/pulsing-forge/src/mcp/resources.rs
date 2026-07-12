//! MCP resource URI validation and response size limits.

use rmcp::model::{ReadResourceResult, ResourceContents};

/// Max decoded/encoded payload per `read_mcp_resource` response (aligned with `Read` tool cap).
pub const MAX_MCP_RESOURCE_BYTES: usize = 2 * 1024 * 1024;

const MAX_MCP_RESOURCE_URI_BYTES: usize = 8 * 1024;

pub fn validate_mcp_resource_uri(uri: &str) -> Result<(), String> {
    let trimmed = uri.trim();
    if trimmed.is_empty() {
        return Err("uri must be a non-empty string".into());
    }
    if trimmed.len() > MAX_MCP_RESOURCE_URI_BYTES {
        return Err(format!(
            "uri too long: {} bytes (max {MAX_MCP_RESOURCE_URI_BYTES})",
            trimmed.len()
        ));
    }
    let parsed = url::Url::parse(trimmed).map_err(|e| format!("invalid uri: {e}"))?;
    if parsed.scheme().is_empty() {
        return Err("uri must include a scheme (e.g. file://, https://)".into());
    }
    Ok(())
}

pub fn resource_content_bytes(content: &ResourceContents) -> usize {
    match content {
        ResourceContents::TextResourceContents { text, .. } => text.len(),
        ResourceContents::BlobResourceContents { blob, .. } => blob.len(),
    }
}

pub fn total_resource_bytes(result: &ReadResourceResult) -> usize {
    result.contents.iter().map(resource_content_bytes).sum()
}

pub fn enforce_resource_size_limit(
    result: ReadResourceResult,
) -> Result<ReadResourceResult, String> {
    let total = total_resource_bytes(&result);
    if total > MAX_MCP_RESOURCE_BYTES {
        return Err(format!(
            "MCP resource content too large: {total} bytes (max {MAX_MCP_RESOURCE_BYTES})"
        ));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmcp::model::ReadResourceResult;

    #[test]
    fn validate_uri_rejects_empty_and_relative() {
        assert!(validate_mcp_resource_uri("").is_err());
        assert!(validate_mcp_resource_uri("   ").is_err());
        assert!(validate_mcp_resource_uri("relative/path").is_err());
    }

    #[test]
    fn validate_uri_accepts_common_schemes() {
        assert!(validate_mcp_resource_uri("file:///tmp/x").is_ok());
        assert!(validate_mcp_resource_uri("https://example.com/r").is_ok());
        assert!(validate_mcp_resource_uri("mcp://server/resource/1").is_ok());
    }

    #[test]
    fn validate_uri_rejects_too_long() {
        let long = format!("file:///{}", "a".repeat(8 * 1024));
        let err = validate_mcp_resource_uri(&long).unwrap_err();
        assert!(err.contains("uri too long"));
    }

    #[test]
    fn enforce_size_limit_rejects_oversized_payload() {
        let big = "x".repeat(MAX_MCP_RESOURCE_BYTES + 1);
        let result = ReadResourceResult::new(vec![ResourceContents::text(big, "file:///x")]);
        let err = enforce_resource_size_limit(result).unwrap_err();
        assert!(err.contains("too large"));
    }

    #[test]
    fn enforce_size_limit_sums_multiple_contents() {
        let half = MAX_MCP_RESOURCE_BYTES / 2 + 1;
        let result = ReadResourceResult::new(vec![
            ResourceContents::text("x".repeat(half), "file:///a"),
            ResourceContents::text("y".repeat(half), "file:///b"),
        ]);
        let err = enforce_resource_size_limit(result).unwrap_err();
        assert!(err.contains("too large"));
    }

    #[test]
    fn enforce_size_limit_counts_blob_bytes() {
        let big = "A".repeat(MAX_MCP_RESOURCE_BYTES + 1);
        let result = ReadResourceResult::new(vec![ResourceContents::blob(big, "file:///x")]);
        let err = enforce_resource_size_limit(result).unwrap_err();
        assert!(err.contains("too large"));
    }
}
