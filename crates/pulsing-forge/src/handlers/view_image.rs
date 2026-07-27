use std::path::Path;

use base64::Engine;
use image::imageops::FilterType;
use image::{GenericImageView, ImageFormat};
use serde_json::Value;
use serde_json::json;

use super::write::resolve_within_cwd;
use super::{err, json_str};
use crate::context::ToolCallContext;
use crate::error::ToolError;
use crate::executor::{ToolExecutor, ToolExecutorFuture};

const VIEW_IMAGE_CAP: usize = 8 * 1024 * 1024;
/// Matches `codex_utils_image::MAX_DIMENSION`: the longer side is capped at
/// this many pixels before the image is attached at "high" detail.
const HIGH_DETAIL_MAX_PX: u32 = 2048;

pub struct ViewImageHandler;

impl ToolExecutor for ViewImageHandler {
    fn tool_name(&self) -> &str {
        "view_image"
    }

    fn spec(&self) -> crate::registry::ToolSpec {
        super::builtin_spec(self.tool_name())
    }

    fn handle<'a>(&'a self, ctx: &'a ToolCallContext, arguments: Value) -> ToolExecutorFuture<'a> {
        let cwd = ctx.cwd.clone();
        Box::pin(async move { view_image_impl(&cwd, &arguments) })
    }
}

fn view_image_impl(cwd: &Path, args: &Value) -> Result<crate::result::ToolResult, ToolError> {
    let path = json_str(args, "path")?;
    let detail = args
        .get("detail")
        .and_then(|v| v.as_str())
        .unwrap_or("high");
    if !matches!(detail, "high" | "original") {
        return err(format!(
            "view_image.detail only supports `high` or `original`; omit `detail` for default \
             high resized behavior, got `{detail}`"
        ));
    }

    let abs = resolve_within_cwd(cwd, path).map_err(ToolError::respond)?;
    if !abs.is_file() {
        return err(format!("image path `{}` is not a file", abs.display()));
    }

    let raw = std::fs::read(&abs).map_err(|e| {
        ToolError::respond(format!("unable to read image at `{}`: {e}", abs.display()))
    })?;
    if raw.len() > VIEW_IMAGE_CAP {
        return err(format!(
            "Image too large for view_image: {} bytes exceeds the {} byte cap.",
            raw.len(),
            VIEW_IMAGE_CAP
        ));
    }

    let (bytes, mime) = encode_for_prompt(&raw, detail)?;
    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
    let data_url = format!("data:{mime};base64,{b64}");

    let structured = json!({
        "content_items": [{
            "type": "input_image",
            "image_url": data_url,
            "detail": detail,
        }],
        "path": abs.to_string_lossy(),
        "bytes": bytes.len(),
    });

    Ok(crate::result::ToolResult {
        content: format!(
            "Attached image {} (detail={detail}, {} bytes)",
            abs.display(),
            bytes.len()
        ),
        is_error: false,
        structured: Some(structured),
    })
}

/// Determines the MIME type from the file's magic bytes rather than its
/// extension, so a mislabeled extension (e.g. a JPEG named `foo.png`) can't
/// smuggle in a mismatched `data:` MIME type. Mirrors how Codex's
/// `codex_utils_image::load_for_prompt_bytes` sniffs the format via
/// `image::guess_format` before trusting any file metadata.
fn sniff_format(raw: &[u8]) -> Result<ImageFormat, ToolError> {
    image::guess_format(raw)
        .map_err(|_| ToolError::respond("not a recognized image format (png/jpeg/gif/webp)"))
}

fn format_to_mime(format: ImageFormat) -> Result<&'static str, ToolError> {
    match format {
        ImageFormat::Png => Ok("image/png"),
        ImageFormat::Jpeg => Ok("image/jpeg"),
        ImageFormat::Gif => Ok("image/gif"),
        ImageFormat::WebP => Ok("image/webp"),
        other => Err(ToolError::respond(format!(
            "unsupported image format: {other:?}"
        ))),
    }
}

fn encode_for_prompt(raw: &[u8], detail: &str) -> Result<(Vec<u8>, String), ToolError> {
    let format = sniff_format(raw)?;
    let mime = format_to_mime(format)?;

    if detail == "original" {
        return Ok((raw.to_vec(), mime.to_string()));
    }

    let img = image::load_from_memory_with_format(raw, format)
        .map_err(|e| ToolError::respond(e.to_string()))?;
    let (w, h) = img.dimensions();
    if w.max(h) <= HIGH_DETAIL_MAX_PX {
        return Ok((raw.to_vec(), mime.to_string()));
    }

    let resized = img.resize(HIGH_DETAIL_MAX_PX, HIGH_DETAIL_MAX_PX, FilterType::Triangle);
    let mut buf = Vec::new();
    resized
        .write_to(&mut std::io::Cursor::new(&mut buf), format)
        .map_err(|e| ToolError::respond(e.to_string()))?;
    Ok((buf, mime.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::path::Path;

    // Valid 1x1 RGBA PNG (CRC-checked).
    const TINY_PNG: &[u8] = &[
        0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44,
        0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1f,
        0x15, 0xc4, 0x89, 0x00, 0x00, 0x00, 0x0a, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x63, 0x00,
        0x01, 0x00, 0x00, 0x05, 0x00, 0x01, 0x0d, 0x0a, 0x2d, 0xb4, 0x00, 0x00, 0x00, 0x00, 0x49,
        0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
    ];

    fn write_tiny_png(path: &Path) {
        std::fs::write(path, TINY_PNG).expect("write png");
    }

    fn run(cwd: &Path, args: Value) -> crate::result::ToolResult {
        match view_image_impl(cwd, &args) {
            Ok(r) => r,
            Err(e) => crate::result::ToolResult::err(e.to_string()),
        }
    }

    #[test]
    fn attaches_png_with_structured_output() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x.png");
        write_tiny_png(&path);
        let out = run(dir.path(), json!({"path": "x.png", "detail": "high"}));
        assert!(!out.is_error, "view_image failed: {}", out.content);
        let structured = out.structured.expect("structured");
        let items = structured["content_items"].as_array().expect("items");
        let url = items[0]["image_url"].as_str().expect("url");
        assert!(url.starts_with("data:image/png;base64,"));
    }

    #[test]
    fn rejects_relative_escape_outside_cwd() {
        let dir = tempfile::tempdir().unwrap();
        let outside = dir.path().parent().unwrap().join("escape.png");
        write_tiny_png(&outside);
        let out = run(dir.path(), json!({"path": "../escape.png"}));
        assert!(out.is_error);
        assert!(out.content.contains("outside working directory"));
    }

    #[test]
    fn rejects_absolute_path_outside_cwd() {
        let dir = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let path = outside.path().join("outside.png");
        write_tiny_png(&path);
        let out = run(dir.path(), json!({"path": path.to_str().expect("utf8")}));
        assert!(out.is_error);
        assert!(out.content.contains("outside working directory"));
    }

    #[test]
    fn rejects_invalid_detail() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny_png(&dir.path().join("x.png"));
        let out = run(dir.path(), json!({"path": "x.png", "detail": "low"}));
        assert!(out.is_error);
        assert!(out.content.contains("view_image.detail only supports"));
    }

    #[test]
    fn rejects_unrecognized_format() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("x.bin"), b"not an image").unwrap();
        let out = run(dir.path(), json!({"path": "x.bin"}));
        assert!(out.is_error);
        assert!(out.content.contains("not a recognized image format"));
    }
}
