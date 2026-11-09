// 内置大屏 — 从账本生成 Mermaid 泳道时序图 HTML（无外部依赖, 打开即看）

pub fn render_sequence_html(journal: &[serde_json::Value], title: &str, out_path: &str) {
    let mut lines = vec!["sequenceDiagram".to_string(), " autonumber".to_string()];
    // 参与者按首次出场
    let mut order: Vec<String> = Vec::new();
    for v in journal {
        for key in ["from", "to"] {
            let name = v[key].as_str().unwrap_or("");
            if !name.is_empty() && name != "orchestrator" && !order.contains(&name.to_string()) {
                order.push(name.to_string());
            }
        }
    }
    if !order.contains(&"orchestrator".to_string()) {
        lines.push(" participant U as 编排器".to_string());
    }
    for name in &order {
        lines.push(format!(
            " participant P{} as {}",
            order.iter().position(|x| x == name).unwrap(),
            name
        ));
    }
    let alias = |name: &str| -> String {
        if name == "orchestrator" {
            "U".to_string()
        } else {
            format!("P{}", order.iter().position(|x| x == name).unwrap_or(0))
        }
    };
    for v in journal {
        let from = alias(v["from"].as_str().unwrap_or(""));
        let to = alias(v["to"].as_str().unwrap_or(""));
        let ts = {
            let ms = v["ts_ms"].as_u64().unwrap_or(0);
            let secs = ms / 1000;
            let h = (secs / 3600) % 24;
            let m = (secs / 60) % 60;
            let s = secs % 60;
            format!("{h:02}:{m:02}:{s:02}")
        };
        let body = v["body"].as_str().unwrap_or("");
        let kind = if body.starts_with("TASK=") {
            "TASK"
        } else if body.starts_with("DRAFT=") {
            "DRAFT"
        } else if body.starts_with("REVISED=") {
            "REVISED"
        } else if body.starts_with("REJECT=") {
            "REJECT"
        } else if body.starts_with("REPORT=") {
            "REPORT"
        } else if body.starts_with("CLEAN=") {
            "CLEAN"
        } else if body.starts_with("NEEDS_FIX=") {
            "NEEDS_FIX"
        } else if body.starts_with("KICK=") {
            "KICK"
        } else if body.starts_with("REVIEW=") {
            "群审"
        } else {
            "信"
        };
        let excerpt: String = body
            .chars()
            .skip(body.find("||").map(|i| i + 2).unwrap_or(0))
            .take(30)
            .collect();
        lines.push(format!(" {from}-->>{to}: {ts} {kind} {excerpt}"));
    }
    let mmd = lines.join("\n");
    let html = format!(
        r#"<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<style>*{{margin:0;padding:0}}body{{background:#fff;font-family:monospace;overflow:auto}}</style>
</head><body>
<div id="c" style="padding:20px"></div>
<script>
mermaid.initialize({{startOnLoad:false,theme:'default',
  sequence:{{actorMargin:70,messageFontSize:12,noteFontSize:11,useMaxWidth:false}}}});
const mmd = {mmd:?};
mermaid.render('g', mmd).then(({{svg}}) => {{
  document.getElementById('c').innerHTML = svg;
}}).catch(e => document.body.innerHTML = '<pre>'+e+'</pre>');
</script></body></html>"#
    );
    std::fs::write(out_path, html).ok();
}
