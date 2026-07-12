//! Line input (plain stdin; reedline/TUI can replace this module later).

use std::io::{self, Write};

pub fn read_line(prompt: &str) -> io::Result<Option<String>> {
    print!("{prompt}");
    let _ = io::stdout().flush();
    let mut line = String::new();
    let n = io::stdin().read_line(&mut line)?;
    if n == 0 {
        return Ok(None);
    }
    Ok(Some(line))
}

pub fn confirm(prompt: &str, default_no: bool) -> io::Result<bool> {
    eprint!("{prompt}");
    let _ = io::stderr().flush();
    let mut line = String::new();
    if io::stdin().read_line(&mut line)? == 0 {
        return Ok(false);
    }
    let answer = line.trim().to_lowercase();
    if answer.is_empty() {
        return Ok(!default_no);
    }
    Ok(matches!(answer.as_str(), "y" | "yes" | "retry"))
}
