//! Small cross-platform helpers for containing shell process trees.

#[cfg(unix)]
pub fn configure(command: &mut tokio::process::Command) {
    use std::os::unix::process::CommandExt;
    command.as_std_mut().process_group(0);
}

#[cfg(not(unix))]
pub fn configure(_command: &mut tokio::process::Command) {}

#[cfg(unix)]
pub fn kill(process_id: u32) {
    // A negative pid targets the process group created with process_group(0).
    unsafe {
        libc::kill(-(process_id as i32), libc::SIGKILL);
    }
}

#[cfg(not(unix))]
pub fn kill(_process_id: u32) {}

pub struct ProcessGroupGuard {
    process_id: Option<u32>,
}

impl ProcessGroupGuard {
    pub fn new(process_id: Option<u32>) -> Self {
        Self { process_id }
    }

    pub fn disarm(&mut self) {
        self.process_id = None;
    }

    pub fn kill_now(&self) {
        if let Some(process_id) = self.process_id {
            kill(process_id);
        }
    }
}

impl Drop for ProcessGroupGuard {
    fn drop(&mut self) {
        if let Some(process_id) = self.process_id {
            kill(process_id);
        }
    }
}
