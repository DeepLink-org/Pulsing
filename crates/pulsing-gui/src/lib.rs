mod app;
mod controller;
mod dock;
mod model;
mod panels;
mod settings;
mod shell;
mod state;
mod ui;

use pulsing_forge::InteractiveConfig;

pub fn run(agent: InteractiveConfig) -> anyhow::Result<()> {
    app::run(agent)
}
