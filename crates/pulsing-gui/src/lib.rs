mod app;
mod controller;
mod model;
mod settings;
mod state;

use pulsing_forge::InteractiveConfig;

pub fn run(agent: InteractiveConfig) -> anyhow::Result<()> {
    app::run(agent)
}
