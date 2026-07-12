use gpui::{
    div, px, AppContext, Application, Context, Entity, IntoElement, ParentElement, Render, Styled,
    Window, WindowOptions,
};
use gpui_component::{
    divider::Divider, h_flex, label::Label, v_flex, ActiveTheme, Root, StyledExt, TitleBar,
};
use pulsing_forge::InteractiveConfig;

use crate::shell::WorkspaceShell;
use crate::ui::style::{brand_mark, status_dot};

pub fn run(agent: InteractiveConfig) -> anyhow::Result<()> {
    let app = Application::new();
    app.run(move |cx| {
        cx.spawn(async move |cx| {
            cx.open_window(
                WindowOptions {
                    titlebar: Some(TitleBar::title_bar_options()),
                    ..WindowOptions::default()
                },
                |window, cx| {
                    gpui_component::init(cx);
                    gpui_component::theme::Theme::global_mut(cx).shadow = true;
                    let shell = cx.new(|cx| WorkspaceShell::new(agent, window, cx));
                    let view = cx.new(|cx| WorkspaceFrame::new(shell, cx));
                    cx.new(|cx| Root::new(view, window, cx))
                },
            )?;
            Ok::<_, anyhow::Error>(())
        })
        .detach();
        cx.activate(true);
    });
    Ok(())
}

struct WorkspaceFrame {
    shell: Entity<WorkspaceShell>,
}

impl WorkspaceFrame {
    fn new(shell: Entity<WorkspaceShell>, cx: &mut Context<Self>) -> Self {
        cx.observe(&shell, |_, _, cx| cx.notify()).detach();
        Self { shell }
    }
}

impl Render for WorkspaceFrame {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let workspace = self.shell.read(cx).workspace.read(cx);
        let runtime = workspace.runtime.clone();
        let workspace_name = workspace
            .layout
            .root
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| workspace.layout.root.display().to_string());

        div()
            .size_full()
            .flex()
            .flex_col()
            .bg(cx.theme().background)
            .child(
                div()
                    .border_b_1()
                    .border_color(cx.theme().border)
                    .bg(cx.theme().popover)
                    .shadow_sm()
                    .child(
                        TitleBar::new().child(
                            h_flex()
                                .w_full()
                                .items_center()
                                .gap_3()
                                .px_3()
                                .py_1()
                                .child(brand_mark(cx))
                                .child(
                                    v_flex()
                                        .gap(px(1.0))
                                        .child(div().text_sm().font_semibold().child("Pulsing"))
                                        .child(
                                            Label::new(workspace_name)
                                                .text_xs()
                                                .text_color(cx.theme().muted_foreground),
                                        ),
                                )
                                .child(status_dot(runtime.local_busy, cx))
                                .child(
                                    Label::new(if runtime.local_busy {
                                        "Agent running"
                                    } else {
                                        "Ready"
                                    })
                                    .text_xs()
                                    .text_color(cx.theme().muted_foreground),
                                )
                                .child(
                                    div().ml_auto().child(
                                        Label::new(runtime.session_title.clone())
                                            .text_xs()
                                            .text_color(cx.theme().muted_foreground),
                                    ),
                                ),
                        ),
                    ),
            )
            .child(Divider::horizontal())
            .child(self.shell.clone())
    }
}
