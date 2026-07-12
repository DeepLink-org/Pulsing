use gpui::{
    div, App, ClickEvent, Context, Entity, EventEmitter, FocusHandle, Focusable, IntoElement,
    ParentElement, Render, Styled, WeakEntity, Window,
};
use gpui_component::{
    button::Button,
    dock::{Panel, PanelEvent},
    h_flex,
    label::Label,
    sidebar::{Sidebar, SidebarGroup, SidebarMenu, SidebarMenuItem},
    tag::Tag,
    v_flex, ActiveTheme, Disableable, Sizable, StyledExt,
};

use crate::dock::layout::PANEL_AGENTS;
use crate::model::{SessionStore, WorkspaceAction, WorkspaceModel};
use crate::shell::WorkspaceShell;
use crate::ui::icons::{labeled, sym};
use crate::ui::style::{sidebar_panel, status_dot, PanelSide};

pub struct AgentsPanel {
    focus_handle: FocusHandle,
    workspace: Entity<WorkspaceModel>,
    sessions: Entity<SessionStore>,
    shell: WeakEntity<WorkspaceShell>,
}

impl AgentsPanel {
    pub fn new(
        workspace: Entity<WorkspaceModel>,
        sessions: Entity<SessionStore>,
        shell: WeakEntity<WorkspaceShell>,
        cx: &mut Context<Self>,
    ) -> Self {
        cx.observe(&workspace, |_, _, cx| cx.notify()).detach();
        cx.observe(&sessions, |_, _, cx| cx.notify()).detach();

        Self {
            focus_handle: cx.focus_handle(),
            workspace,
            sessions,
            shell,
        }
    }

    fn dispatch(&self, action: WorkspaceAction, cx: &mut Context<Self>) {
        self.shell
            .update(cx, |shell, cx| shell.dispatch(action, cx))
            .ok();
    }
}

impl Panel for AgentsPanel {
    fn panel_name(&self) -> &'static str {
        PANEL_AGENTS
    }

    fn title(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        "Agents"
    }

    fn inner_padding(&self, _: &App) -> bool {
        false
    }
}

impl EventEmitter<PanelEvent> for AgentsPanel {}
impl Focusable for AgentsPanel {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Render for AgentsPanel {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let runtime = self.workspace.read(cx).runtime.clone();
        let busy = runtime.local_busy;
        let active = self.sessions.read(cx).active_id();
        let session_items: Vec<_> = self
            .sessions
            .read(cx)
            .sessions()
            .iter()
            .map(|session| {
                let id = session.id;
                let title = session.chat.session_title();
                let is_active = id == active;
                SidebarMenuItem::new(labeled(sym::BOT, title))
                    .active(is_active)
                    .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                        this.dispatch(WorkspaceAction::FocusSession(id), cx);
                    }))
            })
            .collect();

        sidebar_panel(cx, PanelSide::Right).child(
            Sidebar::right()
                .w_full()
                .collapsible(false)
                .header(
                    v_flex()
                        .gap_1()
                        .child(div().text_sm().font_semibold().child("Sessions"))
                        .child(
                            h_flex()
                                .gap_2()
                                .items_center()
                                .child(status_dot(busy, cx))
                                .child(
                                    Label::new(if busy { "Running" } else { "Idle" })
                                        .text_xs()
                                        .text_color(cx.theme().muted_foreground),
                                ),
                        ),
                )
                .footer(
                    Button::new("agents-new")
                        .outline()
                        .w_full()
                        .justify_center()
                        .label(labeled(sym::PLUS, "New chat"))
                        .disabled(busy)
                        .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                            this.dispatch(WorkspaceAction::NewSession, cx);
                        })),
                )
                .child(SidebarGroup::new("Local").child(SidebarMenu::new().children(session_items)))
                .child(
                    SidebarGroup::new("Cluster").child(
                        SidebarMenu::new().child(
                            SidebarMenuItem::new(labeled(sym::GLOBE, "Remote agents"))
                                .suffix(Tag::secondary().small().outline().child("Soon"))
                                .disable(true),
                        ),
                    ),
                ),
        )
    }
}
