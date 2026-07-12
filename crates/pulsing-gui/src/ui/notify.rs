use gpui::{App, Window};
use gpui_component::{
    notification::{Notification, NotificationType},
    WindowExt,
};

pub fn info(
    window: &mut Window,
    title: impl Into<gpui::SharedString>,
    message: impl Into<gpui::SharedString>,
    cx: &mut App,
) {
    window.push_notification(
        Notification::info(message)
            .title(title)
            .with_type(NotificationType::Info),
        cx,
    );
}

pub fn success(
    window: &mut Window,
    title: impl Into<gpui::SharedString>,
    message: impl Into<gpui::SharedString>,
    cx: &mut App,
) {
    window.push_notification(Notification::success(message).title(title), cx);
}

pub fn warning(
    window: &mut Window,
    title: impl Into<gpui::SharedString>,
    message: impl Into<gpui::SharedString>,
    cx: &mut App,
) {
    window.push_notification(Notification::warning(message).title(title), cx);
}

pub fn error(
    window: &mut Window,
    title: impl Into<gpui::SharedString>,
    message: impl Into<gpui::SharedString>,
    cx: &mut App,
) {
    window.push_notification(Notification::error(message).title(title), cx);
}
