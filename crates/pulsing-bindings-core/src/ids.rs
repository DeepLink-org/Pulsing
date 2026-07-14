//! NodeId / ActorId parsing helpers shared across binding paths.

use pulsing_actor::actor::{ActorId, NodeId};

#[derive(Debug, Clone, Copy)]
pub struct PyNodeIdView(pub NodeId);

#[derive(Debug, Clone, Copy)]
pub struct PyActorIdView(pub ActorId);

impl PyNodeIdView {
    pub fn generate() -> Self {
        Self(NodeId::generate())
    }

    pub fn local() -> Self {
        Self(NodeId::LOCAL)
    }

    pub fn id(&self) -> u128 {
        self.0 .0
    }

    pub fn uuid(&self) -> String {
        self.0.to_string()
    }

    pub fn is_local(&self) -> bool {
        self.0.is_local()
    }
}

impl PyActorIdView {
    pub fn generate() -> Self {
        Self(ActorId::generate())
    }

    pub fn id(&self) -> u128 {
        self.0 .0
    }

    pub fn uuid(&self) -> String {
        self.0.to_string()
    }
}

pub fn parse_node_id(id: Option<&str>, as_u128: Option<u128>) -> Result<NodeId, String> {
    if let Some(s) = id {
        let uuid = uuid::Uuid::parse_str(s).map_err(|e| e.to_string())?;
        return Ok(NodeId::new(uuid.as_u128()));
    }
    if let Some(n) = as_u128 {
        return Ok(NodeId::new(n));
    }
    Ok(NodeId::generate())
}

pub fn parse_actor_id(id: Option<&str>, as_u128: Option<u128>) -> Result<ActorId, String> {
    if let Some(s) = id {
        let uuid = uuid::Uuid::parse_str(s).map_err(|e| e.to_string())?;
        return Ok(ActorId::new(uuid.as_u128()));
    }
    if let Some(n) = as_u128 {
        return Ok(ActorId::new(n));
    }
    Ok(ActorId::generate())
}
