//! Live actor registry backed by an MEMH hash table (mmap'd file).
//!
//! Key = actor name, value = `actor_id|node_id|type|module`.
//! On-disk file `pulsing.actors`; probing SQL: `SELECT * FROM pulsing.actors`.

use crate::actor::{ActorId, NodeId, StopReason};
use probing_memtable::discover::ExposedHashTable;
use probing_memtable::Value;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

static ACTOR_HT: OnceLock<Mutex<ExposedHashTable>> = OnceLock::new();

pub(crate) fn init_actor_memtable() {
    match ExposedHashTable::create("pulsing.actors", 256, 65536, 0) {
        Ok(table) => {
            let _ = ACTOR_HT.set(Mutex::new(table));
        }
        Err(e) => {
            eprintln!("pulsing: failed to create actor hash table: {e}");
        }
    }
}

pub(crate) fn write_actor_spawned(
    name: &str,
    actor_id: ActorId,
    node_id: NodeId,
    metadata: &HashMap<String, String>,
) {
    let Some(ht) = ACTOR_HT.get() else { return };
    let Ok(mut table) = ht.lock() else { return };

    let actor_type = metadata
        .get("class")
        .or_else(|| metadata.get("type"))
        .map(|s| s.as_str())
        .unwrap_or("");
    let module = metadata.get("module").map(|s| s.as_str()).unwrap_or("");

    let val = format!("{}|{}|{}|{}", actor_id, node_id, actor_type, module);
    let _ = table.writer().insert(name, &Value::Str(&val));
}

pub(crate) fn write_actor_stopped(
    name: &str,
    _actor_id: ActorId,
    _node_id: NodeId,
    _reason: &StopReason,
) {
    let Some(ht) = ACTOR_HT.get() else { return };
    let Ok(mut table) = ht.lock() else { return };
    table.writer().remove(name);
}
