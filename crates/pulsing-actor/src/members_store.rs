//! Cluster membership state backed by an MEMH hash table (mmap'd file).
//!
//! Key = `node_id`, value = pipe-separated `addr|status|epoch`.
//! Gossip state changes call `upsert_member` / `remove_member`.
//! Probing SQL: `SELECT * FROM pulsing.members`.

use crate::actor::NodeId;
use crate::cluster::NodeStatus;
use probing_memtable::discover::ExposedHashTable;
use probing_memtable::Value;
use std::net::SocketAddr;
use std::sync::{Mutex, Once, OnceLock};

static MEMBERS_HT: OnceLock<Mutex<ExposedHashTable>> = OnceLock::new();
static INIT_MEMBERS_MEMTABLE: Once = Once::new();

pub(crate) fn init_members_memtable() {
    // 64 buckets, 16 KiB arena — plenty for a cluster of dozens of nodes
    INIT_MEMBERS_MEMTABLE.call_once(|| {
        match ExposedHashTable::create("pulsing.members", 64, 16384, 0) {
            Ok(table) => {
                let _ = MEMBERS_HT.set(Mutex::new(table));
            }
            Err(e) => {
                eprintln!("pulsing: failed to create members hash table: {e}");
            }
        }
    });
}

fn status_str(s: NodeStatus) -> &'static str {
    match s {
        NodeStatus::Online => "online",
        NodeStatus::PFail => "suspect",
        NodeStatus::Fail => "fail",
        NodeStatus::Handshake => "handshake",
        NodeStatus::Tombstone => "tombstone",
    }
}

/// Insert or update a member's current state in the hash table.
pub(crate) fn upsert_member(node_id: NodeId, addr: SocketAddr, new_status: NodeStatus, epoch: u64) {
    let Some(ht) = MEMBERS_HT.get() else { return };
    let Ok(mut table) = ht.lock() else { return };

    let key = node_id.to_string();
    let val = format!("{}|{}|{}", addr, status_str(new_status), epoch,);
    let _ = table.writer().insert(&key, &Value::Str(&val));
}

/// Remove a member from the hash table (permanent removal after tombstone).
pub(crate) fn remove_member(node_id: NodeId) {
    let Some(ht) = MEMBERS_HT.get() else { return };
    let Ok(mut table) = ht.lock() else { return };

    let key = node_id.to_string();
    table.writer().remove(&key);
}
