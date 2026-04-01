"""
Pulsing Connect — Out-Cluster Connector SDK

Lightweight connector for accessing actors in a Pulsing cluster without
joining the cluster membership. Connects to any cluster node (acting as
a gateway) and transparently routes requests.

Usage::

    from pulsing.connect import Connect

    conn = await Connect.to("10.0.1.1:8080")
    counter = await conn.resolve("services/counter")
    result = await counter.increment()
    await conn.close()
"""

from .proxy import Connect, ConnectProxy

__all__ = ["Connect", "ConnectProxy"]
