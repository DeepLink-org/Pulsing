import pulsing.core as core
import pulsing._core as c

s = core.get_system()
print("type", type(s))
print("has ActorSystem.spawn attr", hasattr(c.ActorSystem, "spawn"))
print("getattr spawn", getattr(c.ActorSystem, "spawn", None))
print("getattr create", getattr(c.ActorSystem, "create", None))
print("NodeId methods", [x for x in dir(c.NodeId) if not x.startswith("_")])
print("ActorRef all", dir(c.ActorRef))
print(
    "ActorSystemTest",
    hasattr(c, "ActorSystemTest"),
    getattr(c, "ActorSystemTest", None),
)
if hasattr(c, "ActorSystemTest"):
    print("ping", c.ActorSystemTest().ping())
