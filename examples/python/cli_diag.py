import pulsing._core as c

print("ActorSystem methods", [x for x in dir(c.ActorSystem) if not x.startswith("_")])
print("Message methods", [x for x in dir(c.Message) if not x.startswith("_")])
print("ActorRef methods", [x for x in dir(c.ActorRef) if not x.startswith("_")])
