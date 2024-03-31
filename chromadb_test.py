import chromadb

client = chromadb.PersistentClient(path="/path/to/save/to")

# returns a nanosecond heartbeat. Useful for making sure the client remains connected.
client.heartbeat()

# Empties and completely resets the database. ⚠️ This is destructive and not reversible.
# client.reset()

collection = client.create_collection(name="my_collection")
collection = client.get_collection(name="my_collection")
