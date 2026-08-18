# LIST
# - Ordered collection , Mutable , Allows duplicates
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Explain LLMs simply"}
]

# DICTIONARY
# - Key-value pairs , Keys must be unique
config = {
    "model": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 200
}
# TUPLE
# - Ordered , Immutable , Allows duplicates

coordinates = (24.8607, 67.0011)

# -------------------------------
# SET
# - Unordered , Mutable , Does NOT allow duplicates
unique_roles = {"system", "user", "assistant", "user"}


# -------------------------------
# Printing outputs
print("List (messages):", messages)
print("Dictionary (config):", config)
print("Tuple (coordinates):", coordinates)
print("Set (unique_roles):", unique_roles)