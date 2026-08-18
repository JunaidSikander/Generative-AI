import pandas as pd

# Create a dataset (dictionary)
data = {
    "prompt": ["What is LLM?", "What is GenAI?"],
    "response_length": [120, 95]
}

# Convert dictionary into a DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
print(df)

# Example: Inspecting data
print("\nFirst row using .head():")
print(df.head(1))

print("\nSummary statistics:")
print(df.describe())