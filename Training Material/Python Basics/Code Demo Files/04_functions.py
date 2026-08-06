def format_prompt(user_input: str) -> str:
    """
    This function takes user input (a topic) 
    and returns a formatted string prompt.
    """
    return f"Explain simply: {user_input}"







result = format_prompt(input("Enter a topic you want to learn about: "))
print(result)
