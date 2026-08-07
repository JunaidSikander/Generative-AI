# Ask the user for their name (input)
trainer_name = "Muneer"
user_name = input("Hello my name is " + trainer_name + ", what is your name? ")

# Ask the user for their age (input)
user_age = input("Enter your age: ")

# Here we assume age is a number, so we convert it to int
next_year_age = int(user_age) + 1


# Print output using variables
print("Hello,", user_name)
print("You are", user_age, "years old.")
print("Next year, you will be", next_year_age, "years old.")