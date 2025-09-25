from langchain.text_splitter import RecursiveCharacterTextSplitter, Language


text = """import random
import json
import os

SCORE_FILE = "scores.json"

def load_score():
    if not os.path.exists(SCORE_FILE):
        return {"games": 0, "wins": 0}
    with open(SCORE_FILE, "r") as f:
        return json.load(f)

def save_score(score):
    with open(SCORE_FILE, "w") as f:
        json.dump(score, f)

def play_game():
    number = random.randint(1, 100)
    attempts = 0
    while True:
        guess = input("Guess a number between 1 and 100 (or q to quit): ")
        if guess.lower() == "q":
            return False
        try:
            guess = int(guess)
        except:
            print("Enter a valid number.")
            continue
        attempts += 1
        if guess < number:
            print("Too low!")
        elif guess > number:
            print("Too high!")
        else:
            print(f"Correct! You took {attempts} attempts.")
            return True

def main():
    score = load_score()
    while True:
        print(f"\nGames played: {score['games']} Wins: {score['wins']}")
        start = input("Play a game? (y/n): ").strip().lower()
        if start != "y":
            break
        score["games"] += 1
        win = play_game()
        if win:
            score["wins"] += 1
        save_score(score)
    print("Goodbye!")

if __name__ == "__main__":
    main()

"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size = 300,
    chunk_overlap = 0
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks)