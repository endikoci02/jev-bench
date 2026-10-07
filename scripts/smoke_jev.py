# scripts/smoke_jev.py
from dotenv import load_dotenv
from typesafe_sdk import Choice, TypeSafeClient

load_dotenv()
client = TypeSafeClient()
r = client.system_one(
    state="The movie was a total waste of two hours.",
    questions={
        "sentiment": Choice(
            instructions="Overall sentiment of the review",
            criteria={"positive": "Likes the movie", "negative": "Dislikes the movie"},
        )
    },
)
print(r.answers["sentiment"].choice, r.answers["sentiment"].probabilities, r.usage)
