"""
RAG (Retrieval-Augmented Generation) Book Recommender
======================================================
This module implements a simple RAG pipeline that predicts how much a user
will enjoy a book, based on their past book reviews.

How it works (3 steps):
  1. EMBED   — convert every review into a numeric vector (a list of floats)
               that captures the "meaning" of the text.
  2. RETRIEVE — when given a new book description, find the reviews whose
               vectors are closest to the book's vector (most relevant past
               opinions).
  3. GENERATE — send the book description + retrieved reviews to an LLM
               (Claude) and ask it to predict a 1-5 enjoyment rating.

This pattern is called RAG because we *retrieve* relevant context from our
own data before asking the LLM to *generate* an answer. Without retrieval,
the LLM would have no idea what this particular user likes.

Embeddings in this version
--------------------------
We use TF-IDF (Term Frequency–Inverse Document Frequency), a classical NLP
technique built into scikit-learn. The original book example used OpenAI's
embedding API instead, but we replaced it to avoid the paid dependency:

  - TF-IDF works by counting words and weighting rare words more highly.
  - It is fast, free, and requires no internet connection.
  - The trade-off: it only matches on exact words, not meaning. So
    "backpacker" and "hiker" would NOT be considered similar, whereas
    a neural embedding model would recognise them as related concepts.

Vector search with FAISS
------------------------
FAISS (Facebook AI Similarity Search) stores all the review vectors in an
index and lets us search for the nearest neighbours of any query vector very
efficiently. Here we use the simplest index type: IndexFlatL2, which does an
exact brute-force search using Euclidean (L2) distance.

LLM call with Claude
--------------------
Claude is called via the Anthropic Python SDK. The SDK reads the API key
from the ANTHROPIC_API_KEY environment variable automatically — no need to
pass it explicitly. We use claude-haiku-4-5, the smallest and cheapest
Claude model, which is more than capable for this simple rating task.
"""

import logging
import numpy as np
import faiss
import anthropic
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv

# Configure a logger for this module.
# Using __name__ is standard practice — it means the logger is named
# "book_recommender", which makes it easy to filter in larger applications.
# Log level INFO shows the pipeline steps; use DEBUG for even more detail.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Load ANTHROPIC_API_KEY from .env file if present.
# This makes the code work in IntelliJ, VS Code, and any other IDE
# without manually setting environment variables in run configurations.
# The .env file is gitignored so the key is never committed.
load_dotenv()

# --- Shared state ----------------------------------------------------------
# The vectorizer must be fitted on the reviews corpus first (inside
# index_reviews) so that it knows the vocabulary. After fitting, the same
# vocabulary must be used when transforming the query in retrieve_reviews.
# Using a module-level instance ensures both functions share the same state.
_vectorizer = TfidfVectorizer()

# The Anthropic client reads ANTHROPIC_API_KEY from the environment.
# It is created once at module level so it is reused across calls.
client = anthropic.Anthropic()

# --- Sample data -----------------------------------------------------------
# Simulated history of reviews written by a single user. In a real app these
# would be stored in a database and loaded per user.
reviews = [
    "I hate stories about backpacking. It's boring.",
    "A moving exploration of racial injustice and moral growth.",
    "Compelling dystopia, but overwhelmingly bleak.",
    "Timeless romance with sharp social commentary.",
    "Epic sea adventure with philosophical depth.",
    "Mesmerizing magic and romance with rich world-building.",
    "Beautifully descriptive, but predictable plot.",
    "A detailed and emotional journey through loss and art.",
    "Fresh take on Greek mythology, but pacing dragged.",
    "Brilliant exploration of complex relationships and personal growth.",
    "Another bland romantic utopia. This time on a tropical island.",
]


# ---------------------------------------------------------------------------
# Step 1 — Build the vector index from all reviews
# ---------------------------------------------------------------------------
def index_reviews(reviews):
    """Convert all reviews to TF-IDF vectors and store them in a FAISS index.

    Args:
        reviews: List of review strings.

    Returns:
        A FAISS index ready for nearest-neighbour search.

    What happens inside:
      - fit_transform() learns the vocabulary from the reviews AND transforms
        each review into a sparse vector of TF-IDF scores. Every position in
        the vector corresponds to one word in the vocabulary; the value is
        how important that word is in that review.
      - toarray() converts the sparse matrix to a dense numpy array because
        FAISS requires dense float32 arrays.
      - IndexFlatL2 stores the vectors and searches by L2 (Euclidean) distance
        — smaller distance means more similar.
    """
    # Shape: (num_reviews, vocab_size)
    vectors = _vectorizer.fit_transform(reviews).toarray().astype("float32")

    # vocab_size becomes the "dimension" of our vector space
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    logger.info("Indexed %d reviews (vector dimension: %d)", len(reviews), vectors.shape[1])
    return index


# ---------------------------------------------------------------------------
# Step 2 — Retrieve the most relevant reviews for a given book
# ---------------------------------------------------------------------------
def retrieve_reviews(index, query, reviews, k=2):
    """Find the k reviews most similar to the book description.

    Args:
        index:   The FAISS index built by index_reviews().
        query:   A short description of the book we want to predict.
        reviews: The original list of review strings (used to look up text
                 by the integer index returned by FAISS).
        k:       How many reviews to retrieve (default: 2).

    Returns:
        A list of k review strings.

    What happens inside:
      - transform() (not fit_transform) converts the query using the
        *existing* vocabulary learned during index_reviews(). Words in the
        query that were not in the training reviews get ignored.
      - index.search() returns the k nearest vectors and their integer
        positions in the index, which map back to positions in `reviews`.
    """
    # Shape: (1, vocab_size) — FAISS expects a 2-D array even for one query
    query_vector = _vectorizer.transform([query]).toarray().astype("float32")

    # distances shape: (1, k), indices shape: (1, k)
    _, indices = index.search(query_vector, k)

    retrieved = [reviews[i] for i in indices[0]]
    logger.info("Request: %r", query)
    for i, review in enumerate(retrieved, 1):
        logger.info("  Recommended review %d: %r", i, review)
    return retrieved


# ---------------------------------------------------------------------------
# Step 3 — Ask Claude to predict the rating
# ---------------------------------------------------------------------------
def predict_rating(book, related_reviews):
    """Use Claude to predict a 1-5 enjoyment rating for the book.

    Args:
        book:            Description of the book to evaluate.
        related_reviews: List of relevant past reviews retrieved in Step 2.

    Returns:
        A string containing a single digit ("1"–"5").

    Prompt design notes:
      - We provide the book description and the retrieved reviews as context.
        This is the "augmented" part of RAG — we are grounding the LLM's
        response in the user's actual preferences rather than asking it to
        guess from scratch.
      - We instruct the model to reply with just a number to make parsing
        trivial. This is a simple form of output formatting.
      - max_tokens=10 is intentionally tiny — we only expect a single digit.
        Keeping it small reduces cost and latency.
      - We use claude-haiku-4-5, the smallest Claude model. For this simple
        task it performs just as well as larger models at a fraction of the cost.
    """
    reviews_text = "\n".join(related_reviews)

    # The prompt is the key piece of "prompt engineering" here.
    # It clearly defines the task, provides all relevant context, and
    # constrains the output format.
    prompt = (
            "Here is a book I might want to read:\n" +
            book + "\n\n" +
            "Here are relevant reviews from the past:\n" +
            reviews_text + "\n\n" +
            "On a scale of 1 (worst) to 5 (best), "
            "how likely am I to enjoy this book? "
            "Reply with no explanation, just a number."
    )

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )

    # response.content is a list of content blocks. For a plain text reply
    # there is always exactly one block of type "text".
    rating = response.content[0].text
    logger.info("Predicted rating for %r: %s/5", book[:60] + "...", rating)
    return rating
