"""
Tests for the RAG book recommender.

These are *integration* tests — they make real API calls to Claude and
actually run the full pipeline end-to-end. That means:
  - They require ANTHROPIC_API_KEY to be set in the environment.
  - They cost a tiny amount of money per run (fractions of a cent).
  - They are slower than unit tests (~2-3 seconds for the API round-trip).

Why integration tests instead of mocks?
  Mocking the API would let us test the code structure, but it would tell us
  nothing about whether the prompt actually works, whether the model returns
  the expected format, or whether the API key is valid. For LLM applications,
  real calls are the only way to be confident the system works.

Run with:
  pytest test_book_recommender.py -v -s

The -s flag disables output capture so you can see the print() statements
that show what the model actually returned.
"""

from book_recommender import index_reviews, retrieve_reviews, predict_rating, reviews

# The book we use as our test case — same as in the original book example.
# It's about backpacking culture, so it should match negatively against the
# user's review history (they explicitly hate backpacking stories).
BOOK = (
    "The Beach by Alex Garland critiques backpacker culture by exposing the "
    "selfishness and moral decay behind their pursuit of an untouched paradise."
)


def test_retrieve_reviews():
    """
    Tests the retrieval (vector search) part of the pipeline.

    We verify that:
      1. Exactly k=2 reviews are returned.
      2. The reviews are strings from the original list.

    Note on TF-IDF vs semantic embeddings:
      The original book example used OpenAI's neural embeddings, which
      understand meaning. With those, the retriever would correctly return
      the "backpacking" and "tropical island" reviews as most relevant.

      With TF-IDF (keyword matching), the retriever finds reviews that share
      exact words with the book description — here "moral" and "exploration"
      happen to score highest. The results are different, but the overall
      pipeline still works: Claude receives *some* past reviews as context.

      If you want semantically accurate retrieval, the fix is to swap
      TfidfVectorizer for a neural embedding model (e.g. OpenAI embeddings
      or a local sentence-transformers model once PyTorch is upgraded).
    """
    index = index_reviews(reviews)
    related = retrieve_reviews(index, BOOK, reviews)

    assert len(related) == 2
    assert all(r in reviews for r in related), "Retrieved reviews must come from the original list"
    print("\nRetrieved reviews:", related)


def test_predict_rating():
    """
    Tests the generation (LLM call) part of the pipeline.

    We verify that:
      1. Claude returns a single digit string.
      2. The digit is between 1 and 5 (valid rating range).

    We do NOT assert the exact rating value because LLM outputs can vary
    between runs. What matters is that the output is in the right format
    and range. In practice the rating for The Beach tends to be 1 or 2,
    reflecting the user's dislike of backpacking/utopia stories.
    """
    index = index_reviews(reviews)
    related = retrieve_reviews(index, BOOK, reviews)
    rating = predict_rating(BOOK, related)

    print(f"\nPredicted rating: {rating}")
    assert rating.strip().isdigit(), f"Expected a single digit, got: {rating!r}"
    assert 1 <= int(rating.strip()) <= 5, f"Rating out of range: {rating}"