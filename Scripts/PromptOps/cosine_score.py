from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def cosine_score(text1, text2):
    """
    Calculate the cosine similarity score between two texts using SentenceTransformer.

    Parameters:
    text1 (str): The first text to compare.
    text2 (str): The second text to compare.

    Returns:
    float: The cosine similarity score between the two texts.
    """
    # Load the pre-trained SentenceTransformer model
    model = SentenceTransformer('all-mpnet-base-v2')

    # Encode the texts into embeddings
    embeddings1 = model.encode([text1])
    embeddings2 = model.encode([text2])

    # Calculate the cosine similarity between the embeddings
    score = cosine_similarity(embeddings1, embeddings2)

    # Return the similarity score
    return score[0][0]
