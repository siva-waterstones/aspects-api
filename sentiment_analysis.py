from transformers import pipeline

def perform_sentiment_analysis(text):
    sentiment_analysis = pipeline("sentiment-analysis")
    result = sentiment_analysis(text)

    if isinstance(result, list) and len(result) > 0 and 'label' in result[0] and 'score' in result[0]:
        scores = {}
        for sentiment in result:
            label = sentiment['label'].lower()
            score = sentiment['score']
            scores[label] = score
        return scores
    else:
        return {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}

if __name__ == "__main__":
    input_text = "It is absolutely critical that the Government of Maldives distanced itself from the comments by the minister. I know that the government has suspended the ministers, but I think it is important that the Government of Maldives issue a formal apology to the Indian people"
    sentiment_scores = perform_sentiment_analysis(input_text)

    print(f"Sentiment Scores for '{input_text}':")
    for label, score in sentiment_scores.items():
        print(f"{label.capitalize()}: {score:.4f}")

