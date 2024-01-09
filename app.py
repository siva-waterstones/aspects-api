from flask import Flask, request, jsonify
from transformers import pipeline
import urllib.parse

app = Flask(__name__)

# Load the sentiment analysis pipeline
sentiment_analysis_pipeline = pipeline("sentiment-analysis")

@app.route('/analyze-sentiment', methods=['GET'])
def analyze_sentiment():
    try:
        # Extract the textOriginal parameter from the URL
        text_original = request.args.get('textOriginal', '')

        # URL encode the textOriginal parameter
        encoded_text = urllib.parse.quote(text_original)

        # Perform sentiment analysis using Hugging Face Transformers
        sentiment_result = sentiment_analysis_pipeline(text_original)

        # Example response format: {'sentiment': 'POSITIVE', 'score': 0.99}
        response = {
            'sentiment': sentiment_result[0]['label'],
            'score': sentiment_result[0]['score']
        }

        return jsonify(response)

    except Exception as e:
        error_message = {'error': str(e)}
        return jsonify(error_message), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
