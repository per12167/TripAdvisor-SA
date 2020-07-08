from typing import List

from fastapi import FastAPI, Query

from src.models.predict import predict

app = FastAPI()

@app.get('/predict')
async def predict_review(sentences: List[str] = Query(..., description='Sentences to process')):
    predictions = predict(sentences)

    response = [
        {
            'id': idx + 1,
            'sentence': sentence,
            'prediction': sentiment['prediction'],
            'posistive_words': sentiment['positive'],
            'positive_probability': sentiment['positive_prob'],
            'negative_words': sentiment['negative'],
            'negative_probability': sentiment['negative_prob'],
            'unclassiffied_words': sentiment['unclassiffied']
        }
        for idx, (sentence, sentiment) in enumerate(zip(sentences, predictions))
    ]

    return response
