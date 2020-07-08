import pickle
from typing import List

from src.features.tokenize import tokenize_classes


def predict(documents: List[str]):
    document_classes = {
        'UNK': documents
    }

    word_classes = tokenize_classes(document_classes)

    with open('models/model.pkl', 'rb') as input_file:
        model = pickle.load(input_file)

    document_words = word_classes['UNK']

    predictions = []
    for document in document_words:
        positive = model['POS_PROB']
        negative = model['NEG_PROB']
        pos_words = 0
        neg_words = 0
        uncls_words = 0

        for word in document:
            try:
                positive += model['COND_POS_PROBS'][word]['logprob']
                pos_words += 1
            except KeyError:
                positive += model['COND_POS_PROBS'][-1]['logprob']
                uncls_words += 1

            try:
                negative += model['COND_NEG_PROBS'][word]['logprob']
                neg_words += 1
            except KeyError:
                negative += model['COND_NEG_PROBS'][-1]['logprob']
                uncls_words += 1

        if positive >= negative:
            predictions.append(
                {'prediction': 'POS',
                 'positive': pos_words,
                 'positive_prob': positive,
                 'negative': neg_words,
                 'negative_prob': negative,
                 'unclassiffied': uncls_words})
        else:
            predictions.append(
                {'prediction': 'NEG',
                 'positive': pos_words,
                 'positive_prob': positive,
                 'negative': neg_words,
                 'negative_prob': negative,
                 'unclassiffied': uncls_words})

    return predictions
