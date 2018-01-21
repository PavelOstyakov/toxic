from toxic.model import get_model
from toxic.nltk_utils import tokenize_sentences
from toxic.train_utils import train_folds
from toxic.embedding_utils import read_embedding_list, clear_embedding_list, convert_tokens_to_ids

import argparse
import numpy as np
import os
import pandas as pd


UNKNOWN_WORD = "_UNK_"
END_WORD = "_END_"
NAN_WORD = "_NAN_"

CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def main():
    parser = argparse.ArgumentParser(
        description="Recurrent neural network for identifying and classifying toxic online comments")

    parser.add_argument("train-file-path")
    parser.add_argument("test-file-path")
    parser.add_argument("embedding-path")
    parser.add_argument("--result-path", default="toxic_results")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sentences-length", type=int, default=500)
    parser.add_argument("--recurrent-units", type=int, default=64)
    parser.add_argument("--dropout-rate", type=float, default=0.3)
    parser.add_argument("--dense-size", type=int, default=32)

    args = parser.parse_args()

    train_data = pd.read_csv(args.train_file_path)
    test_data = pd.read_csv(args.test_file_path)

    list_sentences_train = train_data["comment_text"].fillna(NAN_WORD).values
    list_sentences_test = test_data["comment_text"].fillna(NAN_WORD).values
    y_train = train_data[CLASSES].values

    tokenized_sentences_train, words_dict = tokenize_sentences(list_sentences_train, {})
    tokenized_sentences_test, words_dict = tokenize_sentences(list_sentences_test, words_dict)

    words_dict[UNKNOWN_WORD] = len(words_dict)

    embedding_list, embedding_word_dict = read_embedding_list(args.embedding_path)
    embedding_size = len(embedding_list[0])

    embedding_list, embedding_word_dict = clear_embedding_list(embedding_list, embedding_word_dict, words_dict)

    embedding_word_dict[UNKNOWN_WORD] = len(embedding_word_dict)
    embedding_list.append([0.] * embedding_size)
    embedding_word_dict[END_WORD] = len(embedding_word_dict)
    embedding_list.append([-1.] * embedding_size)

    embedding_matrix = np.array(embedding_list)

    id_to_word = dict((id, word) for word, id in words_dict.items())
    train_list_of_token_ids = convert_tokens_to_ids(
        tokenized_sentences_train,
        id_to_word,
        embedding_word_dict,
        args.sentences_length)
    test_list_of_token_ids = convert_tokens_to_ids(
        tokenized_sentences_test,
        id_to_word,
        embedding_word_dict,
        args.sentences_length)
    X_train = np.array(train_list_of_token_ids)
    X_test = np.array(test_list_of_token_ids)

    get_model_func = lambda: get_model(
        embedding_matrix,
        args.sentences_length,
        args.dropout_rate,
        args.recurrent_units,
        args.dense_size)

    models = train_folds(X_train, y_train, args.fold_count, args.batch_size, get_model_func)

    test_predicts_list = []
    for fold_id, model in enumerate(models):
        model_path = os.path.join(args.result_path, "model{0}_weights.npy".format(fold_id))
        np.save(model_path, model.get_weights())

        test_predicts_path = os.path.join(args.result_path, "test_predicts{0}.npy".format(fold_id))
        test_predicts = model.predict(X_test, batch_size=args.batch_size)
        test_predicts_list.append(test_predicts)
        np.save(test_predicts_path, test_predicts)

    test_predicts = np.multiply(*test_predicts_list)
    test_predicts **= (1. / len(test_predicts_list))

    test_ids = test_data["id"].values
    test_ids = test_ids.reshape((len(test_ids), 1))

    test_predicts = pd.DataFrame(data=test_predicts, columns=CLASSES)
    test_predicts["id"] = test_ids
    test_predicts = test_predicts[["id"] + CLASSES]
    submit_path = os.path.join(args.result_path, "submit")
    test_predicts.to_csv(submit_path, index=False)

if __name__ == "__main__":
    main()
