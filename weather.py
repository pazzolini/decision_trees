import pandas as pd
from ID3implementation import ID3


def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data = data.drop(data.columns[0], axis=1)

    data['Temp'] = pd.cut(data['Temp'], bins=[-float('inf'), 70, 79, float('inf')],
                                 labels=["<=70", "71-79", ">=80"])
    data['Humidity'] = pd.cut(data['Humidity'], bins=[-float('inf'), 75, 89, float('inf')],
                                labels=["<=75", "76-89", ">=90"])
    return data


def classify_new_examples(classifier, file_path):
    new_data = load_and_preprocess_data(file_path)
    new_data['Predicted Class'] = new_data.apply(lambda x: classifier.classify(x, classifier.tree), axis=1)

    return new_data


if __name__ == "__main__":
    train_file_path = 'DS/weather.csv'
    test_file_path = 'DS/test2.csv'

    training_data = load_and_preprocess_data(train_file_path)

    print("Decision Tree for the Weather Dataset:\n")
    id3_instance = ID3(target_attribute_name=training_data.columns[-1])
    id3_instance.build_and_print_tree(training_data)

    classified_data = classify_new_examples(id3_instance, test_file_path)
    print("\nTesting Data Classified:\n")
    print(classified_data)
