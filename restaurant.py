import pandas as pd
from ID3implementation import ID3


def load_and_preprocess_data(file_path):
    """
    Load and preprocess the dataset from a CSV file.
    """
    # Load the CSV file, specifying not to treat 'None' as NA
    data = pd.read_csv(file_path, na_values=['', 'NA'], keep_default_na=False)
    data = data.drop(data.columns[0], axis=1)

    return data


def classify_new_examples(classifier, file_path):
    """
    Classify new examples using the provided classifier and dataset file path.
    """
    new_data = load_and_preprocess_data(file_path)
    new_data['Predicted Class'] = new_data.apply(lambda x: classifier.classify(x, classifier.tree), axis=1)

    return new_data


if __name__ == "__main__":
    train_file_path = 'DS/restaurant.csv'
    test_file_path = 'DS/test1.csv'

    training_data = load_and_preprocess_data(train_file_path)

    print("Decision Tree for the restaurant dataset:\n")
    id3_instance = ID3(target_attribute_name=training_data.columns[-1])
    id3_instance.build_and_print_tree(training_data)

    classified_data = classify_new_examples(id3_instance, test_file_path)
    print("\nTesting Data:\n")
    print(classified_data)
