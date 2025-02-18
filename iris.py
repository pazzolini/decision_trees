import pandas as pd
from ID3implementation import ID3


def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data = data.drop(data.columns[0], axis=1)

    data['sepallength'] = pd.cut(data['sepallength'], bins=[-float('inf'), 5.3, 6.9, float('inf')],
                                 labels=["<=5.3", "5.4-6.9", ">6.9"])
    data['sepalwidth'] = pd.cut(data['sepalwidth'], bins=[-float('inf'), 2.9, 3.3, float('inf')],
                                labels=["<=2.9", "3.0-3.3", ">3.3"])
    data['petallength'] = pd.cut(data['petallength'], bins=[-float('inf'), 2.0, 4.9, float('inf')],
                                  labels=["<=2.0", "2.1-4.9", ">4.9"])
    data['petalwidth'] = pd.cut(data['petalwidth'], bins=[-float('inf'), 0.5, 1.6, float('inf')],
                                 labels=["<=0.5", "0.6-1.6", ">1.6"])
    return data


def classify_new_examples(classifier, file_path):
    new_data = load_and_preprocess_data(file_path)
    new_data['Predicted Class'] = new_data.apply(lambda x: classifier.classify(x, classifier.tree), axis=1)

    return new_data


if __name__ == "__main__":
    train_file_path = 'DS/iris.csv'
    test_file_path = 'DS/test3.csv'

    training_data = load_and_preprocess_data(train_file_path)

    print("Decision Tree for the Iris Dataset:\n")
    id3_instance = ID3(target_attribute_name=training_data.columns[-1])
    id3_instance.build_and_print_tree(training_data)

    classified_data = classify_new_examples(id3_instance, test_file_path)
    print("\nTesting Data Classified:\n")
    print(classified_data)

    # Save the classified data to a CSV file
    output_file_path = 'DS/classified_test3_results.csv'
    classified_data.to_csv(output_file_path, index=False)
    print(f"Classified data saved to {output_file_path}")
