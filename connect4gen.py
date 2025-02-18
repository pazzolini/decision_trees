import numpy as np
import pandas as pd


class ID3:
    def __init__(self, target_attribute_name='class'):
        """
        Initialize the ID3 classifier with the name of the target attribute and an initially empty decision tree.
        """
        self.target_attribute_name = target_attribute_name  # Name of the attribute to predict
        self.tree = None  # Initially, the decision tree is empty

    def entropy(self, target_col):
        """
        Calculate the entropy of a dataset with respect to the target column.
        Entropy is a measure of the unpredictability or the randomness of the data.
        """
        # Get unique values and their frequency counts
        _, counts = np.unique(target_col, return_counts=True)
        total = counts.sum()  # Total count of elements in the target column
        # Calculate entropy using the formula: -sum(p * log2(p))
        return np.sum([(-count / total) * np.log2(count / total) for count in counts])

    def information_gain(self, data, split_attribute_name):
        """
        Calculate the information gain of a potential split,
        based on the decrease in entropy after the dataset is split on an attribute.
        """
        # Calculate the total entropy of the dataset before any split
        total_entropy = self.entropy(data[self.target_attribute_name])
        # Get the unique values of the attribute and their corresponding counts
        vals, counts = np.unique(data[split_attribute_name], return_counts=True)
        total_counts = counts.sum()  # Total occurrences of all attributes
        # Calculate the weighted entropy after the split
        weighted_entropy = np.sum([
            (counts[i] / total_counts) * self.entropy(data[data[split_attribute_name] == vals[i]][self.target_attribute_name])
            for i in range(len(vals))
        ])
        # Information gain is the difference in entropy before and after the split
        return total_entropy - weighted_entropy

    def id3(self, data, original_data, features, parent_node_class=None):
        """
        Recursively build the decision tree based on the ID3 algorithm using entropy and information gain
        as the criteria for selecting the attribute that best separates the dataset.
        """
        # Check if all examples have the same class. If so, return this class as a leaf node.
        if len(np.unique(data[self.target_attribute_name])) <= 1:
            return np.unique(data[self.target_attribute_name])[0], len(data)

        # If no data is available, or no features are left for splits, return the majority class of the parent node.
        elif len(data) == 0 or len(features) == 0:
            return parent_node_class, len(data)

        # Otherwise, proceed with finding the best feature to split on.
        else:
            # Determine the class that appears most frequently in the current node.
            parent_node_class = np.unique(data[self.target_attribute_name])[
                np.argmax(np.unique(data[self.target_attribute_name], return_counts=True)[1])]

            # Compute the information gain for each feature and find the one with the maximum gain.
            item_values = [self.information_gain(data, feature) for feature in features]
            best_feature_index = np.argmax(item_values)
            best_feature = features[best_feature_index]

            # Create a new decision node with the best feature and an empty dictionary for its branches.
            tree = {best_feature: {}}
            # Remove the best feature from the list of available features for subsequent splits.
            remaining_features = [i for i in features if i != best_feature]

            # Iterate over all the unique values of the best feature and create subtrees for each.
            for value in np.unique(data[best_feature]):
                # Subset the data that has the current value of the best feature.
                sub_data = data[data[best_feature] == value]
                # Recursively call id3 to build the subtree under the current branch.
                subtree = self.id3(sub_data, original_data, remaining_features, parent_node_class)
                # Assign the subtree to the corresponding branch of the decision node.
                tree[best_feature][value] = subtree

            # Return the tree node, which may have multiple branches leading to further decision nodes or leaf nodes.
            return tree

    def print_tree(self, tree, indent="", last=True):
        """
        Print the decision tree in a structured format to visualize its branches and decisions.
        """
        # Choose the appropriate prefix based on whether the node is the last in its set of siblings or not.
        prefix = "└── " if last else "├── "
        # If the current tree node is a dictionary, it has children and thus represents a decision node.
        children = list(tree.items()) if isinstance(tree, dict) else []

        if not children:
            # If the node has no children, it is a leaf node. Display its decision and the sample count.
            if isinstance(tree, tuple):
                decision, count = tree
                print(f"{indent}{prefix}[Decision] {decision} (Count: {count})")
        else:
            # If the node is a decision node, display the attribute it splits on.
            attribute, branches = children[0]
            print(f"{indent}{prefix}[Attribute] {attribute}")
            # Recursively print each branch of the decision node.
            for i, (value, subtree) in enumerate(sorted(branches.items()), 1):
                # Determine if this is the last child for correct branching character.
                last_child = i == len(branches)
                subtree_indent = indent + ('    ' if last_child else '│   ')
                print(f"{subtree_indent}[Value] {value}:")
                self.print_tree(subtree, subtree_indent, last_child)

    def build_and_print_tree(self, data):
        """
        Build the decision tree from the provided dataset and then print it using the previous function.
        """
        # Extract features from the dataset excluding the target attribute column.
        features = data.columns[:-1].tolist()
        # Build the decision tree using the ID3 algorithm.
        self.tree = self.id3(data, data, features)
        # Print the tree starting from the root.
        self.print_tree(self.tree)

    def classify(self, example, tree):
        """
        Recursively classify an example based on the decision tree.
        """
        if not isinstance(tree, dict):
            # If the tree is not a dictionary, it should be a tuple representing a leaf node.
            if isinstance(tree, tuple):
                return tree[0]  # The class label
            return tree

        # Otherwise, get the root attribute and its corresponding subtree.
        attribute = next(iter(tree))
        value = example[attribute]
        subtree = tree[attribute].get(value, tree[attribute].get(None, None))  # Use a default case if value not found

        if subtree is None:
            return None  # Return None if no subtree and no default case

        # Recursively classify using the subtree.
        return self.classify(example, subtree)
        
    def save_tree(self, filename):
        def tree_to_str(tree, indent="", last=True):
            prefix = "└── " if last else "├── "
            children = list(tree.items()) if isinstance(tree, dict) else []

            if not children:
                if isinstance(tree, tuple):
                    decision, count = tree
                    return f"{indent}{prefix}[Decision] {decision} (Count: {count})\n"
                return f"{indent}{prefix}[{tree}]\n"
            else:
                attribute, branches = children[0]
                result = f"{indent}{prefix}[Attribute] {attribute}\n"
                for i, (value, subtree) in enumerate(sorted(branches.items()), 1):
                    last_child = i == len(branches)
                    subtree_indent = indent + ('    ' if last_child else '│   ')
                    result += f"{subtree_indent}[Value] {value}:\n"
                    result += tree_to_str(subtree, subtree_indent, last_child)
                return result

        tree_str = tree_to_str(self.tree)
        with open(filename, 'w') as f:
            f.write(tree_str)


def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data = data.drop(data.columns[0], axis=1)
    return data


def classify_new_examples(classifier, file_path):
    new_data = load_and_preprocess_data(file_path)
    new_data['Predicted Class'] = new_data.apply(lambda x: classifier.classify(x, classifier.tree), axis=1)
    return new_data


if __name__ == "__main__":
    train_file_path = 'DS/connect4_edited.csv'
    test_file_path = 'DS/test4.csv'
    tree_file_path = 'decision_tree.txt'

    training_data = load_and_preprocess_data(train_file_path)

    print("Decision Tree for the Connect4 Dataset:\n")
    id3_instance = ID3(target_attribute_name=training_data.columns[-1])
    id3_instance.build_and_print_tree(training_data)
    id3_instance.save_tree(tree_file_path)

    classified_data = classify_new_examples(id3_instance, test_file_path)
    print("\nTesting Data Classified:\n")
    print(classified_data)

    # Save the classified data to a CSV file
    output_file_path = 'DS/classified_test4_results.csv'
    classified_data.to_csv(output_file_path, index=False)
    print(f"Classified data saved to {output_file_path}")
