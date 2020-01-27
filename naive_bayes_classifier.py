import math

'''
Features and classes must be categorical
Every possible feature value and class label must
be included at least once in the training data
'''
class NaiveBayesClassifier:
    def __init__(self):
        self.all_classes = None
        self.class_priors = None
        self.all_attribute_values = None
        self.attribute_likelihoods = None

    def fit(self, x, y):
        self.all_classes = set(y)
        self.class_priors = dict.fromkeys(self.all_classes, 0)
        for label in y:
            self.class_priors[label] += 1

        self.all_attribute_values = set()
        for data in x:
            for i in range(len(data)):
                self.all_attribute_values.add((i, data[i]))
        self.attribute_likelihoods = {label: dict.fromkeys(self.all_attribute_values, 0) for label in self.class_priors}
        for data, label in zip(x, y):
            for i in range(len(data)):
                key = (i, data[i])
                self.attribute_likelihoods[label][key] += 1/self.class_priors[label]
        
        for label in self.class_priors:
            self.class_priors[label] /= len(y)

    def print_probabilities(self):
        print('Class priors:')
        for label, prob in self.class_priors.items():
            print(label, prob)
        print()
        print('Attribute likelihoods:')
        for label, atts in self.attribute_likelihoods.items():
            print('Label:', label)
            for feature, prob in atts.items():
                print(feature, prob)
            print()

    def predict(self, x):
        class_probabilities = dict.fromkeys(self.class_priors, 1)
        for label, prob in self.class_priors.items():
            for i in range(len(x)):
                class_probabilities[label] *= self.attribute_likelihoods[label][(i, x[i])]
            class_probabilities[label] *= prob
        print(class_probabilities)
        max_label, max_prob = None, -math.inf
        for label, prob in class_probabilities.items():
            if prob > max_prob:
                max_label, max_prob = label, prob
        return max_label