import naive_bayes_classifier as nbc

features, labels = [], []
with open('golf.csv', 'r') as f:
    data = map(lambda x: x.split(','), f.read().strip().split('\n')[1:])
    for line in data:
        features.append(line[:-1])
        labels.append(line[-1])
print(features)
print(labels)
print()

classifier = nbc.NaiveBayesClassifier()
classifier.fit(features, labels)
classifier.print_probabilities()
res = classifier.predict(['sunny', 'cool', 'high', 'true'])
print(res)