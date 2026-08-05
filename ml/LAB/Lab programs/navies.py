from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

X = [
    [4,128],
    [8,256],
    [16,512],
    [4,256],
    [8,512],
    [2,64]
]

y = ["No","Yes","Yes","No","Yes","No"]

model = GaussianNB()
model.fit(X, y)

pred = model.predict(X)

print("Prediction:", pred)
print("Confusion Matrix:\n", confusion_matrix(y, pred))
print("Accuracy:", accuracy_score(y, pred))