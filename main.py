import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("Iris.csv")


X = df.drop(["Species", "Id"], axis=1)
y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

def get_flower_data():
    sepal_length = float(input("enter sepal length (cm): "))
    sepal_width = float(input("enter sepal width (cm): "))
    petal_length = float(input("enter petal length (cm): "))
    petal_width = float(input("enter petal width (cm): "))
    return [[sepal_length, sepal_width, petal_length, petal_width]]
flower = pd.DataFrame(
    get_flower_data(),
    columns=X.columns
)
prediction = model.predict(flower)
print(f"Predicted Species: {prediction[0]}")

probabilities = model.predict_proba(flower)
for species, probability in zip(model.classes_, probabilities[0]):
    print(f"{species}: {probability:.2%}")





