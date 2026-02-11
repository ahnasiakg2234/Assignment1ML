import pandas as pd 
# Created a dataframe based on model evaluation metrics
data = {
    "Model": ["KNN(k=10)", "Decision Tree(max_depth=5)", "Random Forest(min_samples_split=15)"],
    "Accuracy": [0.96, 0.95, 0.96 ],
    "Precision": [0.97, 0.96, 0.96],
    "Recall": [0.96, 0.96, 0.99],
    "F1-Score": [0.96, 0.96, 0.97],
}

df = pd.DataFrame(data)
# Display the dataframe
print(df)
