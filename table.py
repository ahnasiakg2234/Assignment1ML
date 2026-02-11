
import pandas as pd 

data = {
    "Model": ["KNN", "Decision Tree", "Random Forest"],
    "Accuracy": [0.95, 0.94, 0.96 ],
    "Precision": [0.96, 0.96, 0.96],
    "Recall": [0.96, 0.94, 0.99],
    "F1-Score": [0.96, 0.95, 0.97],
}

df = pd.DataFrame(data)
df = df.sort_values(by="F1-Score", ascending=False)
print(df)

df2 = df.sort_values(by="Recall", ascending=False)
print(df2)