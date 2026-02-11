
import pandas as pd 
# Created a dataframe based on model evaluation metrics
data = {
    "Model": ["KNN", "Decision Tree", "Random Forest"],
    "Accuracy": [0.95, 0.94, 0.96 ],
    "Precision": [0.96, 0.96, 0.96],
    "Recall": [0.96, 0.94, 0.99],
    "F1-Score": [0.96, 0.95, 0.97],
}

#Comparing values to choose "best" model
df = pd.DataFrame(data)
df = df.sort_values(by="F1-Score", ascending=False)
print(df)
#Sorted by F1 score to evaluate overall balance
df2 = df.sort_values(by="Recall", ascending=False)
print(df2)
#Sorted by recall to prioritize minimizing false negatives

#In either case the Random forest model appears to be the best choice.
#All models are equally precise, and the random forest model was also the most accurate. 