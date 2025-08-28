import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("manual classification network.csv")

# reformat the data

# first define a new df (there's some funky stuff with the csv)
new_df = pd.DataFrame()
new_df["label"] = df["Label"]
new_df.dropna(inplace=True)

# replace the headers with the 1st column (to account for the 'umbrella' headers like color and shape)
new_colnames = df.iloc[0]
df = df[1:]
df.columns = new_colnames

# replace all nans with 0s
df.fillna(0, inplace=True)

# put the dfs together into one big df
new_df = pd.concat([new_df, df.drop(columns=[df.columns[0]])], axis=1)

# rename the columns for consistency
new_df.columns = [x.lower() for x in new_df.columns]

# standardize the shape numbers using min-max scaling
scaler = MinMaxScaler()
new_df[["length", "width", "height"]] = scaler.fit_transform(new_df[["length", "width", "height"]])

# drop unnecessary columns
new_df.drop(["dull", "rough"], axis=1, inplace=True)
new_df = new_df.rename(columns={"shiny": "is_shiny", "smooth": "is_smooth"})

print(new_df)


# separate the data into labels and features
y = new_df["label"]
X = new_df.drop(["label"], axis=1)

# split the data (30% test 70% train)
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.3, 
                                                    random_state=42)

# fit model
dtc = DecisionTreeClassifier(random_state=42)

dtc.fit(X_train, y_train)

# do a prediction
y_pred = dtc.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# visualize tree
plt.figure(figsize=(8, 8))
plot_tree(dtc, feature_names=X.columns, class_names=y.unique())
plt.show()


