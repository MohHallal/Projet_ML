import pandas as pd
import matplotlib.pyplot as plt
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import numpy as np
from sklearn.metrics import precision_score, accuracy_score,recall_score, f1_score
from keras.models import Model
from keras.layers import Dense, Input

data = pd.read_csv('/home/tibome/Downloads/project_ML/TCGA-PANCAN-HiSeq-801x20531/data.csv', header=0, index_col=0)
labels = pd.read_csv('/home/tibome/Downloads/project_ML/TCGA-PANCAN-HiSeq-801x20531/labels.csv', header=0, index_col=0)


# Pre processing data

""" In the documentation of the file, it is precised that there in no missing values, now we need to check it"""
print("___Preprocessing___")
if data.isnull().any().any():
    print("There are missing values in the DataFrame.")
else:
    print("No missing values in the DataFrame.")

""" Now, as we are sure there are no missing values in the dataset, we need to eliminate the columns that have only 
zeroes which represents the genes that are not expressed in the data """

print("Shape of the dataframe before eliminating the non expressed genes: ", data.shape)
data = data.loc[:, (data != 0).any(axis=0)] # Drop columns with only zeroes
print("Shape of the dataframe after eliminating the non expressed genes: ", data.shape)

""" Now we proceed to eliminate the genes that have similair values between the samples using standard deviation """

data_sd = data.std()
plt.figure(figsize=(10, 6))
plt.hist(data_sd, bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Standard Deviations')
plt.xlabel('Standard Deviation')
plt.ylabel('Frequency')
plt.savefig('standard_deviation_distribution.png')
plt.show()

threshold = 1
selected_genes = data_sd[data_sd >= threshold].index # Filter out genes with standard deviation below the threshold
print("Shape of the dataframe before eliminating the genes that have low variations: ", data.shape)
data = data[selected_genes] # Create a new DataFrame with only the selected genes
print("Shape of the dataframe before eliminating the genes that have low variations: ", data.shape)

""" Now we proceed to do a PCA """

# Step 1: Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Step 2: Perform PCA
num_components = min(scaled_data.shape[0], scaled_data.shape[1])
pca = PCA(n_components=num_components)
principal_components = pca.fit_transform(scaled_data)

# Choose the number of components
num_selected_components = 2
selected_principal_components = principal_components[:, :num_selected_components]

# Create a new DataFrame with the selected principal components
pca_df = pd.DataFrame(
    data=selected_principal_components,
    columns=[f'PC{i}' for i in range(1, num_selected_components + 1)],
    index=data.index
)

# Merge with labels based on row index
pca_df_with_metadata = pd.merge(pca_df, labels, left_index=True, right_index=True)

# Convert 'Class' column to numeric values
class_mapping = {label: i for i, label in enumerate(pca_df_with_metadata['Class'].unique())}
pca_df_with_metadata['Class_numeric'] = pca_df_with_metadata['Class'].map(class_mapping)

# Plot the first two principal components with colors based on metadata
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    pca_df_with_metadata['PC1'],
    pca_df_with_metadata['PC2'],
    c=pca_df_with_metadata['Class_numeric'],  # Use the numeric column instead
    cmap='viridis'
)
plt.title('PCA: First Two Principal Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Create a legend without relying on scatter.legend_elements()
legend_labels = pca_df_with_metadata['Class'].unique()
legend_handles = [plt.Line2D([0], [0], marker='o', color=plt.cm.viridis(class_mapping[label]), label=label)
                  for label in legend_labels]
plt.legend(handles=legend_handles, title='Class', loc='best')
plt.savefig('PCA.png')
plt.show()
""" Changing the values of labels, making it 1 when there is 'BRCA' and 0 when it is not """
labels['Class'] = labels['Class'].apply(lambda x: 1 if x == 'BRCA' else 0)
""" Splitting the data"""
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, random_state=42)
""" Train the model (logistic regression) """
logreg = linear_model.LogisticRegression()
logreg.fit(X_train, y_train)
y_predicted_logistic = logreg.predict(X_train)
y_train = np.squeeze(y_train)
""" Evaluation of the model of logistic regression on training data """

print("___Evalutaion of the model (logistic regression) on the training set___")
print(pd.crosstab(y_train,y_predicted_logistic))
precision_log_reg_train = precision_score(y_train, y_predicted_logistic)
accuracy_log_reg_train = accuracy_score(y_train, y_predicted_logistic)
recall_log_reg_train = recall_score(y_train, y_predicted_logistic)
f1_log_reg_train = f1_score(y_train, y_predicted_logistic)
print("Precision: ", precision_log_reg_train)
print("Accuaracy: ", accuracy_log_reg_train)
print("Recall: ", recall_log_reg_train)
print("F score", f1_log_reg_train)

""" Evaluation of the model of logistic regression on test data """

print("___Evalutaion of the model (logistic regression) on the test set___")
y_predicted_logistic_test = logreg.predict(X_test)
y_test = np.squeeze(y_test)
print(pd.crosstab(y_test,y_predicted_logistic_test))
precision_log_reg_test = precision_score(y_test, y_predicted_logistic_test)
accuracy_log_reg_test = accuracy_score(y_test, y_predicted_logistic_test)
recall_log_reg_test = recall_score(y_test, y_predicted_logistic_test)
f1_log_reg_test = f1_score(y_test, y_predicted_logistic_test)
print("Precision: ", precision_log_reg_test)
print("Accuaracy: ", accuracy_log_reg_test)
print("Recall: ", recall_log_reg_test)
print("F score", f1_log_reg_test)

""" Learning curve of the model"""
examples = []
train_accuracies = []
test_accuracies = []
for i in range(10, X_train.shape[0],int(X_train.shape[0]/10)):
    logreg.fit(X_train.iloc[:i], y_train.iloc[:i])
    # Calculate training accuracy
    training_accuracy = accuracy_score(logreg.predict(X_train), y_train)
    # Calculate testing accuracy
    testing_accuracy = accuracy_score(logreg.predict(X_test), y_test)
    # Append accuracies to the lists
    train_accuracies.append(training_accuracy)
    test_accuracies.append(testing_accuracy)
    examples.append(i)
learning_curve_df = pd.DataFrame({
    'Training Accuracy': train_accuracies,
    'Testing Accuracy': test_accuracies,
    'Number of Training Examples': examples
})
# Plot the learning curve
plt.figure(figsize=(10, 6))
plt.plot(learning_curve_df['Number of Training Examples'], learning_curve_df['Training Accuracy'], label='Training Accuracy')
plt.plot(learning_curve_df['Number of Training Examples'], learning_curve_df['Testing Accuracy'], label='Testing Accuracy')
plt.title('Learning Curve of the logitic regression model')
plt.xlabel('Number of Training Examples')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('learning_curve_logistic_regression.png')
plt.show()

"""Train the model (Tree based model)"""
dmat = xgboost.DMatrix(data=data, label=labels)
dtrain = xgboost.DMatrix(data=X_train, label=y_train)
params = {'objective': 'binary:logistic', 'silent': True}
#training the model
bst = xgboost.train(params = params, dtrain = dtrain)
preds = bst.predict(data = dtrain)
# Convert predicted probabilities to binary predictions using a threshold
threshold = 0.5
preds = (preds > threshold).astype(int)
preds = pd.Series(preds, index=y_train.index)

""" Evaluation of the tree based model on training data """

print("___Evalutaion of the model (tree based) on the training set___")
print(pd.crosstab(y_train,preds))
precision_tree_train = precision_score(y_train, preds)
accuracy_tree_train = accuracy_score(y_train, preds)
recall_tree_train = recall_score(y_train, preds)
f1_tree_train = f1_score(y_train, preds)
print("Precision: ", precision_tree_train)
print("Accuaracy: ", accuracy_tree_train)
print("Recall: ", recall_tree_train)
print("F score", f1_tree_train)

""" Evaluation of the tree based model on test data """

print("___Evalutaion of the model (tree based) on the test set___")
dtest = xgboost.DMatrix(data=X_test, label=y_test)
y_tree_test = bst.predict(dtest)
y_tree_test = (y_tree_test > threshold).astype(int)
y_tree_test = pd.Series(y_tree_test, index=y_test.index)
print(pd.crosstab(y_test,y_tree_test))
precision_tree_test = precision_score(y_test, y_tree_test)
accuracy_tree_test = accuracy_score(y_test, y_tree_test)
recall_tree_test = recall_score(y_test, y_tree_test)
f1_tree_test = f1_score(y_test, y_tree_test)
print("Precision: ", precision_tree_test)
print("Accuaracy: ", accuracy_tree_test)
print("Recall: ", recall_tree_test)
print("F score", f1_tree_test)

""" Learning curve of the model"""
examples = []
train_accuracies = []
test_accuracies = []
for i in range(10, X_train.shape[0],int(X_train.shape[0]/10)):
    dtrain_lear_curve = xgboost.DMatrix(data=X_train.iloc[:i], label=y_train.iloc[:i])
    # training the model
    bst_lear_curve = xgboost.train(params=params, dtrain=dtrain_lear_curve)
    preds_train = bst_lear_curve.predict(data=dtrain)
    preds_train = (preds_train > threshold).astype(int)
    preds_train = pd.Series(preds_train, index=y_train.index)
    training_accuracy = accuracy_score(preds_train, y_train)
    preds_test = bst_lear_curve.predict(dtest)
    preds_test = (preds_test > threshold).astype(int)
    preds_test = pd.Series(preds_test, index=y_test.index)
    testing_accuracy = accuracy_score(preds_test, y_test)
    train_accuracies.append(training_accuracy)
    test_accuracies.append(testing_accuracy)
    examples.append(i)
learning_curve_df = pd.DataFrame({
    'Training Accuracy': train_accuracies,
    'Testing Accuracy': test_accuracies,
    'Number of Training Examples': examples
})
# Plot the learning curve
plt.figure(figsize=(10, 6))
plt.plot(learning_curve_df['Number of Training Examples'], learning_curve_df['Training Accuracy'], label='Training Accuracy')
plt.plot(learning_curve_df['Number of Training Examples'], learning_curve_df['Testing Accuracy'], label='Testing Accuracy')
plt.title('Learning Curve of the tree based model')
plt.xlabel('Number of Training Examples')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('learning_curve_tree.png')
plt.show()

""" Neural networks """
init = 'random_uniform'

input_layer = Input(shape=(data.shape[1],))
x = input_layer

for i in range(1, 4):
    x = Dense(int(data.shape[1] / (2**i)), activation='relu', kernel_initializer=init)(x)

output_layer = Dense(1, activation='sigmoid', kernel_initializer=init)(x)
model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
# Fitting the model
model.fit(X_train, y_train, batch_size=8, epochs=8, verbose=1)
y_predicted_neural = model.predict(X_test)
y_predicted_neural = (y_predicted_neural > threshold).astype(int)
y_predicted_neural = pd.Series(y_predicted_neural.flatten(), index=y_test.index)
""" Evaluation of the neural network model on test data """

print("___Evalutaion of the model (neural network) on the test set___")
print(pd.crosstab(y_test,y_predicted_neural))
precision_neural_test = precision_score(y_test,y_predicted_neural)
accuracy_neural_test = accuracy_score(y_test,y_predicted_neural)
recall_neural_test = recall_score(y_test,y_predicted_neural)
f1_neural_test = f1_score(y_test,y_predicted_neural)
print("Precision: ", precision_neural_test)
print("Accuaracy: ", accuracy_neural_test)
print("Recall: ", recall_neural_test)
print("F score", f1_neural_test)

""" Evaluation of the neural network model on training data """

print("___Evalutaion of the model (neural network) on the training set___")
y_predicted_neural_train = model.predict(X_train)
y_predicted_neural_train = (y_predicted_neural_train > threshold).astype(int)
y_predicted_neural_train = pd.Series(y_predicted_neural_train.flatten(), index=y_train.index)
print(pd.crosstab(y_train,y_predicted_neural_train))
precision_neural_train = precision_score(y_train,y_predicted_neural_train)
accuracy_neural_train = accuracy_score(y_train,y_predicted_neural_train)
recall_neural_train = recall_score(y_train,y_predicted_neural_train)
f1_neural_train = f1_score(y_train,y_predicted_neural_train)
print("Precision: ", precision_neural_train)
print("Accuaracy: ", accuracy_neural_train)
print("Recall: ", recall_neural_train)
print("F score", f1_neural_train)
