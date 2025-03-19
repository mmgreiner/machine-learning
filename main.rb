require 'rumale'
require 'csv'
require 'open-uri'

# Load the Iris dataset from the UCI repository (CSV format)
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
lines = URI.open(url) { f| f.read }
data = CSV.parse(lines, headers: false)
data = data.filter(&:any?)
data.shuffle!

# Split data into features and target
X = data.map { |row| row[0..3].map(&:to_f) }  # Features (sepal length, width, petal length, width)
y = data.map { |row| row[4] }                 # Target (species)

# Convert target classes into numeric labels (0, 1, 2)
target_labels = y.uniq
y = y.map { |label| target_labels.index(label) }

# Split the data into training and testing sets (80% training, 20% testing)
train_size = (X.size * 0.8).to_i
X_train, X_test = X[0...train_size], X[train_size..-1]
y_train, y_test = y[0...train_size], y[train_size..-1]

# Create and train the Support Vector Machine (SVM) classifier
svm = Rumale::LinearModel::SVC.new
svm.fit(X_train, y_train)

# Predict using the trained model on the test set
y_pred = svm.predict(X_test)

# Calculate the accuracy of the model
accuracy = Numo::NArray.column_stack([y_pred, y_test]).count { |pred, actual| pred == actual }.to_f / y_test.size

puts "Accuracy: #{accuracy * 100}%"
