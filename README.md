# SVM Classification of Cell Samples

This repository contains code for classifying cell samples using Support Vector Machine (SVM) with Scikit-learn. The dataset used includes various features of cell samples, and the project involves preprocessing the data, training the SVM model, and evaluating its performance using metrics such as the confusion matrix, F1 score, and Jaccard index.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Feature Selection](#feature-selection)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Installation

To run the code in this repository, you will need to have Python installed along with the following libraries:

```bash
pip install scikit-learn==0.23.1
pip install pandas
pip install matplotlib
```

## Dataset

The dataset used in this project is a collection of cell samples that includes features such as `Clump Thickness`, `Uniformity of Cell Size`, `Uniformity of Cell Shape`, `Marginal Adhesion`, `Single Epithelial Cell Size`, `Bare Nuclei`, `Bland Chromatin`, `Normal Nucleoli`, and `Mitoses`. The target variable (`Class`) indicates whether the cells are benign (2) or malignant (4).

### Downloading the Dataset

The dataset can be downloaded using the following code:

```python
import requests

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/cell_samples.csv'
response = requests.get(url)

with open('cell_samples.csv', 'wb') as file:
    file.write(response.content)
```

## Feature Selection

The following features are selected for training the model:

- `Clump Thickness`
- `Uniformity of Cell Size`
- `Uniformity of Cell Shape`
- `Marginal Adhesion`
- `Single Epithelial Cell Size`
- `Bare Nuclei`
- `Bland Chromatin`
- `Normal Nucleoli`
- `Mitoses`

Data preprocessing includes handling missing values in the `Bare Nuclei` column.

## Modeling

The model is built using the Support Vector Machine (SVM) algorithm with an RBF kernel:

```python
from sklearn import svm

clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
```

## Evaluation

The model is evaluated using the following metrics:

- **Confusion Matrix**: Displays the true positive, false positive, true negative, and false negative counts.
- **F1 Score**: The weighted average of precision and recall.
- **Jaccard Index**: A similarity measure that is used to compare the actual labels with the predicted labels.

The confusion matrix can be visualized using the `plot_confusion_matrix` function:

```python
from sklearn.metrics import confusion_matrix

cnf_matrix = confusion_matrix(y_test, yhat, labels=[2,4])

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'], normalize=False, title='Confusion matrix')
```

## Results

- **Jaccard Index**: The Jaccard index for the model is computed as follows:

```python
from sklearn.metrics import jaccard_score
print(jaccard_score(y_test, yhat, pos_label=2))
```

## Contributing

Contributions are welcome! If you have any improvements or suggestions, please feel free to create a pull request or raise an issue.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project is part of the IBM Developer Skills Network's machine learning course. Special thanks to the course creators for providing the dataset and the initial framework for this project.
