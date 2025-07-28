
## 🔧 **1. Installing and Importing Required Libraries**

```python
!pip install minisom
```

* Installs the `MiniSom` library, a lightweight implementation of the Self-Organizing Map.

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

* These are standard libraries for numerical computation, data visualization, and data handling.

---

## 📄 **2. Loading the Dataset**

```python
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values  # All features except the last column
y = dataset.iloc[:, -1].values   # Last column, usually a label (approved = 1 / not approved = 0)
```

* You load the dataset, split features into `X`, and the labels into `y`.

---

## 📊 **3. Feature Scaling**

```python
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
X = sc.fit_transform(X)
```

* SOMs are distance-based models and sensitive to scale, so **Min-Max scaling** is applied to scale all features between `0` and `1`.

---

## 🧠 **4. Training the Self-Organizing Map**

```python
from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)
```

* A SOM is initialized with a **10x10 grid** (i.e., 100 neurons).
* `input_len=15` because your dataset has 15 features.
* `sigma` is the neighborhood radius.
* The SOM is trained on `X` using 100 iterations.

---

## 📈 **5. Visualizing the SOM**

```python
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)  # U-Matrix showing distances between neurons
colorbar()
```

* `distance_map()` computes the **U-Matrix** which helps visualize how "similar" or "distant" each neuron is from its neighbors.
* Light/high values = **anomalies/outliers** (suspicious data)

Then this loop plots markers:

```python
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)  # Get winning neuron
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],     # 'o' or 's' depending on label
         markeredgecolor=colors[y[i]],  # Red for not approved, green for approved
         markerfacecolor='None',
         markersize=10,
         markeredgewidth=2)
show()
```

* Plots each application on the grid.
* Approved and rejected applications are shown in **different shapes and colors**.

---

## 🔍 **6. Finding the Frauds**

```python
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(8,1)], mappings[(6,9)]), axis = 0)
frauds = sc.inverse_transform(frauds)
```

* `win_map()` gives all data points mapped to each neuron.
* You **manually pick** neurons `(8,1)` and `(6,9)` — likely identified as outliers based on U-Matrix or visual inspection.
* These neurons are likely to represent **anomalous/fraudulent applications**.
* Then you reverse the scaling using `inverse_transform` to get the original values.

---

## 📃 **7. Output: Suspected Fraud Applications**

The printed output (`print(frauds)`) shows a list of credit card applications with original values that are flagged as suspicious.

Each row represents an applicant's details (e.g., ID, age, income, etc.), and these are the ones considered most **unusual** or **potentially fraudulent**.

---

## ✅ Summary

You created a **fraud detection model** using an **unsupervised learning technique (SOM)** to:

* Visualize customer groups
* Identify anomalies (potential fraud)
* Output details of those suspicious applicants


