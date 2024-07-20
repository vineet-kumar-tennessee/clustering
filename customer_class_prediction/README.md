# Customer Class Prediction Project

This project aims to segment customers into different classes based on their annual income and spending score using K-Means clustering. The goal is to identify distinct customer groups to better understand their behavior and preferences.

## Demo Video

[Watch the demo video](https://youtu.be/liUShAItfoQ?si=uRuTcsF9mr_r8rH3)

## Setup and Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/customer_class_prediction.git
    cd customer_class_prediction
    ```

2. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the `define_cluster.py` script to train and save the clustering model:**

    ```bash
    python define_cluster.py
    ```

4. **Deploy the Flask application:**

    ```bash
    python application_deploy.py
    ```

## Usage

- Navigate to `http://127.0.0.1:5000/` in your web browser.
- Upload the customer dataset to predict the cluster each customer belongs to.
- View the clustering results on the results page.

## Front End User Input Interface

![Front End User Input Interface](https://github.com/yourusername/customer_class_prediction/blob/main/front_end_images/front_end_input.png)

This is how the front end user input interface looks. The model will predict customer clusters based on these inputs.

## Code Explanation

### `define_cluster.py`

This script loads the customer data, applies K-Means clustering, and saves the trained model using `pickle`:

```python
import pandas as pd
from sklearn.cluster import KMeans
import pickle

d = pd.read_csv(r"data location\Mall_Customers.csv")
x = d.iloc[:, [3, 4]].values

k = KMeans(n_clusters=5, init='k-means++', random_state=0)
k.fit(x)

with open('kmeans_model.pkl', 'wb') as file:
    pickle.dump(k, file)
