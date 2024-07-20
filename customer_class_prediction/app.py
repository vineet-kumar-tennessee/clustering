<<<<<<< HEAD
from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import io
import base64

app = Flask(__name__)

# Load the saved KMeans model
with open('kmeans_model.pkl', 'rb') as file:
    kmeans = pickle.load(file)

# Load the dataset for plotting clusters
d = pd.read_csv(r"C:\Users\khull\Desktop\N_IT\july\july12\Mall_Customers.csv")
x = d.iloc[:, [3, 4]].values

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    income = float(request.form['income'])
    score = float(request.form['score'])
    
    # Prepare data for prediction
    data = [[income, score]]
    
    # Predict the cluster
    cluster = kmeans.predict(data)[0]

    # Plot the clusters
    plt.figure(figsize=(10, 6))
    plt.scatter(x[kmeans.labels_ == 0, 0], x[kmeans.labels_ == 0, 1], s=100, c='red', label='Cluster 1')
    plt.scatter(x[kmeans.labels_ == 1, 0], x[kmeans.labels_ == 1, 1], s=100, c='blue', label='Cluster 2')
    plt.scatter(x[kmeans.labels_ == 2, 0], x[kmeans.labels_ == 2, 1], s=100, c='green', label='Cluster 3')
    plt.scatter(x[kmeans.labels_ == 3, 0], x[kmeans.labels_ == 3, 1], s=100, c='cyan', label='Cluster 4')
    plt.scatter(x[kmeans.labels_ == 4, 0], x[kmeans.labels_ == 4, 1], s=100, c='magenta', label='Cluster 5')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
    plt.scatter(income, score, s=300, c='black', label='New Data Point', marker='X')
    plt.title('Clusters of customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()

    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode()
    plt.close()

    return render_template('result.html', cluster=cluster, plot_url=plot_data)

if __name__ == '__main__':
    app.run(debug=True)
=======

>>>>>>> 5995d1dbbcf43833366e7772bd254cbcb92249b4
