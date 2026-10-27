# DimViz — PCA & t-SNE Dimensionality Reduction Visualizer

**Hack-o-week Week 11–12 • 5th Semester**

DimViz is a beginner-friendly web application for exploring **dimensionality reduction**. Upload a CSV dataset and use **PCA** and **t-SNE** to transform high-dimensional numerical data into two dimensions for visualization.

---

## 1. Features

- Upload any CSV dataset.
- Automatically detect numerical and categorical columns.
- Display rows, columns, numerical features and missing values.
- Handle missing numerical values using median imputation.
- Standardize numerical features using `StandardScaler`.
- Run **PCA** and view PC1/PC2.
- Show PCA explained variance and cumulative variance.
- Run **t-SNE** with safe perplexity handling.
- Color points using an optional categorical/class column.
- Interactive 2D charts using Plotly.js.
- PCA vs t-SNE comparison table.
- Friendly error messages for invalid datasets and parameters.
- Includes a ready-to-use student-performance sample dataset.

---

## 2. Technology Used

### Frontend
- HTML
- CSS
- JavaScript
- Plotly.js

### Backend
- Python
- Flask
- Flask-CORS

### Machine Learning / Data Processing
- Pandas
- NumPy
- Scikit-learn

---

## 3. Folder Structure

```text
Dimensionality_Reduction_Visualizer/
│
├── backend/
│   ├── app.py
│   ├── preprocessing.py
│   ├── pca.py
│   └── tsne.py
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── data/
│   └── sample_dataset.csv
│
├── requirements.txt
├── README.md
└── .gitignore
```

No virtual environment is included.

---

## 4. Requirements

Recommended:

- Python 3.10 or newer
- pip
- A modern browser

Python 3.11 is recommended for a smooth setup with the pinned Scikit-learn version.

---

## 5. Installation

Open Terminal in the project folder:

```bash
cd Dimensionality_Reduction_Visualizer
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows:

```text
.venv\Scripts\activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## 6. Run the Flask Backend

From the project root:

```bash
cd backend
python app.py
```

The backend runs at:

```text
http://127.0.0.1:5000
```

Keep this Terminal window running.

---

## 7. Open the Frontend

Open:

```text
frontend/index.html
```

in Chrome, Safari or another modern browser.

You can also use VS Code's **Live Server** extension.

The frontend sends requests to:

```text
http://127.0.0.1:5000
```

---

## 8. Example Workflow

1. Start Flask.
2. Open `frontend/index.html`.
3. Upload `data/sample_dataset.csv`.
4. Check the dataset statistics.
5. Select `Performance_Level` as the class column.
6. Click **Run PCA**.
7. Observe PC1, PC2 and explained variance.
8. Inspect the interactive PCA scatter plot.
9. Set t-SNE perplexity to `30`.
10. Click **Run t-SNE**.
11. Compare the PCA and t-SNE visualizations.
12. Upload another CSV and repeat.

---

## 9. How PCA Works

**Principal Component Analysis (PCA)** is a linear dimensionality reduction technique.

Before PCA, DimViz:

1. Selects numerical features.
2. Converts invalid numerical values to missing values.
3. Replaces missing values with the median.
4. Standardizes the features using `StandardScaler`.
5. Applies PCA.

For the 2D visualization:

- **PC1** = direction containing the highest variance.
- **PC2** = second most important direction of variance.

The explained variance ratio tells us how much information/variance each component captures.

Example:

```text
PC1: 42.5%
PC2: 21.8%
Total retained: 64.3%
```

A higher total retained variance means the two components preserve more of the original variance.

---

## 10. How t-SNE Works

**t-SNE (t-distributed Stochastic Neighbor Embedding)** is a non-linear dimensionality reduction technique.

It tries to keep **similar data points close together** in the lower-dimensional representation.

This makes t-SNE useful for:

- Visualizing clusters.
- Exploring local patterns.
- Understanding groups in high-dimensional data.

Important: t-SNE axes do not have the same direct mathematical interpretation as PCA's PC1 and PC2.

DimViz automatically adjusts perplexity if the requested value is invalid for a small dataset because Scikit-learn requires:

```text
perplexity < number of samples
```

---

## 11. PCA vs t-SNE

| Aspect | PCA | t-SNE |
|---|---|---|
| Type | Linear | Non-linear |
| Main idea | Maximize variance | Preserve local similarities |
| Speed | Usually faster | Usually slower |
| Interpretation | Components have mathematical meaning | Axes are mainly visual |
| Main use | Reduction + visualization | Cluster visualization |
| Reproducibility | Deterministic for the same data | Uses a fixed random state here |

### Simple memory trick

**PCA = Preserve maximum variance**

**t-SNE = Try to preserve nearby/similar points**

---

## 12. API Endpoints

### GET `/`

Checks that the Flask API is running.

Example:

```text
http://127.0.0.1:5000/
```

### POST `/api/upload`

Uploads and analyzes a CSV file.

Returns:

- Dataset ID
- Filename
- Rows
- Columns
- Numerical features
- Categorical features
- Missing values
- Preview
- Detected columns

### POST `/api/pca`

Runs PCA on the uploaded dataset.

Example JSON:

```json
{
  "dataset_id": "your-dataset-id",
  "class_column": "Performance_Level"
}
```

Returns:

- PC1 values
- PC2 values
- Explained variance
- Cumulative variance
- Class values

### POST `/api/tsne`

Runs t-SNE.

Example JSON:

```json
{
  "dataset_id": "your-dataset-id",
  "class_column": "Performance_Level",
  "perplexity": 30
}
```

Returns:

- Dimension 1
- Dimension 2
- Actual safe perplexity
- Class values

---

## 13. Error Handling

DimViz handles:

- Empty CSV files.
- Non-CSV uploads.
- CSV files with no rows.
- CSV files without numerical columns.
- Missing numerical values.
- Very small datasets.
- Invalid t-SNE perplexity.
- Missing dataset sessions.
- Backend processing errors.

---

## 14. Sample Dataset

`data/sample_dataset.csv` contains 160 synthetic student records with:

- Study_Hours
- Attendance
- Assignment_Score
- Midterm_Score
- Lab_Score
- Sleep_Hours
- Previous_GPA
- Final_Score
- Performance_Level

The dataset is synthetic and intended for educational demonstration only.

---

## 15. Viva Questions and Answers

### Q1. What is dimensionality reduction?
Dimensionality reduction reduces the number of features while trying to preserve useful information.

### Q2. Why do we use dimensionality reduction?
It makes high-dimensional data easier to visualize, analyze and sometimes process faster.

### Q3. What is PCA?
PCA is a linear technique that creates new components capturing maximum variance.

### Q4. What is PC1?
PC1 is the principal component containing the highest variance in the data.

### Q5. What is PC2?
PC2 is the second principal component and captures the next highest amount of variance.

### Q6. What is explained variance?
It tells how much of the original data variance is captured by each principal component.

### Q7. Why do we standardize data before PCA?
Features may have different scales, so standardization prevents large-scale features from dominating PCA.

### Q8. How are missing values handled?
DimViz replaces missing numerical values using median imputation.

### Q9. What is t-SNE?
t-SNE is a non-linear technique mainly used to visualize high-dimensional data and local clusters.

### Q10. What does perplexity mean in t-SNE?
Perplexity roughly controls the effective number of nearby points considered when preserving local structure.

### Q11. Why can t-SNE be slower than PCA?
t-SNE performs an iterative non-linear optimization, while PCA has a comparatively direct linear decomposition.

### Q12. Can t-SNE components be interpreted like PCA components?
No. t-SNE's dimensions are mainly useful for visualization and do not represent fixed directions of maximum variance.

### Q13. PCA vs t-SNE in one line?
PCA preserves variance with a linear transformation, while t-SNE emphasizes local similarities for visualization.

---

## 16. Important Note About Plotly.js

The dashboard uses the Plotly.js browser library from the official Plotly CDN so the interactive visualization requires internet access when the HTML page loads.

The **Flask backend and all ML processing are local** and do not require any API key or paid service.

If completely air-gapped/offline browser operation is required, a local copy of Plotly.js can be vendored into the `frontend` folder.

---

## 17. Educational Scope

This is a 5th-semester educational project demonstrating:

- Data preprocessing
- Standardization
- Dimensionality reduction
- PCA
- t-SNE
- Interactive visualization
- Basic Flask API development
- Frontend-backend integration

It is not intended as a production data-science platform.
