# GoodBooks Recommendation System

## Project Overview
This project implements a complete book recommendation system based on the GoodBooks-10k dataset, encompassing four main tasks: data preparation, user-based collaborative filtering, item-based collaborative filtering, and SVD model evaluation. The project aims to demonstrate the implementation and performance comparison of different recommendation algorithms, as well as how to address the challenges of high-sparsity datasets.

## Dataset Description
- **Data Source**: GoodBooks-10k dataset
- **Main Files**:
  - `ratings.csv`: Contains approximately 1,000,000 user rating records
  - `books.csv`: Contains detailed information on 10,000 books (title, author, ISBN, etc.)
- **Data Characteristics**: Ratings range from 1-5 stars, the dataset is large but highly sparse

## Project Structure
```
├── Book-Recomm-Sys.ipynb     # Jupyter Notebook integrated version
├── README.md                 # Project documentation (English)
├── README-ZH.md              # Project documentation (Chinese)
├── requirements.txt          # Dependency list
├── main.py                   # Main program (executes all tasks in sequence)
├── inspect_data.py           # Preliminary dataset structure analysis
├── task1_data_preparation.py # Data preparation and sparsity analysis
├── task2_user_based_CF.py    # User-based collaborative filtering algorithm
├── task3_item_based_CF.py    # Item-based collaborative filtering algorithm
├── task4_SVD_evaluation.py   # SVD matrix factorization model evaluation
├── books.csv                 # Book information data
├── ratings.csv               # User rating data
├── filtered_ratings.csv      # Filtered rating data
├── user_ratings_distribution.png  # User rating distribution chart
├── svd_predictions_scatter.png    # SVD prediction results scatter plot
└── svd_factors_rmse.png           # Latent factor analysis chart
```

## Functional Modules Explained

### 1. Data Preparation and Sparsity Analysis
- **Function**: Filter active users and popular books, analyze sparsity
- **Implementation**:
  - Select the 300 most active users and 300 most popular books
  - Calculate and analyze the sparsity of the user-item matrix
  - Generate user rating distribution visualization
- **Output**: `filtered_ratings.csv`, `user_ratings_distribution.png`
- **Core Metrics**: Matrix sparsity (typically >95%)

### 2. User-Based Collaborative Filtering
- **Function**: Recommend books to target users based on user similarity
- **Implementation**:
  - Construct user-item rating matrix
  - Calculate cosine similarity between users
  - Compute predicted ratings based on weighted average of similar users' ratings
  - Recommend unread books with highest predicted ratings
- **Advantages**: Intuitive and easy to understand, can discover users' potential interests
- **Disadvantages**: Performance degrades with high sparsity data, high computational cost

### 3. Item-Based Collaborative Filtering
- **Function**: Recommend books similar to those the user has rated highly
- **Implementation**:
  - Construct item-item similarity matrix
  - Find similar books starting from highly-rated books by the user
  - Display detailed information of recommended books
- **Advantages**: Better stability, smaller cold-start problem for new users
- **Disadvantages**: May lack content diversity, difficult to discover completely different types of interests

### 4. SVD Matrix Factorization Model
- **Function**: Use matrix factorization techniques for dimensionality reduction and user rating prediction
- **Implementation**:
  - Train model using the SVD algorithm from the Surprise library
  - Analyze the impact of different numbers of latent factors on performance
  - Calculate prediction RMSE and visualize results
- **Advantages**: Can effectively handle sparse data, solid theoretical foundation
- **Disadvantages**: Complex parameter tuning, lower interpretability

## Implementation Challenges and Solutions

### 1. Data Sparsity Problem
- **Challenge**: Extremely high sparsity (>95%) in filtered dataset affects recommendation quality
- **Solutions**:
  - Expand filtering range to 300 users/books (originally planned for 100)
  - Dynamically adjust similarity calculation methods and thresholds
  - Use simple averaging instead of weighted averaging when weights sum to zero

### 2. User Recommendation Condition Problem
- **Challenge**: Few users rated more than 5 books, making it difficult to find suitable target users
- **Solutions**:
  - Remove minimum rating quantity threshold
  - Directly select users with the most ratings as target users
  - Add exception handling logic

### 3. Balancing Computational Efficiency and Accuracy
- **Challenge**: Similarity calculations are computationally expensive and less accurate on sparse data
- **Solutions**:
  - Optimize cosine similarity calculation process
  - Add edge case handling
  - Dynamically adjust latent factor quantity for SVD model

### 4. Book Information Matching Problem
- **Challenge**: Some book_ids don't have corresponding information in books.csv
- **Solutions**:
  - Add ID existence checking
  - Use appropriate placeholder information for missing data
  - Enhance error handling capabilities

## System Performance and Metrics
- **Sparsity**: Typical sparsity between 95-98%
- **Recommendation Accuracy**: SVD model RMSE typically between 0.8-1.2
- **Computational Efficiency**: Fast running speed on filtered dataset, full dataset requires optimization
- **Cold Start Handling**: System can handle new users, but recommendation quality may be poor

## Running Instructions
```bash
# Install dependencies
pip install -r requirements.txt

# Run all tasks
python main.py

# Or run tasks individually
python task1_data_preparation.py
python task2_user_based_CF.py
python task3_item_based_CF.py
python task4_SVD_evaluation.py
```

## Performance Optimization and Future Improvements
1. **Data Level**:
   - Use complete dataset for training, rather than highly filtered subset
   - Consider introducing implicit feedback data to enhance sparse matrices
   - Use richer book metadata to assist recommendations

2. **Algorithm Level**:
   - Implement hybrid recommendation systems combining content and collaborative filtering
   - Try deep learning recommendation models such as NCF, DeepFM, etc.
   - Implement more types of matrix factorization techniques for sparse data

3. **Evaluation Level**:
   - Add more comprehensive evaluation metrics such as precision, recall, F1 score
   - Implement cross-validation to prevent overfitting
   - Add A/B testing framework to evaluate recommendation quality

4. **Engineering Level**:
   - Optimize large-scale data processing performance
   - Add capability for incremental updates to recommendation models
   - Implement API interfaces for external system calls

## Dependencies
- Python 3.6+
- pandas
- numpy
- scikit-learn==1.0.2
- scikit-surprise==1.1.3
- matplotlib==3.5.3

## Author
- Chen Suzeyu
- chensuzeyu@qq.com

## License
- MIT License
