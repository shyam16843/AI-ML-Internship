# Movie Recommendation System using Collaborative Filtering

## Project Description
This project implements a comprehensive movie recommendation system using collaborative filtering with cosine similarity and matrix factorization (SVD). The system provides personalized movie recommendations, similarity analysis, and detailed explanations for why movies are recommended to users.

## 1. Project Objective
Develop a robust movie recommendation system that can:

- Provide personalized movie recommendations based on user preferences

- Identify similar movies using content-based features

- Handle cold start problems with SVD backup and popularity fallback

- Explain why specific recommendations are made to users

- Evaluate system performance with comprehensive metrics

## 2. Dataset Information
- **Source**: MovieLens Latest Small dataset (ml-latest-small.zip)
- **Records**: 100,836 ratings from 610 users on 9,724 movies
- **Files**:
- ratings.csv: User ratings (userId, movieId, rating, timestamp)
- movies.csv: Movie information (movieId, title, genres)
- **Rating Range**: 0.5 to 5.0 stars
- **Average Range**: 3.50 stars

## 3. Methodology
### Data Preprocessing
- **Data Cleaning**: Filtered for movies with sufficient ratings (≥10) and active users (≥5 ratings)
- **Feature Engineering**:
  - Extracted year from movie titles
  - One-hot encoded genres for hybrid recommendations
  - Created user-item rating matrix
- **Recommendation Approaches**:
  - **Primary Method**: Collaborative Filtering with Cosine Similarity
    - Item-item similarity matrix using cosine similarity
    - Similarity thresholding (min_similarity=0.1)
    - Weighted average prediction with user rating adjustment
  - **Secondary Method**: Matrix Factorization (SVD)
    - TruncatedSVD for dimensionality reduction
    - Randomized SVD for efficient computation
    - Used as backup for cold start problems
  - **Fallback Method**:  Popularity-based Recommendations
    - Most rated movies with highest average ratings
    - Genre-based filtering when preferences are known

### Advanced Features
- **Parameter Tuning**: Automated optimization of similarity thresholds
- **Recommendation Diversification**: Ensures varied recommendations (max_similarity=0.7)
- **Explanation System**: Shows why movies are recommended based on user's rating history
- **Hybrid Features**: Combines collaborative filtering with content-based features

## 4. System Performance
### Data Processing Results
- **Original Data**: 100,836 ratings, 9,742 movies, 610 users
- **Filtered Data**: 25,759 ratings, 1,107 movies, 606 users
- **User-Item Matrix**: (606, 1107) shape
- **Similarity Matrix**: (1107, 1107) shape

### Recommendation Quality
- **Average Predicted Rating**: 4.02 (realistic range)
- **Prediction Range**: 3.205 - 4.704
- **Precision@5**: Evaluated through train-test split
- **Diversity Score**: Measures variety in recommendations

## 5. Key Features Implemented
### Core Functionality
- Personalized movie recommendations for individual users
- Movie similarity analysis ("movies like this")
- User rating history profiling
- Comprehensive evaluation metrics

### Technical Features
- Cosine similarity-based collaborative filtering
- SVD matrix factorization backup system
- Automated parameter tuning
- Recommendation diversification
- Cold start problem handling
- Explanation system for recommendations

### Visualization & Analysis
- Rating distribution charts
- Top movies visualization
- Similarity heatmaps
- Cluster analysis visualizations

## 6. Business Applications
### For Streaming Platforms
- Personalized content discovery
- Improved user engagement
- Reduced churn through better recommendations
- Cross-selling similar content

### For Users
- Discover new movies based on preferences
- Understand why recommendations are made
- Explore similar content to favorites
- Get diversified suggestions

## 7. Project Setup and Requirements

### Requirements
- Python 3.x
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- requests


### Installation
Install dependencies by running:

```bash
pip pip install pandas numpy matplotlib seaborn scikit-learn requests
```

### Running the Project
1. The system automatically downloads the MovieLens dataset.
2. Run the main script:
```bash
python movie.py
```

### The system will:

   -  Download and preprocess data
   -  Train recommendation models
   -  Generate recommendations and explanations
   -  Create visualizations and export results

### Outputs saved:
   - **optimized_similarity_matrix.csv**: Cosine similarity between all movies
   - **tuned_parameters.csv**: Optimized hyperparameters for the system
   - **comprehensive_recommendations.csv**: Recommendations with explanations

## 8. Visualization Overview
A comprehensive set of visualizations supporting this project is provided separately in the [Visualization Document](Visualizations.md). This document includes detailed descriptions and analyses of all key plots

### Accessing Visualizations

The actual plot images referenced in the visualization document are stored in the `/images` directory within the project repository.

We recommend reviewing the visualization document alongside the main README for a thorough understanding of the model's performance and insightful data interpretations.

## 9. Future Enhancements

### Technical Improvements
- Implement deep learning-based recommendations
- Add real-time recommendation capabilities
- Incorporate temporal dynamics of user preferences
- Add social features and friend recommendations

### Business Features
- Multi-platform integration
- A/B testing framework
- Recommendation performance monitoring
- Personalized email recommendations

### User Experience
- Interactive web interface
- Mobile app integration
- Voice-based recommendations
- Group recommendation features

## 10. Contact
For questions or collaboration:
- **Name**: Ghanashyam T V
- **Email**: ghanashyamtv16@gmail.com
- **LinkedIn**: [linkedin.com/in/ghanashyam-tv](https://www.linkedin.com/in/ghanashyam-tv)

---

Thank you for exploring the Movie Recommendation System! This project demonstrates advanced collaborative filtering techniques with practical applications for content discovery and personalized recommendations.

---