# Classification Model for Imbalanced Three-Class Dataset

This repository contains the implementation of a classification model aimed at solving a three-class classification problem with imbalanced data. The goal of this project was to predict the target variable, which contains three possible classes: `0`, `1`, and `2`. Class `0` represents the majority class, with approximately 80% of the data, class `2` contains around 15%, and class `1` has a very small representation.

## Project Overview

In this project, we experimented with multiple machine learning models and data preprocessing techniques to address the class imbalance issue and improve prediction accuracy. The process included the following steps:

1. **Initial Model Selection**: We started by applying Logistic Regression, but the results were unsatisfactory due to the high imbalance in the dataset. The model also failed to predict the minority class (`target = 1`), leading to an `UndefinedMetricWarning`.
2. **Data Preprocessing**: We explored the data and found inconsistent values in certain categorical features (e.g., `sex`), which were then cleaned by removing rows with invalid values. We also identified the class imbalance and considered strategies like resampling to address it.
3. **Model Selection and Hyperparameter Tuning**:
   - **Logistic Regression**: After applying the One-vs-Rest strategy and balancing class weights, the performance of Logistic Regression improved, but it still failed to predict class `1` correctly.
   - **K-Nearest Neighbors (KNN)**: We tested KNN with various configurations and used GridSearchCV for hyperparameter tuning. The best-performing parameters were `n_neighbors=2` and `metric='manhattan'`.
   - **Decision Tree Classifier**: Despite attempts to optimize hyperparameters and apply class weights, the decision tree model didn’t show significant improvements.
   - **Random Forest Classifier**: Similar to the decision tree, we tried hyperparameter optimization, but the default parameters gave the best results.
4. **Handling Imbalanced Data**:
   - We experimented with **SMOTE (Synthetic Minority Over-sampling Technique)** to address class imbalance by generating synthetic examples for the minority class. This approach significantly improved model performance compared to standard scaling techniques like **StandardScaler**.
   - We also explored the effects of removing irrelevant features and adjusting feature importance based on their distribution to optimize the training dataset.
5. **Final Model Selection**:
   - After optimizing the models, the best performing models were **KNN** (`n_neighbors=2`, `metric='manhattan'`) and **Random Forest** (default parameters). These models provided the most balanced and accurate results, particularly for the minority class.

## Technologies and Libraries Used

- Python 3.x
- scikit-learn
- pandas
- numpy
- matplotlib
- imbalanced-learn (for SMOTE)

## How to Run the Code

1. Clone the repository:

   ```
   bash
   
   
   复制代码
   git clone https://github.com/yourusername/repository-name.git
   ```

2. Install dependencies:

   ```
   bash
   
   
   复制代码
   pip install -r requirements.txt
   ```

3. Run the script to train the model:

   ```
   bash
   
   
   复制代码
   python model_training.py
   ```

## Model Results

- **KNN** (`n_neighbors=2`, `metric='manhattan'`): The KNN model achieved the best performance, handling imbalanced classes effectively.
- **Random Forest**: This model showed solid performance with the default configuration.
- **Decision Tree**: Despite multiple optimizations, this model performed less effectively than KNN and Random Forest.

## Conclusion

The final model selected for this project includes a K-Nearest Neighbors model with optimized hyperparameters (`n_neighbors=2`, `metric='manhattan'`). This model, combined with SMOTE for handling class imbalance, performed significantly better than other models. Random Forest was also a strong contender, but KNN provided the best results for this specific problem.

## Future Work

Future improvements could include:

- Exploring other resampling techniques like **ADASYN** or **Borderline-SMOTE**.
- Fine-tuning more hyperparameters across models.
- Trying deep learning models for better performance on more complex data.

## License

This project is licensed under the MIT License - see the LICENSE file for details.