# MachineLearningAHS

A collection of machine learning projects, from classic scikit-learn classifiers to neural networks and a genetic algorithm that plays Tetris.

## Projects

| # | Project | Task | Approach | Result |
|---|---------|------|----------|--------|
| 0 | [Fruit Classification](Project_0) | Classify fruit from color, shape, luster and texture | Decision tree, logistic regression, ridge classifier | — |
| 1 | [Titanic Survival](Project_1) | Predict whether a passenger survived | Decision tree, SVM (one-vs-one), random forest, logistic regression + random search tuning | ~84% accuracy (random forest), ~83% (SVM OvO) |
| 2 | [Chicago House Prices](Project_2) | Predict a home's selling price | Scraped listings with HomeHarvest; random forest, linear, SVR, KNN, gradient boosting, XGBoost | Tuned XGBoost, R² ≈ 0.73 |
| 3 | [Customer Purchase Prediction](Project_3) | Predict whether a customer buys an item | Rolling-mean trend features; LSTM, KNN, random forest, boosting, Naive Bayes | Bernoulli Naive Bayes, ~57–59% CV accuracy |
| 4 | [Spam Filter](Project_4) | Classify emails as spam or ham | Hand-built syntax/sender features + count vectorization; Multinomial Naive Bayes | ~98% test and CV accuracy |
| 5 | [Cursive Letter Recognition](Project_5) | Identify handwritten cursive letters | OpenCV preprocessing; PyTorch CNN trained on class data combined with EMNIST | ~90% accuracy |
| 6 | [AI Tetris Player](Project_6) | Survive as long as possible in Tetris | Genetic algorithm with custom board features (holes, bumpiness, peaks, line clears) | Longest run: 49 hours |

## Built with
Python, Jupyter, pandas, NumPy, scikit-learn, XGBoost, PyTorch, OpenCV, NLTK, Pygame
