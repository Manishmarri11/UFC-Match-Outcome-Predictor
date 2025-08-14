# UFC Fight Outcome Predictor

This project is a machine learning model that predicts the winner of UFC fights using match-level fighter-vs-fighter statistics and historical fight data. The model achieved a 66.5% accuracy, which is very tough to attain in highly unpredictable sports like UFC. The model uses features like betting odds, win/loss streaks, fight record breakdowns (KO, submission, decision), and physical attributes (height, reach, weight) to generate probability-based predictions.

The project includes an interactive Streamlit web app where users can select two fighters. If the fighters are from different weight classes, a warning is issued. Otherwise, the model predicts the outcome and shows winning probabilities.

**Performance Metrics:**
- Accuracy: 66.5%
- F1 Score: 74%
