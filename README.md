# Design Starbucks Rewards Program

Using data provided by Starbucks, this project aims to explore the implementation of a hybrid model that combines 
gradient boost decision trees (GBDT) with logistic regression (LR), as proposed by the Facebook research paper
[“Practical Lessons from Predicting Clicks on Ads at Facebook”](https://research.fb.com/wp-content/uploads/2016/11/practical-lessons-from-predicting-clicks-on-ads-at-facebook.pdf) 
to predict whether a customer would view a Starbucks App promotion offer.

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Future Work](#futurework)
5. [Licensing, Authors, and Acknowledgments](#licensing)

## Installation <a name="installation"></a>

Python 3.0 is required to run the code.

In addition, the following libraries are also required. 
- sqlite3
- pandas
- numpy
- sklearn
- lightgbm
- bayes_opt
- matplotlib
- seaborn

## Project Motivation <a name="motivation"></a>

By identifying the offers with which customers are most likely to engage, this project provides an approach using 
machine learning techniques to devise an optimized strategy that drives the highest return on each offer sent 
by Starbucks.

## File Descriptions <a name="files"></a>

The following Jupyter notebook files provide details of each exploration step with data visualizations. The 
filenames explain what each file does.

- `1_Data_Reformat_Exploration_Cleaning.ipynb`
- `2_Create_Trx_Based_Features.ipynb`
- `3_EDA_Normalization_MissingValue.ipynb`
- `4_Modeling.ipynb`

The following python scripts are organized so as to provide single-point access to run the whole program. 

- `main.py`
- `load_data.py`
- `data_transformation.py`
- `create_additional_trx_features.py`
- `modeling.py`
- `utils.py`

By simply running the following command in your command line in the correct directory, you will be able to get 
a comparison of results from each model utilized in this project and predicted results generated from each model, 
saved as .csv files:

```
python main.py
```

Also included in this repository:
- `Proposal.pdf`: proposal for this project
- `Project Report.pdf`: final report for this project
- `result\lgbm_predictions.csv`, `result\lgbm_predictions.csv`, `result\lgbm_predictions.csv`: predicted results generated from each model
- `data\portfolio.json`, `data\profile.json`, `data\transcript.json`, `data\portfolio_clean.csv`,`data\profile_clean.csv`,`data\transcript_clean.csv`:  raw data as JSON files and cleaned data as CSV files 

## Results <a name="results"></a>

This project compares the performance of the hybrid model (Gradient Boosting Decision Trees + Logistic Regression) 
against the performance of using the Logistic Regression model alone and using the Gradient Boosting Decision 
Tree model alone.

The evaluation metric is `Normalized Entropy` as proposed in the aforementioned Facebook paper.

Below is the summary of the comparison result. 

|Model     | NE (training) | NE (validation) | NE (Testing)    |
|----------|---------------|-----------------|-----------------|
|GBDT + LR | 1.03e-05% 	   | 1.06e-05% 		 | **1.54e-05%**   |
|GBDT only | 0.31% 		   | 0.32% 			 | **1.12%** 	   |
|LR only   |45.74% 		   | 45.64% 		 | **83.00%** 	   |

The GBDT-only model achieves a significantly lower Normalized Entropy than the LR-only model. As such, the GBDT-
only model is remarkably more performant than LR in this specific task.

However, even though the GBDT-only model already achieves small Normalized Entropy, the hybrid (GBDT + LR) model generates extremely small testing Normalized Entropy ( 1.54e-05%).

This project arrives at the same conclusion as the Facebook paper: that combining the decision
tree based model with probabilistic linear classifiers significantly improves the prediction
accuracy of a model using just the probabilistic linear classifiers alone.

## Future Work <a name="futurework"></a>

Based on the hybrid model’s almost perfect Normalized Entropy result, the hybrid model might
be too complex for this project. A good model not only should be able to predict results
accurately but also should maintain a simple structure. Future work might involve applying the
hybrid model to a larger dataset or other tasks to test if this model can be well extended to a
broad range of use cases.

## Licensing, Authors, Acknowledgments <a name="licensing"></a>

Many thanks to Starbucks for providing the data.
