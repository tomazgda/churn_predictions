# churn_predictions

## Overview
A project for my work experience at Sky UK.

Dataset used is the Telco Customer Churn dataset:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn.

Built with python, [scikit-learn](https://scikit-learn.org/stable/index.html) and [pandas](https://pandas.pydata.org/).

Note: there are so many commits because I didn't set up a local building environment - everything was done through github.

## Goal
The goal is to be able to input some training data, and data without targets ('New' data), and output predictions for the new data as well as scores in the form of a report on model performance.

![Process](process.png)

## As it stands
Right now the action builds a model, fits the training data, makes some predictions and produces a report including accuracy and mean absolute error against a split testing dataset.

- the training data is located under data/ in the repository 
- the report is exposed as a Github Actions artefact labeled 'model_report'

## Todo
- [X] Seperate scores script from model building
- [ ] Remake the process chart to reflect: run-once -> build model & produce report, run-again -> log predictions
