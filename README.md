## Loan Eligibility Prediction System

## Author: Ekeoma Onuoha
## GitHub: github.com/AirchildDev/LoanPrediction_Project

--------------------------------------------------------------------------------

## Overview

This project is an end to end machine learning system that predicts whether a loan applicant is eligible for approval, based on personal, financial, and credit history information. The system covers the full pipeline from raw data to a deployed, user facing application, including data preprocessing, model training and selection, evaluation, and deployment through a FastAPI web service with both a JSON API and a browser based form.

The dataset used is the standard loan eligibility dataset consisting of applicant demographic information, income details, loan terms, credit history, and property area, with a binary target indicating whether the loan was approved.

--------------------------------------------------------------------------------

## Project Structure

loan_prediction_project/
    LoanpredictEnv/            virtual environment, not committed to version control
    data/
        train.csv              training data with labels
        test.csv                unlabeled data for batch prediction
    model/
        loan_model.pkl          trained pipeline, produced by train_model.py
    outputs/
        model_comparison.csv    cross validation scores for all candidate algorithms
        model_comparison.png    bar chart comparing candidate algorithms
        confusion_matrix.png    confusion matrix for the selected model
        feature_importance.png  feature importance chart, when supported by the model
        metrics_report.txt      full text summary of model selection and evaluation
    templates/
        index.html              HTML form used by the browser based interface
    train_model.py               training script, produces model/loan_model.pkl
    predict_batch.py              generates predictions for data/test.csv
    main.py                       FastAPI application, JSON API and web form
    requirements.txt              exact package versions used in this project

--------------------------------------------------------------------------------

## Requirements

Python version 3.10 or higher is recommended.

The following packages are required and are pinned in requirements.txt:

pandas
scikit-learn
fastapi
uvicorn[standard]
joblib
pydantic
matplotlib
seaborn
jinja2
python-multipart

--------------------------------------------------------------------------------

## Environment Setup

Create the project folder and the virtual environment.

mkdir loan_prediction_project
cd loan_prediction_project
python -m venv LoanpredictEnv

Activate the virtual environment.

Windows:
LoanpredictEnv\Scripts\activate

Mac or Linux:
source LoanpredictEnv/bin/activate

Install the required packages.

pip install pandas scikit-learn fastapi uvicorn[standard] joblib pydantic matplotlib seaborn jinja2 python-multipart
pip freeze > requirements.txt

It is recommended to create the virtual environment outside the project folder, or to explicitly exclude it from the development server file watcher, since it contains a large number of files that are unrelated to the application source code.

--------------------------------------------------------------------------------

## Data

Place the training file at data/train.csv and, if batch prediction on unseen data is required, place the unlabeled file at data/test.csv.

The dataset includes the following columns.

Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area, Loan_Status

Loan_Status is the target column and is present only in the training file. It is encoded internally as 1 for approved and 0 for rejected.

--------------------------------------------------------------------------------

## Model Training

Run the training script from the project root.

python train_model.py

The script performs the following steps in order.

1. Loads and cleans the training data, removing the identifier column and any rows with a missing target value.
2. Builds a preprocessing pipeline that imputes missing numeric values with the median, imputes missing categorical values with the most frequent category, scales numeric features, and one hot encodes categorical features.
3. Splits the data into training and test sets, using stratified sampling to preserve the class balance.
4. Compares five candidate algorithms, Logistic Regression, Random Forest, Gradient Boosting, Support Vector Machine, and K Nearest Neighbors, using five fold cross validation, scored on accuracy, F1, and ROC AUC.
5. Selects the algorithm with the highest cross validated ROC AUC and trains it on the full training split.
6. Evaluates the selected model on the held out test set and prints accuracy and a full classification report.
7. Saves the trained pipeline to model/loan_model.pkl.
8. Saves a text summary of the comparison and evaluation to outputs/metrics_report.txt.
9. Saves a bar chart comparing all candidate algorithms to outputs/model_comparison.png.
10. Saves a confusion matrix for the selected model to outputs/confusion_matrix.png.
11. Saves a feature importance chart to outputs/feature_importance.png, when the selected algorithm exposes feature importances.

The selected algorithm is not fixed. It is chosen automatically at training time based on cross validated performance, and the winning algorithm can differ slightly between environments due to differences in library versions, since tree based ensemble methods are sensitive to internal implementation changes even when a fixed random seed is used.

--------------------------------------------------------------------------------

## Batch Prediction

To generate predictions for the unlabeled test file, run the following after training.

python predict_batch.py

This reads data/test.csv, applies the saved pipeline, and writes predictions together with approval probabilities to test_predictions.csv.

--------------------------------------------------------------------------------

## Running the Application

Development mode, with automatic reload on code changes.

uvicorn main:app --reload

If the virtual environment is located inside the project folder, restrict the file watcher to the application source files only, to avoid the reload process repeatedly triggering on changes inside the installed package files.

uvicorn main:app --reload --reload-dir main.py --reload-dir templates

The training script is intentionally excluded from the file watcher, since editing it has no effect on the running application until it is executed manually and produces a new model/loan_model.pkl file. After retraining, restart the server, or save any watched file, to load the newly trained model.

--------------------------------------------------------------------------------

## Using the Application

Web form interface.

Open http://127.0.0.1:8000 in a browser. The form accepts applicant details and returns an approval decision together with a confidence percentage, rendered on the same page.

JSON API.

Endpoint: POST /predict
Content type: application/json

Example request body.

{
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "0",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 5849,
    "CoapplicantIncome": 0,
    "LoanAmount": 128,
    "Loan_Amount_Term": 360,
    "Credit_History": 1,
    "Property_Area": "Urban"
}

Example response.

{
    "loan_status": "Approved",
    "approval_probability": 0.7427
}

Interactive documentation, generated automatically by FastAPI, is available at http://127.0.0.1:8000/docs.

A health check endpoint is available at GET /health.

--------------------------------------------------------------------------------

## Input Validation

All numeric fields, ApplicantIncome, CoapplicantIncome, LoanAmount, and Loan_Amount_Term, are required to be greater than or equal to zero. Credit_History is required to be either zero or one. These constraints are enforced on the server for both the JSON API and the web form, so invalid input is rejected even if a request bypasses the browser interface. Submitting the web form with an invalid value displays a message on the page rather than a raw error response.

--------------------------------------------------------------------------------

## Model Behavior and Limitations

The selected model places substantial weight on Credit_History when making its decision. In the training data, applicants with a good credit history were approved at a considerably higher rate than applicants with a poor credit history, independent of income level. As a result, the model can approve applicants with very low or zero recorded income when credit history is favorable, and can reject applicants with strong income when credit history is unfavorable.

This behavior reflects a pattern present in the historical data used for training, rather than an error in the model or the pipeline. Before this system is used to inform real lending decisions, it is recommended to review whether this weighting is appropriate for the intended use case, and to consider adding explicit business rules on top of the model output where necessary, for example a rule that rejects any application where total combined income is zero regardless of the model prediction.

--------------------------------------------------------------------------------

## Deployment

The application can be run in several ways depending on the target environment.

Local development.

uvicorn main:app --reload

Production style process management, without a container.

pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000

Long running deployment on a Linux server, managed by systemd, so that the application restarts automatically after a crash or a server reboot. A service file should be created under /etc/systemd/system, pointing ExecStart at the gunicorn command above, using the full path to the virtual environment.

Containerized deployment, using a Dockerfile that installs the dependencies from requirements.txt, copies main.py, templates, and model, and starts the application using the gunicorn command above.

Platform as a service deployment, for example Render or Railway, using either the Dockerfile or a native Python build configuration, with the start command set to the gunicorn command above, replacing the fixed port with the platform provided PORT environment variable where required.

--------------------------------------------------------------------------------

## Results Summary

The final selected model achieved an accuracy of approximately eighty two percent on the held out test set, drawn from a total of six hundred and fourteen labeled applicants. Full precision, recall, and F1 scores by class, along with the complete algorithm comparison table, are written to outputs/metrics_report.txt at the end of every training run, and will vary slightly depending on which algorithm is selected and the library versions installed in the environment.

--------------------------------------------------------------------------------

## License

This project is provided for educational and demonstration purposes.



