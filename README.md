# MLOps Fundamentals
- Used Sklearn's MLPClassifier to perform classification on a diabetes dataset.

## DVC
### When a push is made to a new branch the following occurs:
- Runs <b>dvc.yaml</b> to execute all four 02_ scikit-learn scripts.
- DVC syncs with GitHub Actions to generate a report (.github\workflows\dvc_report.yaml).
- This pipeline collects data, processes data, trains the model (as per params.yaml) and outputs metrics.
- It also executes ./tests/test_fit.py for basic unit tests, etc.

## WandB
- Running model_algorithm() in 03a_wandb.. will fit the classifier and display the performance on WandB.
- You must do this for each individual model you want to add to the comparisons.
- You can use Sweep for further hyperparameter tuning once the algorithm is chosen.
