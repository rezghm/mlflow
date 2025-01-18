import warnings
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow
import mlflow.sklearn
from pathlib import Path
import os

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

#get arguments from command
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, required=False, default=0.7)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.7)
args = parser.parse_args()

#evaluation function
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from local
    data = pd.read_csv("red-wine-quality.csv")
    # os.mkdir('data/')
    data.to_csv('data/red-wine-quality.csv', index=False)

    # data.to_csv("data/red-wine-quality.csv", index=False)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)
    train.to_csv('data/train.csv', index=False)
    test.to_csv('data/test.csv', index=False)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = args.alpha
    l1_ratio = args.l1_ratio

    # mlflow.set_tracking_uri(uri="")
    # mlflow.set_tracking_uri(uri="./mytracks")
    # mlflow.set_tracking_uri(uri="file:/Users/r3z8/PycharmProjects/test1")
    mlflow.set_tracking_uri(uri="")
    print("SET_TRACKING_URI IS: ", mlflow.get_tracking_uri())
    # exp = mlflow.set_experiment(experiment_name="experiment_for_uri")
#     exp_id = mlflow.create_experiment(
#         name="exp_create_exp_artifact",
#         tags={"version": "v1", "priority": "p1"},
#         artifact_location=Path.cwd().joinpath('myartifacts').as_uri()
# )
#     exp = mlflow.set_experiment(experiment_name="experiment_1")
    exp = mlflow.set_experiment(experiment_name="experiment_5")
    # get_exp = mlflow.get_experiment(exp_id)

    # print(f"Name: {get_exp.name}")
    # print(f"Experiment_id: {get_exp.experiment_id}")
    # print(f"Tags: {get_exp.tags}")
    # print(f"Lifecycle_stage: {get_exp.lifecycle_stage}")
    # print(f"Creation timestamp: {get_exp.creation_time}")

    print(f"Name: {exp.name}")
    print(f"Experiment_id: {exp.experiment_id}")
    print(f"Tags: {exp.tags}")
    print(f"Lifecycle_stage: {exp.lifecycle_stage}")
    print(f"Creation timestamp: {exp.creation_time}")


    # with mlflow.start_run(experiment_id=exp.experiment_id):
    # with mlflow.start_run(experiment_id=exp_id):
    # with mlflow.start_run(experiment_id=exp.experiment_id):
    # with mlflow.start_run(experiment_id=exp.experiment_id, run_name='run_1'):
    # with mlflow.start_run(run_id='5b18884725bc479c94fdc5c6d74b0a53'):
    #     lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    #     lr.fit(train_x, train_y)
    #
    #     predicted_qualities = lr.predict(test_x)
    #
    #     (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)
    #
    #     print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
    #     print("  RMSE: %s" % rmse)
    #     print("  MAE: %s" % mae)
    #     print("  R2: %s" % r2)
    #
    #     mlflow.log_param("alpha", alpha)
    #     mlflow.log_param("l1_ratio", l1_ratio)
    #     mlflow.log_metric("rmse", rmse)
    #     mlflow.log_metric("r2", r2)
    #     mlflow.log_metric("mae", mae)
    #     mlflow.sklearn.log_model(lr, "my_new_model")
    mlflow.start_run()

    mlflow.set_tag("release.version", "0.1")

    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)

    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    # mlflow.log_param("alpha", alpha)
    # mlflow.log_param("l1_ratio", l1_ratio)
    params = {
        'alpha': alpha,
        'l1_ratio': l1_ratio
    }
    mlflow.log_params(params)
    # mlflow.log_metric("rmse", rmse)
    # mlflow.log_metric("r2", r2)
    # mlflow.log_metric("mae", mae)
    metrics = {
        'rmse': rmse,
        'r2': r2,
        'mae': mae
    }
    mlflow.log_metrics(metrics)

    mlflow.sklearn.log_model(lr, "my_new_model_1")
    # mlflow.log_artifact('red-wine-quality.csv')
    mlflow.log_artifacts('data/')

    artifact_uri = mlflow.get_artifact_uri()
    print('The artifact path is:', artifact_uri)

    # run = mlflow.active_run()
    # run = mlflow.last_active_run()
    # print(f"Active run id is: {run.info.run_id}")
    # print(f"Active run name is: {run.info.run_name}")

    mlflow.end_run('FINISHED')
    # can only use last_active_run here
    run = mlflow.last_active_run()
    print(f"Active run id is: {run.info.run_id}")
    print(f"Active run name is: {run.info.run_name}")







