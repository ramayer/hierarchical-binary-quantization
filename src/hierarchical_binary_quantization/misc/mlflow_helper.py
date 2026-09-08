import mlflow
from mlflow.tracking import MlflowClient

# uv run mlflow ui
class MLFlowHelper:
    """
    Avoid mlflow hanging when running multiple training runs in parallel.
    https://share.google/aimode/0xsbUN6W8CnyGRV9q
    """

    def __init__(self, experiment_name, run_name=None, loggable_params={"reminder":"log your parameters"}):

        self.experiment_name = experiment_name
        self.run_name = run_name
        self.loggable_params = loggable_params
        self.logged_step = 0
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(self.experiment_name)

        if run_name is None:
            with mlflow.start_run() as run:
                self.run_name = run.info.run_name
                self.run_id = run.info.run_id

        self.run_id = self.get_mflow_run_id()
        if not self.run_id:
            with mlflow.start_run(run_name=self.run_name):
                pass
            self.run_id = self.get_mflow_run_id()

        with mlflow.start_run(run_id=self.run_id):
            for k,v in loggable_params.items():
                mlflow.log_param(k,v)  

    def get_mflow_run_id(self):
        runs = mlflow.search_runs(
            experiment_names=[self.experiment_name],
            filter_string=f"attributes.run_name = '{self.run_name}'"
        )

        if not runs.empty:
            mlflow_run_id = runs.iloc[0]["run_id"]
            print(f"Found existing run: {mlflow_run_id}. Resuming...")
            return mlflow_run_id
        else:
            print(f"No existing run found with run_name {self.run_name}.")
            return None

    def log_to_mlflow(self, mlflow_step, metrics):
        with mlflow.start_run(run_id=self.run_id):
            for k,v in metrics.items():
                mlflow.log_metric(k, v, step=mlflow_step)


# for i in range(3):
#     mfh.log_to_mlflow(i, {"elapsed_seconds":1,"samples_per_second":1,"loss":1})
# mlflow.end_run()
