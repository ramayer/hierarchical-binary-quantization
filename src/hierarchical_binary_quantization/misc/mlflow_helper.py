import mlflow
from mlflow.tracking import MlflowClient
from dataclasses import asdict,is_dataclass

# ------------------------------------------------------------
# 🔥 MLFLOW: LOG METRICS
# View with mlflow ui
# ------------------------------------------------------------



import mlflow
from mlflow.tracking import MlflowClient

def emergency_mlflow_reset(new_db_path="mlflow_recovered.db"):
    print("⚠️ Purging stale MLflow connections and tracking registries...")
    
    # 1. Clear out internal SQLAlchemy store registries completely
    try:
        mlflow.tracking._tracking_service.utils._tracking_store_registry.stores.clear()
    except Exception:
        pass

    # 2. Point to a brand new SQLite destination file to force schema generation
    new_uri = f"sqlite:///{new_db_path}"
    mlflow.set_tracking_uri(new_uri)
    
    # 3. Instantiate a fresh client isolated from previous file handles
    client = MlflowClient(tracking_uri=new_uri)
    
    # 4. Spin up a new experiment and run structure 
    # (Since the old run uuid cannot be mapped to the deleted schema)
    try:
        # Gracefully sever ties with the active context if it exists
        mlflow.end_run() 
    except Exception:
        pass
        
    exp_id = mlflow.set_experiment("recovered_runs")
    new_run = mlflow.start_run()
    
    print(f"🚀 MLflow successfully bound to clean backend! New Run ID: {new_run.info.run_id}")
    return new_run



class MLFlowHelper:
    """
    Avoid mlflow hanging when running multiple training runs in parallel,
    and easier continuing mlflow runs after pausing and checkpointing.
    https://share.google/aimode/0xsbUN6W8CnyGRV9q
    """

    def __init__(self, experiment_name, loggable_params, run_name=None, extra_params={}):

        try:

            self.experiment_name = experiment_name
            self.run_name = run_name
            self.loggable_params = loggable_params
            self.logged_step = 0
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            mlflow.set_experiment(self.experiment_name)
            
            if is_dataclass(loggable_params):
                loggable_params = asdict(loggable_params) # type:ignore

            loggable_params = loggable_params | extra_params
            
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
        except mlflow.MlflowException:
            print("mlflow exception - resetting")
            emergency_mlflow_reset()


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
        try: 
            with mlflow.start_run(run_id=self.run_id):
                for k,v in metrics.items():
                    mlflow.log_metric(k, v, step=int(mlflow_step))
        except mlflow.MlflowException:
            print("mlflow exception - resetting")
            emergency_mlflow_reset()

