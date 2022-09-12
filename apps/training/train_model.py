from apps.core.logger import Logger
import json
from sklearn.model_selection import train_test_split
from apps.core.file_operation import FileOperation
from apps.tuning.model_tuner import ModelTuner
from apps.ingestion.load_validate import LoadValidate
from apps.preprocess.preprocessor import Preprocessor
from apps.tuning.cluster import KMeansCluster


class TrainModel:
    def __init__(self, run_id, data_path):
        self.run_id = run_id
        self.data_path = data_path
        self.logger = Logger(self.run_id, 'TrainModel', 'training')
        self.loadValidate = LoadValidate(self.run_id, self.data_path, 'training')
        self.preProcess = Preprocessor(self.run_id, self.data_path, 'training')
        self.modelTuner = ModelTuner(self.run_id, self.data_path, 'training')
        self.fileOperation = FileOperation(self.run_id, self.data_path, 'training')
        self.cluster = KMeansCluster(self.run_id, self.data_path)

    def training_model(self):
        try:
            self.logger.info('Start of Training')
            self.logger.info('Run_id:' + str(self.run_id))
            self.loadValidate.validate_trainset()
            self.X, self.y = self.preProcess.preprocess_trainset()
            columns = {"data_columns":[col for col in self.X.columns]}
            with open('apps/database/columns.json','w') as f:
                f.write(json.dumps(columns))
            number_of_clusters = self.cluster.elbow_plot(self.X)
            self.X= self.cluster.create_clusters(self.X, number_of_clusters)
            self.X['Labels'] = self.y
            list_of_clusters = self.X['Cluster'].unique()
            for i in list_of_clusters:
                cluster_data=self.X[self.X['Cluster'] == i]
                cluster_features=cluster_data.drop(['Labels', 'Cluster'], axis=1)
                cluster_label= cluster_data['Labels']
                x_train, x_test, y_train, y_test = train_test_split(cluster_features, cluster_label, test_size=0.2, random_state = 0)
                best_model_name, best_model = self.modelTuner.get_best_model(x_train, y_train, x_test, y_test)
                save_model = self.fileOperation.save_model(best_model, best_model_name+str(i))
            self.logger.info('End of Training')
        except Exception:
            self.logger.exception('Unsuccessful End of Training')
            raise Exception