import pandas as pd
import pickle

class Params:
    def __init__(self, **kwargs):
        self.problem=''
        self.data_path=""
        self.data_process_save_path=''
        self.param_save_path=''
        self.model_save_path=''
        self.fig_path=''
        self.train_sheet=''
        self.val_sheet=''
        self.is_onehot=True
        self.is_value=True
        self.is_save_processdata=False
        self.one_hot_feature=[]
        self.Ordinal_features=[]
        self.value_feature=[]
        self.max_cat_numbers=10
        self.X_drop_feature=[]
        self.y=''
        self.get_value_feature_names()
        self.cat_feature_index=self.get_feature_index(self.one_hot_feature+self.Ordinal_features)
        self.value_feature_index=self.get_feature_index(self.value_feature)
        self.train_samples=0
        self.val_samples=0
        self.model_name=None


    def display(self):
        # 显示所有的参数和它们的值
        for key, value in self.__dict__.items():
            print(f"{key}={value}")

    def get_value_feature_names(self):
        data = pd.read_excel(self.data_path, sheet_name=self.train_sheet)
        feature_columns = [col for col in data.columns if col not in self.X_drop_feature and col != self.y]
        for col in feature_columns:
            unique_count = data[col].nunique()
            if unique_count > self.max_cat_numbers:
                self.value_feature.append(col)

    def get_feature_index(self,names):
        data = pd.read_excel(self.data_path, sheet_name=self.train_sheet)
        index = [data.columns.get_loc(name) if name in data.columns else None for name in names]
        return index

    def get_samples_index(self,train_data,val_data):
        self.train_samples=train_data.shape[0]
        self.val_samples=val_data.shape[0]

    def save_param(self):
        with open(self.param_save_path,'wb') as file:
            pickle.dump(self, file)

    def load_params(self):
        with open(self.param_save_path,'rb') as file:
            loaded_data = pickle.load(file)
        return loaded_data
