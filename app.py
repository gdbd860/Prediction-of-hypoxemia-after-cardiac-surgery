#streamlit run app.py
import streamlit as st
import pandas as pd
import pickle
import numpy as np

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

def app():
    # 加载模型和参数
    with open("./XGBClassifier_ca.pkl", 'rb') as file:
        model = pickle.load(file)
    with open('./param.pkl', 'rb') as file:
        params = pickle.load(file)
    data = pd.read_excel(params.data_process_save_path, params.train_sheet)
    X = data.drop(columns=[params.y[0]])
    feature_names = np.array(X.columns, dtype=str)

    #页面布局
    st.set_page_config(layout='wide')
    st.title("Hypoxemia Prediction")

    #侧边栏
    st.sidebar.title("About This App")
    st.sidebar.markdown("""
    This application is designed for hypoxemia risk prediction.
    
    - Supports batch data upload (CSV/Excel)
    - Supports single sample online prediction
    - Downloadable prediction results
    
    ---

    **How to use:**
    1. Select the data input mode
    2. Upload your data file or enter feature values
    3. Click the prediction button
    4. View and download the results
    
    **Feature introduction**
    """)
    st.sidebar.info("For questions, please contact:\n498424860@qq.com")

    #输入数据预测
    input_mode = st.radio("Choose data input method:", ("Batch upload (CSV/Excel)", "Single sample"))
    if input_mode == "Batch upload (CSV/Excel)":
        uploaded_file = st.file_uploader("Upload your data file (CSV or Excel format)", type=["csv", "xlsx", "xls"])
        if uploaded_file is not None:
            # Determine file type and read
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                data = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format, please upload a CSV or Excel file.")
                st.stop()
            st.write("Preview of the uploaded data:")
            st.dataframe(data.head())
            if st.button("Start batch prediction"):
                predictions = model.predict(data)
                data['Prediction Result'] = predictions
                st.write("Prediction results:")
                st.dataframe(data)
                csv = data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download prediction results",
                    data=csv,
                    file_name='prediction_results.csv',
                    mime='text/csv'
                )
        else:
            st.info("Please upload a CSV or Excel format data file.")

    elif input_mode == "Single sample":
        st.write("Please enter the feature values for a single sample:")
        input_data = []
        for feature in feature_names:
            value = st.text_input(f"{feature}:")
            input_data.append(value)
        if st.button("Start single sample prediction"):
            try:
                input_data = [float(x) for x in input_data]
                sample_df = pd.DataFrame([input_data], columns=feature_names)
                prediction = model.predict(sample_df)[0]
                st.success(f"Prediction result: {prediction}")
            except Exception as e:
                st.error(f"Input error, please check! Error message: {e}")

if __name__ == "__main__":
    app()





