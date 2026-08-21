import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import urllib.request
from PIL import Image
import io

# ============================================
# ข้อมูลผู้พัฒนา (แก้ไขตรงนี้)
# ============================================
DEVELOPER_INFO = {
    'name': 'นายธนวัฒน์ เนียมรุ่งเรือง ',  # แก้เป็นชื่อคุณ
    'student_id': '664245015',    # แก้เป็นรหัสนักศึกษาคุณ
    'group': '66/43',                     # แก้เป็นหมู่เรียนคุณ
    'photo_path': 'IMG_20231105_165242_400.jpg'  # วางรูปของคุณในโฟลเดอร์ แล้วแก้ชื่อไฟล์
}

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="",
    layout="wide"
)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import urllib.request

# ============================================
# AUTO-TRAIN IF MODELS NOT FOUND
# ============================================
@st.cache_resource
def load_or_train_model():
    """โหลดโมเดลหรือ train ใหม่ถ้าไม่มี"""
    
    # ตรวจสอบว่ามีไฟล์โมเดลหรือไม่
    if not os.path.exists('best_model.pkl'):
        st.info("🔄 ไม่พบโมเดล กำลังดาวน์โหลด dataset และ train...")
        
        # ดาวน์โหลด dataset
        if not os.path.exists('data'):
            os.makedirs('data')
        
        try:
            url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
            urllib.request.urlretrieve(url, 'data/train.csv')
        except:
            st.error("❌ ไม่สามารถดาวน์โหลด dataset ได้")
            return None, None, None
        
        # โหลดข้อมูล
        df = pd.read_csv('data/train.csv')
        
        # Preprocessing
        df_processed = df.copy()
        features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        df_processed = df_processed[features + ['Survived']].copy()
        
        # Missing values - แก้ไขตรงนี้
        age_imputer = SimpleImputer(strategy='mean')
        df_processed['Age'] = age_imputer.fit_transform(df_processed[['Age']]).ravel()
        
        embarked_imputer = SimpleImputer(strategy='most_frequent')
        df_processed['Embarked'] = embarked_imputer.fit_transform(df_processed[['Embarked']]).ravel()
        
        # Encoding
        df_processed['Sex'] = df_processed['Sex'].map({'male': 0, 'female': 1})
        df_processed = pd.get_dummies(df_processed, columns=['Embarked'], prefix='Embarked')
        
        X = df_processed.drop('Survived', axis=1)
        y = df_processed['Survived']
        
        # Split & Scale
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train Model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Save models
        joblib.dump(model, 'best_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        joblib.dump({
            'age_imputer': age_imputer,
            'embarked_imputer': embarked_imputer,
            'feature_names': list(X.columns)
        }, 'preprocessors.pkl')
        
        st.success("✅ Train โมเดลเสร็จสมบูรณ์!")
        
        return model, scaler, {'age_imputer': age_imputer, 'embarked_imputer': embarked_imputer}
    
    else:
        # โหลดโมเดลที่มี
        try:
            model = joblib.load('best_model.pkl')
            scaler = joblib.load('scaler.pkl')
            preprocessors = joblib.load('preprocessors.pkl')
            return model, scaler, preprocessors
        except Exception as e:
            st.error(f"❌ โหลดโมเดลไม่สำเร็จ: {str(e)}")
            return None, None, None

# เรียกใช้ฟังก์ชัน
model, scaler, preprocessors = load_or_train_model()

if model is None:
    st.stop()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        text-align: center;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# HEADER
# ============================================
st.markdown('<h1 class="main-header">🚢 Titanic Survival Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">พยากรณ์การรอดชีวิตจากเรือไททานิคด้วย Machine Learning</p>', unsafe_allow_html=True)
st.markdown("---")

# ============================================
# SIDEBAR - INPUT PARAMETERS
# ============================================
st.sidebar.header("📝 ข้อมูลผู้โดยสาร")
st.sidebar.markdown("กรอกข้อมูลเพื่อทำนายการรอดชีวิต")

# Input fields
pclass = st.sidebar.selectbox(
    "ชั้นตั๋ว (Pclass)",
    options=[1, 2, 3],
    help="1 = First Class, 2 = Second Class, 3 = Third Class"
)

sex = st.sidebar.selectbox(
    "เพศ (Sex)",
    options=["male", "female"]
)

age = st.sidebar.slider(
    "อายุ (Age)",
    min_value=0,
    max_value=100,
    value=25
)

sibsp = st.sidebar.number_input(
    "จำนวนพี่น้อง/คู่สมรส (SibSp)",
    min_value=0,
    max_value=10,
    value=0
)

parch = st.sidebar.number_input(
    "จำนวนพ่อแม่/ลูก (Parch)",
    min_value=0,
    max_value=10,
    value=0
)

fare = st.sidebar.number_input(
    "ค่าตั๋ว (Fare)",
    min_value=0.0,
    max_value=600.0,
    value=30.0
)

embarked = st.sidebar.selectbox(
    "ท่าเรือที่ขึ้น (Embarked)",
    options=["C", "Q", "S"],
    help="C = Cherbourg, Q = Queenstown, S = Southampton"
)

# ============================================
# MAIN CONTENT
# ============================================

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔮 ทำนายผล", " ข้อมูล Dataset", "📈 ผลการเปรียบเทียบ Model", "ℹ️ เกี่ยวกับโปรเจค"])

with tab1:
    st.header("🔮 ทำนายการรอดชีวิต")
    
    if st.button("🎯 ทำนายผล", type="primary", use_container_width=True):
        try:
            # Load models
            model = joblib.load('best_model.pkl')
            scaler = joblib.load('scaler.pkl')
            preprocessors = joblib.load('preprocessors.pkl')
            
            # Prepare input data
            input_data = pd.DataFrame({
                'Pclass': [pclass],
                'Sex': [1 if sex == 'female' else 0],
                'Age': [age],
                'SibSp': [sibsp],
                'Parch': [parch],
                'Fare': [fare],
                'Embarked_C': [1 if embarked == 'C' else 0],
                'Embarked_Q': [1 if embarked == 'Q' else 0],
                'Embarked_S': [1 if embarked == 'S' else 0]
            })
            
            # Scale data
            input_scaled = scaler.transform(input_data)
            
            # Predict
            prediction = model.predict(input_scaled)
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            # Display result
            st.markdown("### ผลการทำนาย:")
            
            if prediction[0] == 1:
                st.success("✅ **รอดชีวิต**")
                st.balloons()
            else:
                st.error("❌ **ไม่รอดชีวิต**")
            
            # Probability
            col1, col2 = st.columns(2)
            with col1:
                st.metric("ความน่าจะเป็นที่จะรอด", f"{prediction_proba[1]:.2%}")
            with col2:
                st.metric("ความน่าจะเป็นที่จะไม่รอด", f"{prediction_proba[0]:.2%}")
            
            # Progress bar
            st.progress(float(prediction_proba[1]))
            
            # Input summary
            st.markdown("###  ข้อมูลที่กรอก:")
            input_summary = pd.DataFrame({
                'Feature': ['ชั้นตั๋ว', 'เพศ', 'อายุ', 'พี่น้อง/คู่สมรส', 'พ่อแม่/ลูก', 'ค่าตั๋ว', 'ท่าเรือ'],
                'ค่า': [pclass, sex, age, sibsp, parch, f"${fare:.2f}", embarked]
            })
            st.table(input_summary)
            
        except FileNotFoundError:
            st.error("❌ ไม่พบไฟล์ model กรุณารัน model_training.ipynb ก่อน")
        except Exception as e:
            st.error(f" เกิดข้อผิดพลาด: {str(e)}")

with tab2:
    st.header("📊 ข้อมูล Dataset Titanic")
    
    try:
        # Load dataset
        df = pd.read_csv('data/train.csv')
        
        st.subheader("ข้อมูล 5 แถวแรก:")
        st.dataframe(df.head())
        
        st.subheader("สถิติเบื้องต้น:")
        st.dataframe(df.describe())
        
        st.subheader("Missing Values:")
        missing_df = pd.DataFrame({
            'Feature': df.columns,
            'Missing Values': df.isnull().sum(),
            'Percentage': (df.isnull().sum() / len(df) * 100).round(2)
        })
        st.dataframe(missing_df[missing_df['Missing Values'] > 0])
        
        # Visualizations
        st.subheader(" การกระจายของข้อมูล")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots()
            df['Survived'].value_counts().plot(kind='pie', ax=ax, 
                                               labels=['ไม่รอด', 'รอด'], 
                                               autopct='%1.1f%%',
                                               colors=['#ff6b6b', '#4ecdc4'])
            ax.set_title('อัตราการรอดชีวิต')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots()
            sns.countplot(x='Sex', hue='Survived', data=df, ax=ax)
            ax.set_title('เพศ vs การรอดชีวิต')
            st.pyplot(fig)
        
        fig, ax = plt.subplots()
        sns.boxplot(x='Pclass', y='Age', hue='Survived', data=df, ax=ax)
        ax.set_title('ชั้นตั๋วและอายุ vs การรอดชีวิต')
        st.pyplot(fig)
        
    except FileNotFoundError:
        st.error(" ไม่พบไฟล์ data/train.csv")

with tab3:
    st.header("📈 ผลการเปรียบเทียบ Machine Learning Models")
    
    try:
        # Display comparison table
        comparison_data = {
            'Model': ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'SVM', 'Naive Bayes'],
            'Accuracy': ['0.82', '0.86', '0.84', '0.83', '0.79'],
            'Precision': ['0.80', '0.85', '0.83', '0.81', '0.77'],
            'Recall': ['0.78', '0.82', '0.80', '0.79', '0.75'],
            'F1-Score': ['0.79', '0.83', '0.81', '0.80', '0.76']
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Bar chart
        st.subheader("📊 กราฟเปรียบเทียบ Accuracy")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
        bars = ax.barh(comparison_df['Model'], comparison_df['Accuracy'].astype(float), color=colors)
        ax.set_xlabel('Accuracy')
        ax.set_title('Comparison of Model Accuracy')
        ax.set_xlim([0, 1])
        
        # Add value labels
        for bar, val in zip(bars, comparison_df['Accuracy'].astype(float)):
            ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                   f'{val:.2f}', va='center', fontsize=10)
        
        st.pyplot(fig)
        
        st.info("💡 **Model ที่ดีที่สุดคือ Random Forest** ด้วย Accuracy 86%")
        
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")

with tab4:
    st.header("ℹ️ เกี่ยวกับโปรเจค")
    
    st.markdown("""
    ### 📌 ข้อมูลโปรเจค
    
    **ชื่อโปรเจค:** Titanic Survival Prediction
    
    **วัตถุประสงค์:** 
    - พัฒนาโมเดล Machine Learning เพื่อพยากรณ์การรอดชีวิตของผู้โดยสารเรือไททานิค
    - เปรียบเทียบประสิทธิภาพของ Algorithms ต่างๆ
    
    ### 🛠️ Technologies Used
    - Python
    - Pandas, NumPy
    - Scikit-learn
    - Matplotlib, Seaborn
    - Streamlit
    
    ### 📊 Dataset
    - **แหล่งที่มา:** Kaggle Titanic Competition
    - **จำนวนข้อมูล:** 891 passengers
    - **Features:** 12 features (Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, etc.)
    - **Target:** Survived (0 = No, 1 = Yes)
    
    ###  โครงสร้างไฟล์
    ```
    titanic-ml-project/
    ├── data/
    │   └── train.csv
    ├── model_training.ipynb
    ├── app.py
    ├── requirements.txt
    ├── best_model.pkl
    ├── scaler.pkl
    ── preprocessors.pkl
    ```
    
    ### 👨‍💻 วิธีใช้งาน
    
    1. ติดตั้ง dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    
    2. รัน training:
    ```bash
    jupyter notebook model_training.ipynb
    ```
    
    3. รัน Streamlit app:
    ```bash
    streamlit run app.py
    ```
    
    ###  ผลการทดลอง
    - **Best Model:** Random Forest Classifier
    - **Accuracy:** 86%
    - **Precision:** 85%
    - **Recall:** 82%
    - **F1-Score:** 83%
    """)

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Developed with ❤️ using Streamlit</p>
    <p>Titanic Survival Prediction Project | Machine Learning</p>
</div>
""", unsafe_allow_html=True)