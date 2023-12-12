import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# **Load the Dataset**

# Load the Calories burnt dataset
df=pd.read_csv(r"C:\Users\91903\Desktop\project\calories_data.csv")
# df

# **Data Preprocessing**

#Check the missing values
df.isna().sum()

# **Drop Columns**

#Dropping column(User_ID) since it does not contribute to r2 score
df.drop(['User_ID'],axis=1,inplace=True)
# df

# **Converting the text data to numerical values**

df['Gender'].value_counts()

# Label Encoding is a technique used to convert categorical variables to numerical values
le=LabelEncoder()
df['Gender']=le.fit_transform(df['Gender'])
# df

# Save the LabelEncoder object to a file using joblib
joblib.dump(le,r"C:\Users\91903\Desktop\project\label_encoder_gender.pkl")
# **Separating features and Target**

# Assuming you have a 'Target' column that you want to predict
X = df.iloc[:,:-1]     # Features(Independent variable)
# X

# Target variable
y= df.iloc[:,-1]
# y

# **Splitting the data into training data and Test data**

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# **Hyperparameter Tuning:**

# hyperparameter tuning also know as hyperparameter optimization,is the process of finding the best set of hyperparameters for a machine learning model to achieve optimal performance

xgb1=XGBRegressor()
params={'n_estimators': [100, 200, 300],'max_depth': [3, 4, 5]}
reg=GridSearchCV(xgb1,params,cv=10)
reg.fit(X_train,y_train)

#taking best parameters
print(reg.best_params_)

xgb_new=XGBRegressor(n_estimators=300,max_depth=4)
xgb_new.fit(X_train,y_train)
y_pred=xgb_new.predict(X_test)

#save the trained model
joblib.dump(xgb_new,"model.joblib")

st.title("Calories Prediction App")

st.write("Enter the following details to predict the Calorie:")

# Load the trained model
loaded_model = joblib.load(r"C:\Users\91903\Desktop\project\model.joblib")

# During prediction, load the LabelEncoders
le=joblib.load(r"C:\Users\91903\Desktop\project\label_encoder_gender.pkl")

def set_bg_hack_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(https://img.freepik.com/free-photo/confident-sportswoman-with-dumbbell-dark_23-2147752923.jpg?w=1060&t=st=1700150386~exp=1700150986~hmac=312ce91542a5af5054fdae105fb1c4d730515a95e304a3915ea76a0c379acd97);
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
set_bg_hack_url()
#
#
 # Create input fields for user to enter data
gender = st.selectbox("Gender", ["male", "female"])
age = st.number_input("Age", key="age_input",step=1)
height = st.number_input("Height (cm)", key="height_input",step=1)
weight = st.number_input("Weight (kg)", key="weight_input",step=1)
duration = st.number_input("Exercise Duration (minutes)", key="duration_input",step=1)
heart_rate = st.number_input("Heart Rate (bpm)", key="heart_rate_input",step=1)
body_temp = st.number_input("Body Temperature (°C)", key="body_temp_input")

# Process input data for prediction
if st.button("Predict Calories"):
    if gender not in le.classes_:
        st.error("Invalid Gender. Please choose a valid option.")
    else:
        gender_encoded =le.transform([gender])[0]
    # Create a DataFrame with the input data
    user_input_data = pd.DataFrame({
        'Gender': [gender_encoded],
        'Age': [age],
        'Height': [height],
        'Weight': [weight],
        'Duration': [duration],
        'Heart_Rate': [heart_rate],
        'Body_Temp': [body_temp]
    })

    # Now, make the prediction
    predicted_calories = loaded_model.predict(user_input_data)

    st.write(f"Predicted Calories: {predicted_calories[0]}")
