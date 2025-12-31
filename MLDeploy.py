import streamlit as st #streamlit for web app
import pickle #for loading the model
import pandas as pd #used for data manipulation
import sqlite3 #for database management


#unloading the trained models and label encoder========================================

#unloading the label encoder from the pkl file
with open("label_encoder.pkl", "rb") as file:
    le_loaded = pickle.load(file)

#unloading the random forest model from the pkl file
with open("random_forest_model.pkl", "rb") as file:
    rf_loaded = pickle.load(file)

#unloading the logistic regression model from the pkl file
with open("logistic_regression_model.pkl", "rb") as file:
    reg_loaded = pickle.load(file)




#creating the web app====================================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

conn = sqlite3.connect("iris_app.db", check_same_thread=False)
cursor = conn.cursor()
#table for storing user credentials (for demo purposes only, not secure)
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()



#table for storing predictions
cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    sepal_length REAL,
    sepal_width REAL,
    petal_length REAL,
    petal_width REAL,
    predicted_species TEXT,
    model_used TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()


def login_signup_page():
    st.title("🔐 Welcome")

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1: #login tab
        st.subheader("Login")

        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login"):
            cursor.execute(
                "SELECT * FROM users WHERE username=? AND password=?",
                (username, password)
            )
            user = cursor.fetchone()

            if user:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")

    with tab2: #signup tab
        st.subheader("Create New Account")

        new_username = st.text_input("New Username", key="signup_user")
        new_password = st.text_input("New Password", type="password", key="signup_pass")

        if st.button("Sign Up"):
            if not new_username or not new_password:
                st.warning("Please fill all fields")
            else:
                try:
                    cursor.execute(
                        "INSERT INTO users (username, password) VALUES (?, ?)",
                        (new_username, new_password)
                    )
                    conn.commit()
                    st.success("Account created! You can now login.")
                except sqlite3.IntegrityError:
                    st.error("Username already exists")


def prediction_page():
    st.title("🌸 Iris Flower Prediction App")

    st.subheader("Enter Flower Measurements")
    #getting user input for predictions in web app
    sepal_length = st.number_input("Sepal Length", value=5.1)
    sepal_width  = st.number_input("Sepal Width", value=3.5)
    petal_length = st.number_input("Petal Length", value=1.4)
    petal_width  = st.number_input("Petal Width", value=0.2)

    model_choice = st.selectbox(
    "Choose Model",
    ["Random Forest", "Logistic Regression"]
    )

    if st.button("Predict"):
        user_input = pd.DataFrame(
            [[sepal_length, sepal_width, petal_length, petal_width]],
            columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
        )

        if model_choice == "Random Forest":
            pred_encoded = rf_loaded.predict(user_input)
        else:
            pred_encoded = reg_loaded.predict(user_input)

        pred_label = le_loaded.inverse_transform(pred_encoded)

        #storing the prediction in the database
        cursor.execute("""
        INSERT INTO predictions (
            username,
            sepal_length,
            sepal_width,
            petal_length,
            petal_width,
            predicted_species,
            model_used
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            st.session_state.username,
            sepal_length,
            sepal_width,
            petal_length,
            petal_width,
            pred_label[0],
            model_choice
        ))

        conn.commit()


        #displaying the prediction result
        st.success(f"Predicted Species: {pred_label[0]}")
        st.markdown("### 🔎 Input Summary")

        st.info(f"📏 Sepal Length : {sepal_length:.2f}")
        st.info(f"📐 Sepal Width  : {sepal_width:.2f}")
        st.info(f"🌿 Petal Length : {petal_length:.2f}")
        st.info(f"🌸 Petal Width  : {petal_width:.2f}")

        #displaying prediction history
        st.markdown("### 📜 Your Prediction History")

        cursor.execute("""
        SELECT sepal_length, sepal_width, petal_length, petal_width,
            predicted_species, model_used, timestamp
        FROM predictions
        WHERE username = ?
        ORDER BY timestamp DESC
        """, (st.session_state.username,))

        rows = cursor.fetchall()

        if rows:
            df_history = pd.DataFrame(
                rows,
                columns=[
                    "Sepal Length", "Sepal Width",
                    "Petal Length", "Petal Width",
                    "Prediction", "Model", "Time"
                ]
            )
            st.dataframe(df_history)
        else:
            st.info("No predictions yet.")


if not st.session_state.logged_in:
    login_signup_page()
else:
    prediction_page()

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()
