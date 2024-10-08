from flask import Flask, render_template, request, redirect, url_for, send_file
from io import BytesIO
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_friend@003'
app.config['STATIC_FOLDER'] = 'static'

# Initialize Flask-Login
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Define the User class for Flask-Login
class User(UserMixin):
    pass

@login_manager.user_loader
def load_user(user_id):
    if user_id == '1':
        user = User()
        user.id = user_id
        return user
    return None

# Define the login form
class LoginForm(FlaskForm):
    username = StringField('Username:', validators=[DataRequired()])
    password = PasswordField('Password:', validators=[DataRequired()])
    submit = SubmitField('Login')

# Define the home route
@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    result_rf = None

    if request.method == 'POST':
        if current_user.is_authenticated:
            # Load the latest DataFrame from the Excel file
            df = pd.read_excel('sepsis data.xlsx')
            
            # Load or create "sepsis data op.xlsx" for new data tracking
            try:
                df_op = pd.read_excel('sepsis data op.xlsx')
            except FileNotFoundError:
                df_op = pd.DataFrame()  # Create an empty DataFrame if the file doesn't exist

            # Check for missing values in the 'WARD' column
            if df['WARD'].isnull().any():
                df = df.dropna(subset=['WARD'])

            # Define features (X) and target (y)
            X = df.drop(['Name', 'IP no.', 'WARD'], axis=1)
            y = df['WARD']

            # Initialize label encoders for each categorical column
            label_encoders = {}
            categorical_columns = ['GENDER', 'MATERNAL HEALTH', 'DELIVERY MODE']

            for column in categorical_columns:
                label_encoders[column] = LabelEncoder()
                X[column] = label_encoders[column].fit_transform(X[column])

            # Load the RandomForest model
            clf_rf = joblib.load('random_forest_model.joblib')

            # Get user input from the form
            user_data = {
                'Name': request.form['name'],
                'IP no.': int(request.form['ip_no']),
                'GESTATIONAL AGE': float(request.form['gestational_age']),
                'RR': float(request.form['resp_rate']),
                'HR': int(request.form['heart_rate']),
                'SPO2': int(request.form['oxygen_saturation']),
                'CRP': float(request.form['crp']),
                'MATERNAL HEALTH': request.form['maternal_health'],
                'DELIVERY MODE': request.form['mode_of_delivery'],
                'BIRTH WEIGHT': float(request.form['birth_weight']),
                'GENDER': request.form['gender'],
                'TLC': int(request.form['tlc']),
                'I/T RATIO': float(request.form['it_ratio']),
                'APGAR 1MIN': int(request.form['apgar_1min']),
                'APGAR 5MIN': int(request.form['apgar_5min']),
            }

            # Prepare user data for prediction
            user_data_df = pd.DataFrame(user_data, index=[0])

            # Label encode categorical variables for user input
            for column in categorical_columns:
                user_data_df[column] = label_encoders[column].transform([user_data[column]])[0]

            # Align user_data_df with X to ensure the same columns are present
            user_data_df = user_data_df.reindex(columns=X.columns, fill_value=0)

            # Make prediction
            prediction = clf_rf.predict(user_data_df)

            # Display the result
            result = f"Predicted ward: {prediction[0]}"

            # Add predicted WARD to the user data DataFrame
            user_data_df['WARD'] = prediction

            # Decode categorical variables for adding to the Excel sheet
            for column in categorical_columns:
                user_data_df[column] = label_encoders[column].inverse_transform(user_data_df[column])

            # Append user details to the existing DataFrame (sepsis data.xlsx)
            user_data_df['Name'] = request.form['name']
            user_data_df['IP no.'] = int(request.form['ip_no'])
            updated_df = pd.concat([df, user_data_df], ignore_index=True)

            # Save the updated DataFrame to 'sepsis data.xlsx'
            updated_df.to_excel('sepsis data.xlsx', index=False)

            # Append new user data to 'sepsis data op.xlsx' (only new entries)
            df_op = pd.concat([df_op, user_data_df], ignore_index=True)

            # Save only new data to 'sepsis data op.xlsx'
            df_op.to_excel('sepsis data op.xlsx', index=False)

            return render_template('index.html', result=result)

    return render_template('index.html', result=None)

# Define the login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()

    if form.validate_on_submit():
        if form.username.data == 'Doctor' and form.password.data == 'Doc@12345678$':
            user = User()
            user.id = '1'
            login_user(user)
            return redirect(url_for('index'))

    return render_template('login.html', form=form)

# Define the logout route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# Define a new route for downloading the "sepsis data op.xlsx" file
@app.route('/download_excel')
@login_required
def download_excel():
    # Load the latest DataFrame from "sepsis data op.xlsx"
    df_op = pd.read_excel('sepsis data op.xlsx')

    # Create a BytesIO object to store the Excel file
    excel_data = BytesIO()

    # Use pandas to_excel function to write the DataFrame to BytesIO as an Excel file
    df_op.to_excel(excel_data, index=False)

    # Set the file pointer to the beginning of the BytesIO object
    excel_data.seek(0)

    # Return the Excel file as a response with appropriate headers
    return send_file(excel_data, download_name='sepsis_data_op.xlsx', as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
