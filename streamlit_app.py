import streamlit as st # type: ignore
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

model = joblib.load('voting_model.pkl')
df = pd.read_csv('Salary_Data.csv')

st.set_page_config(page_title="Salary Predictor", layout="centered")
st.title("Salary Prediction App")
st.markdown("Enter employee details to predict salary using a machine learning model.")

st.sidebar.header("Insights")

if st.sidebar.checkbox("Salary Distribution"):
    fig, ax = plt.subplots()
    sns.histplot(df['Salary'], kde=True, ax=ax, color="skyblue")
    ax.set_title("Salary Distribution")
    st.sidebar.pyplot(fig)

if st.sidebar.checkbox("Experience vs Salary"):
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="Years of Experience", y="Salary", hue="Education Level", ax=ax)
    ax.set_title("Experience vs Salary")
    st.sidebar.pyplot(fig)

if st.sidebar.checkbox("Top 10 Highest Salaries"):
    top_10 = df.sort_values(by="Salary", ascending=False).head(10)
    fig, ax = plt.subplots()
    sns.barplot(data=top_10, x="Salary", y="Job Title", hue="Gender", ax=ax)
    ax.set_title("Top 10 Highest Paid Employees")
    st.sidebar.pyplot(fig)

if st.sidebar.checkbox("Average Salary by Job Title"):
    avg_salary_job = df.groupby("Job Title")["Salary"].mean().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots()
    avg_salary_job.plot(kind="barh", color="teal", ax=ax)
    ax.set_title("Average Salary by Job Title")
    st.sidebar.pyplot(fig)

if st.sidebar.checkbox("Average Salary by Education Level"):
    avg_salary_edu = df.groupby("Education Level")["Salary"].mean()
    fig, ax = plt.subplots()
    avg_salary_edu.plot(kind="bar", color="salmon", ax=ax)
    ax.set_ylabel("Average Salary")
    ax.set_title("Average Salary by Education Level")
    st.sidebar.pyplot(fig)

if st.sidebar.checkbox("Gender-wise Salary Comparison"):
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="Gender", y="Salary", ax=ax, palette="pastel")
    ax.set_title("Gender-wise Salary Distribution")
    st.sidebar.pyplot(fig)

if st.sidebar.checkbox("Salary by Age Group"):
    bins = [18, 25, 30, 35, 40, 50, 65]
    labels = ["18–25", "26–30", "31–35", "36–40", "41–50", "51–65"]
    df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels)
    avg_age_salary = df.groupby("Age Group")["Salary"].mean()
    fig, ax = plt.subplots()
    avg_age_salary.plot(kind="bar", color="purple", ax=ax)
    ax.set_ylabel("Average Salary")
    ax.set_title("Average Salary by Age Group")
    st.sidebar.pyplot(fig)

if st.sidebar.checkbox("Salary by Remote Work Status"):
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="Remote Work Status", y="Salary", ax=ax, palette="coolwarm")
    ax.set_title("Remote Work vs Salary")
    st.sidebar.pyplot(fig)

if st.sidebar.checkbox("Average Salary by Industry"):
    avg_salary_industry = df.groupby("Industry")["Salary"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots()
    avg_salary_industry.plot(kind="bar", ax=ax, color="orange")
    ax.set_ylabel("Average Salary")
    ax.set_title("Average Salary by Industry")
    st.sidebar.pyplot(fig)

if st.sidebar.checkbox("Salary by Company Size"):
    avg_salary_by_company = df.groupby("Company Size")["Salary"].mean()
    fig, ax = plt.subplots()
    ax.pie(avg_salary_by_company, labels=avg_salary_by_company.index, autopct='%1.1f%%', startangle=90)
    ax.set_title("Salary Distribution by Company Size")
    st.sidebar.pyplot(fig)

if st.sidebar.checkbox("Salary by City/Region"):
    top_cities = df['City or Region'].value_counts().head(10).index.tolist()
    city_df = df[df['City or Region'].isin(top_cities)]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=city_df, x="City or Region", y="Salary", ax=ax)
    ax.set_title("City-wise Salary Distribution")
    plt.xticks(rotation=45)
    st.sidebar.pyplot(fig)

st.subheader("Employee Details")
age = st.slider("Age", 18, 65, 25)
gender = st.selectbox("Gender", df["Gender"].unique())
education = st.selectbox("Education Level", df["Education Level"].unique())
job_title = st.text_input("Job Title", "Software Engineer")
experience = st.slider("Years of Experience", 0, 40, 2)
remote_status = st.selectbox("Remote Work Status", df["Remote Work Status"].unique())
industry = st.selectbox("Industry", df["Industry"].unique())
company_size = st.selectbox("Company Size", df["Company Size"].unique())
city = st.selectbox("City or Region", df["City or Region"].unique())

if st.button("Predict Salary"):
    input_df = pd.DataFrame([[age, gender, education, job_title, experience,
                              remote_status, industry, company_size, city]],
                            columns=["Age", "Gender", "Education Level", "Job Title", "Years of Experience",
                                     "Remote Work Status", "Industry", "Company Size", "City or Region"])
    
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted Salary: ₹{int(prediction):,}")
