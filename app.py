{\rtf1\ansi\ansicpg1252\cocoartf2868
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import streamlit as st\
import pandas as pd\
import numpy as np\
from sklearn.ensemble import RandomForestClassifier\
from sklearn.preprocessing import OneHotEncoder\
from sklearn.compose import ColumnTransformer\
from sklearn.pipeline import Pipeline\
from sklearn.model_selection import train_test_split\
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix\
\
# --------------------------------------------------\
# Page Setup\
# --------------------------------------------------\
\
st.set_page_config(\
    page_title="Dog Behavior Interpreter",\
    page_icon="\uc0\u55357 \u56374 ",\
    layout="centered"\
)\
\
st.title("\uc0\u55357 \u56374  Dog Behavior Interpreter")\
st.caption("AI prototype for interpreting short-duration dog behavior using image + context inputs.")\
\
# --------------------------------------------------\
# Synthetic Data Creation\
# --------------------------------------------------\
\
def generate_synthetic_data(n=500, random_state=42):\
    np.random.seed(random_state)\
\
    behaviors = ["panting", "pacing", "whining", "hiding", "toy-seeking", "resting"]\
    activities = ["none", "low", "high"]\
    environments = ["indoor", "outdoor", "warm", "noisy"]\
    durations = ["short", "medium", "long"]\
    assumptions = ["anxiety", "boredom", "overstimulation", "recovery", "unsure"]\
\
    rows = []\
\
    for _ in range(n):\
        behavior = np.random.choice(behaviors)\
        activity = np.random.choice(activities)\
        environment = np.random.choice(environments)\
        duration = np.random.choice(durations)\
        assumption = np.random.choice(assumptions)\
\
        # Heuristic labeling logic for prototype training data\
        if behavior == "panting" and activity == "high":\
            label = "recovery"\
        elif behavior == "panting" and environment == "warm":\
            label = "recovery"\
        elif behavior in ["pacing", "whining", "hiding"] and environment == "noisy":\
            label = "anxiety"\
        elif behavior == "pacing" and duration == "long" and activity == "none":\
            label = "anxiety"\
        elif behavior == "toy-seeking" and activity in ["none", "low"]:\
            label = "boredom"\
        elif behavior in ["panting", "pacing"] and activity == "high" and duration in ["short", "medium"]:\
            label = "overstimulation"\
        elif behavior == "resting":\
            label = "neutral"\
        else:\
            label = np.random.choice(\
                ["anxiety", "boredom", "overstimulation", "recovery", "neutral"],\
                p=[0.25, 0.25, 0.2, 0.2, 0.1]\
            )\
\
        rows.append(\{\
            "observed_behavior": behavior,\
            "recent_activity": activity,\
            "environment": environment,\
            "duration": duration,\
            "owner_assumption": assumption,\
            "label": label\
        \})\
\
    return pd.DataFrame(rows)\
\
\
df = generate_synthetic_data()\
\
# --------------------------------------------------\
# Model Training\
# --------------------------------------------------\
\
features = ["observed_behavior", "recent_activity", "environment", "duration", "owner_assumption"]\
target = "label"\
\
X = df[features]\
y = df[target]\
\
categorical_features = features\
\
preprocessor = ColumnTransformer(\
    transformers=[\
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)\
    ]\
)\
\
model = RandomForestClassifier(\
    n_estimators=150,\
    random_state=42,\
    class_weight="balanced"\
)\
\
pipeline = Pipeline(steps=[\
    ("preprocessor", preprocessor),\
    ("model", model)\
])\
\
X_train, X_test, y_train, y_test = train_test_split(\
    X, y, test_size=0.25, random_state=42, stratify=y\
)\
\
pipeline.fit(X_train, y_train)\
\
y_pred = pipeline.predict(X_test)\
accuracy = accuracy_score(y_test, y_pred)\
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)\
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)\
\
# --------------------------------------------------\
# Recommendation Logic\
# --------------------------------------------------\
\
recommendations = \{\
    "recovery": "Provide water and allow rest. Reassess in 10\'9615 minutes.",\
    "anxiety": "Reduce stimuli, create a calm environment, and observe for additional stress signals.",\
    "boredom": "Offer enrichment such as a puzzle toy, scent game, or short training session.",\
    "overstimulation": "Pause activity, reduce excitement, and give your dog a quiet break.",\
    "neutral": "Continue observing. No immediate intervention appears necessary."\
\}\
\
explanations = \{\
    "recovery": "The behavior appears consistent with recovery, especially when paired with recent activity or a warm environment.",\
    "anxiety": "The behavior may reflect stress or anxiety, especially when paired with pacing, whining, hiding, or a noisy environment.",\
    "boredom": "The behavior may reflect unmet stimulation needs, especially when activity has been low.",\
    "overstimulation": "The behavior may reflect excess excitement or arousal, especially after high activity.",\
    "neutral": "The inputs do not strongly indicate distress or unmet needs."\
\}\
\
# --------------------------------------------------\
# Sidebar Model Metrics\
# --------------------------------------------------\
\
with st.sidebar:\
    st.header("Prototype Model Metrics")\
    st.metric("Accuracy", f"\{accuracy:.2%\}")\
    st.metric("Weighted Precision", f"\{precision:.2%\}")\
    st.metric("Weighted Recall", f"\{recall:.2%\}")\
\
    with st.expander("About this prototype"):\
        st.write(\
            """\
            This prototype uses synthetic training data and a Random Forest classifier.\
            The image upload is included to simulate the intended user journey, but this version\
            primarily uses structured context inputs for prediction.\
            """\
        )\
\
# --------------------------------------------------\
# User Interface\
# --------------------------------------------------\
\
st.subheader("1. Upload Dog Image")\
\
uploaded_image = st.file_uploader(\
    "Upload an image of your dog",\
    type=["jpg", "jpeg", "png"]\
)\
\
if uploaded_image:\
    st.image(uploaded_image, caption="Uploaded dog image", use_container_width=True)\
\
st.subheader("2. Add Context")\
\
observed_behavior = st.selectbox(\
    "What behavior do you observe?",\
    ["panting", "pacing", "whining", "hiding", "toy-seeking", "resting"]\
)\
\
recent_activity = st.selectbox(\
    "Recent activity level",\
    ["none", "low", "high"]\
)\
\
environment = st.selectbox(\
    "Environment",\
    ["indoor", "outdoor", "warm", "noisy"]\
)\
\
duration = st.selectbox(\
    "How long has this been happening?",\
    ["short", "medium", "long"]\
)\
\
owner_assumption = st.selectbox(\
    "What do you think is happening? Optional",\
    ["unsure", "anxiety", "boredom", "overstimulation", "recovery"]\
)\
\
st.subheader("3. Generate AI Interpretation")\
\
if st.button("Analyze Behavior", type="primary"):\
    input_df = pd.DataFrame([\{\
        "observed_behavior": observed_behavior,\
        "recent_activity": recent_activity,\
        "environment": environment,\
        "duration": duration,\
        "owner_assumption": owner_assumption\
    \}])\
\
    prediction = pipeline.predict(input_df)[0]\
    probabilities = pipeline.predict_proba(input_df)[0]\
    classes = pipeline.classes_\
\
    prob_df = pd.DataFrame(\{\
        "Behavior State": classes,\
        "Probability": probabilities\
    \}).sort_values("Probability", ascending=False)\
\
    top_probability = prob_df.iloc[0]["Probability"]\
    secondary_state = prob_df.iloc[1]["Behavior State"]\
    secondary_probability = prob_df.iloc[1]["Probability"]\
\
    st.success("AI Interpretation Complete")\
\
    st.markdown("## \uc0\u55358 \u56800  Behavioral Interpretation")\
    st.write(f"**Likely state:** \{prediction.title()\}")\
    st.write(f"**Confidence:** \{top_probability:.0%\}")\
    st.write(f"**Secondary possibility:** \{secondary_state.title()\} (\{secondary_probability:.0%\})")\
\
    st.markdown("## Why the system predicted this")\
    st.write(explanations[prediction])\
\
    st.markdown("## Recommended Action")\
    st.info(recommendations[prediction])\
\
    if owner_assumption != "unsure":\
        st.markdown("## User Assumption Check")\
        if owner_assumption == prediction:\
            st.write("The AI interpretation aligns with the owner\'92s initial assumption.")\
        else:\
            st.write(\
                f"The owner initially assumed **\{owner_assumption\}**, "\
                f"but the model predicted **\{prediction\}**. "\
                "This case may represent a potential interpretation correction."\
            )\
\
    st.markdown("## Probability Distribution")\
    st.bar_chart(prob_df.set_index("Behavior State"))\
\
    st.subheader("4. Feedback Loop")\
\
    col1, col2 = st.columns(2)\
\
    with col1:\
        accurate = st.radio(\
            "Was this interpretation accurate?",\
            ["Yes", "No", "Unsure"],\
            horizontal=True\
        )\
\
    with col2:\
        improved = st.radio(\
            "Did the behavior improve after action?",\
            ["Yes", "No", "Not yet"],\
            horizontal=True\
        )\
\
    if st.button("Submit Feedback"):\
        feedback_row = \{\
            "observed_behavior": observed_behavior,\
            "recent_activity": recent_activity,\
            "environment": environment,\
            "duration": duration,\
            "owner_assumption": owner_assumption,\
            "model_prediction": prediction,\
            "confidence": round(float(top_probability), 3),\
            "accurate_feedback": accurate,\
            "behavior_improved": improved\
        \}\
\
        feedback_df = pd.DataFrame([feedback_row])\
\
        try:\
            existing_feedback = pd.read_csv("feedback_log.csv")\
            updated_feedback = pd.concat([existing_feedback, feedback_df], ignore_index=True)\
        except FileNotFoundError:\
            updated_feedback = feedback_df\
\
        updated_feedback.to_csv("feedback_log.csv", index=False)\
\
        st.success("Feedback saved for future model improvement.")\
        st.dataframe(feedback_df)\
\
else:\
    st.info("Upload an image if available, enter context, then click Analyze Behavior.")}