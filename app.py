from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "Final_Random_Forest_Output" / "final_random_forest_model.joblib"

st.set_page_config(
    page_title="Karachi Traffic Accident Risk Prediction",
    page_icon="🚦",
    layout="wide",
)

AREAS = [
    "Clifton", "DHA", "Gulshan", "Karimabad", "Korangi", "Lyari",
    "Malir", "NIPA Road", "Nazimabad", "North Nazimabad", "PECHS",
    "Saddar", "Sohrab Goth",
]

# Approximate area-centre coordinates for visualization only.
AREA_COORDS = {
    "Clifton": (24.8138, 67.0305),
    "DHA": (24.8070, 67.0520),
    "Gulshan": (24.9200, 67.0900),
    "Karimabad": (24.9140, 67.0630),
    "Korangi": (24.8300, 67.1300),
    "Lyari": (24.8800, 67.0100),
    "Malir": (24.9000, 67.2000),
    "NIPA Road": (24.9310, 67.1130),
    "Nazimabad": (24.9300, 67.0350),
    "North Nazimabad": (24.9600, 67.0550),
    "PECHS": (24.8670, 67.0650),
    "Saddar": (24.8540, 67.0200),
    "Sohrab Goth": (24.9860, 67.0900),
}

METRICS = {
    "Accuracy": "68.67%",
    "Balanced accuracy": "69.44%",
    "Macro precision": "61.58%",
    "Macro recall": "69.44%",
    "Macro F1": "63.87%",
    "Weighted F1": "69.93%",
}

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def make_input(vehicle_count, road_type, road_condition, area, lanes,
               traffic_signal, weather, hour):
    return pd.DataFrame([{
        "Vehicle Count": int(vehicle_count),
        "Road Type": road_type,
        "Road Condition": road_condition,
        "Area": area,
        "Number of Lanes": int(lanes),
        "Traffic Signal": int(traffic_signal),
        "ML Weather": weather,
        "Hour Sin": float(np.sin(2 * np.pi * hour / 24)),
        "Hour Cos": float(np.cos(2 * np.pi * hour / 24)),
    }])

def risk_score(probabilities, classes):
    weights = {"Low": 0, "Medium": 1, "High": 2}
    return sum(float(p) * weights.get(str(c), 1) for p, c in zip(probabilities, classes))

st.title("Karachi Traffic Accident Risk Prediction")
st.caption("Machine-learning decision-support prototype using the selected Random Forest model.")

with st.sidebar:
    st.header("Model Information")
    st.write("**Algorithm:** Random Forest Classifier")
    st.write("**Class balancing:** class_weight='balanced'")
    st.write("**Training data:** Original dataset (no resampling)")
    st.write("**Evaluation:** Chronological holdout")
    st.divider()
    st.subheader("Test-set performance")
    for metric, value in METRICS.items():
        st.metric(metric, value)
    st.caption("Test set: 300 records. Accuracy and class scores are dataset-based evaluation results, not a guarantee of real-world safety.")

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

model = load_model()

left, right = st.columns([1.1, 1], gap="large")
with left:
    st.subheader("Prediction Inputs")
    with st.form("prediction_form"):
        area = st.selectbox("Karachi area", AREAS)
        hour = st.slider("Hour of day", 0, 23, 18, format="%d:00")
        vehicle_count = st.number_input("Estimated vehicle count", min_value=0, max_value=3000, value=450, step=10)
        road_type = st.selectbox("Road type", [
            "Highway", "Industrial Road", "Intersection", "Main Road",
            "Narrow Road", "Service Road"
        ])
        road_condition = st.selectbox("Road condition", [
            "Normal", "Congested", "Heavy Traffic", "Construction"
        ])
        lanes = st.number_input("Number of lanes", min_value=1, max_value=12, value=3)
        traffic_signal = st.selectbox("Traffic signal present", ["Yes", "No"])
        weather = st.selectbox("Weather", ["Dry", "Rainy"])
        submitted = st.form_submit_button("Predict Accident Risk", type="primary", use_container_width=True)

with right:
    st.subheader("Prediction Result")
    if submitted:
        X = make_input(
            vehicle_count, road_type, road_condition, area, lanes,
            1 if traffic_signal == "Yes" else 0, weather, hour
        )
        prediction = str(model.predict(X)[0])
        classes = list(model.classes_) if hasattr(model, "classes_") else list(model.named_steps["classifier"].classes_)
        probabilities = model.predict_proba(X)[0]
        score = float(probabilities[list(classes).index(prediction)])
        st.markdown(f"### Predicted risk: **{prediction}**")
        st.metric("Model score for predicted class", f"{score:.1%}")
        st.caption("This score is the model's predicted class score; it is not a calibrated real-world accident probability.")
        prob_df = pd.DataFrame({"Risk class": classes, "Model score": probabilities})
        st.bar_chart(prob_df.set_index("Risk class"))

        if prediction == "High":
            st.warning("High risk predicted. Consider checking alternative areas before travelling.")
            comparisons = []
            for candidate in AREAS:
                if candidate == area:
                    continue
                candidate_X = make_input(
                    vehicle_count, road_type, road_condition, candidate, lanes,
                    1 if traffic_signal == "Yes" else 0, weather, hour
                )
                candidate_probs = model.predict_proba(candidate_X)[0]
                candidate_class = str(model.predict(candidate_X)[0])
                comparisons.append({
                    "Area": candidate,
                    "Predicted risk": candidate_class,
                    "Risk score": risk_score(candidate_probs, classes),
                })
            alternatives = pd.DataFrame(comparisons).sort_values("Risk score")
            alternatives = alternatives[alternatives["Risk score"] < risk_score(probabilities, classes)]
            if alternatives.empty:
                st.info("The model did not identify another area with a lower predicted risk under the same inputs.")
            else:
                st.write("Areas with lower model-estimated risk under the same input conditions:")
                st.dataframe(alternatives[["Area", "Predicted risk"]].head(5), hide_index=True, use_container_width=True)
                map_rows = []
                for _, row in alternatives.head(5).iterrows():
                    coords = AREA_COORDS.get(row["Area"])
                    if coords:
                        map_rows.append({"latitude": coords[0], "longitude": coords[1]})
                selected_coords = AREA_COORDS.get(area)
                if selected_coords:
                    map_rows.append({"latitude": selected_coords[0], "longitude": selected_coords[1]})
                if map_rows:
                    st.map(pd.DataFrame(map_rows))
                st.caption("These are area-level comparisons, not turn-by-turn routes or validated safest-route recommendations.")
        else:
            st.success("No High-risk class was predicted for the selected inputs.")
    else:
        st.info("Enter the traffic and road conditions, then select **Predict Accident Risk**.")

st.divider()
st.subheader("About this prototype")
st.write(
    "The model was trained on 1,500 labelled records and evaluated on a chronological "
    "future holdout of 300 records. The original dataset already contained the Accident Risk "
    "labels; their derivation from independently verified crash records has not been established. "
    "Therefore, outputs should be interpreted as model predictions for research demonstration, "
    "not verified crash probabilities or guarantees of road safety."
)
