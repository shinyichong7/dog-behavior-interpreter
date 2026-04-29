import os, json, base64
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

st.set_page_config(page_title="Dog Behavior Interpreter", page_icon="🐶", layout="wide")

# ==================================================
# STYLE
# ==================================================

st.markdown("""
<style>
.main .block-container {
    padding-top: 1.5rem;
    max-width: 1180px;
    background-color: #FAF7F2;
}

html, body, [class*="css"] {
    color: #2F2A25;
}

.disclaimer-card {
    padding: 1rem;
    border-radius: 18px;
    border: 1px solid #D9A441;
    background-color: #FFF4D8;
    color: #5C3B00;
    margin-bottom: 1rem;
    font-size: 0.92rem;
}

.stepper {
    display: flex;
    gap: 8px;
    margin: 12px 0 20px 0;
    flex-wrap: wrap;
}

.step-active {
    padding: 0.55rem 0.85rem;
    border-radius: 999px;
    background: #7A9E7E;
    color: white;
    font-weight: 700;
    font-size: 0.9rem;
}

.step-complete {
    padding: 0.55rem 0.85rem;
    border-radius: 999px;
    background: #DCEBD8;
    color: #355E3B;
    font-weight: 700;
    font-size: 0.9rem;
}

.step-inactive {
    padding: 0.55rem 0.85rem;
    border-radius: 999px;
    background: #EFE8DD;
    color: #7A6A58;
    font-weight: 600;
    font-size: 0.9rem;
}

.result-card {
    padding: 1.4rem;
    border-radius: 22px;
    border: 1px solid #BFD8BD;
    background-color: #EEF6EA;
    margin-bottom: 1rem;
    box-shadow: 0 2px 10px rgba(76, 97, 70, 0.08);
}

.action-hero {
    padding: 1.4rem;
    border-radius: 22px;
    border: 1px solid #B7D4D1;
    background-color: #EAF5F3;
    margin-bottom: 1rem;
    box-shadow: 0 2px 10px rgba(65, 99, 96, 0.08);
}

.quality-card, .kpi, .section-card {
    padding: 1.2rem;
    border-radius: 20px;
    border: 1px solid #D8C8B2;
    background-color: #FFFCF7;
    margin-bottom: 1rem;
    box-shadow: 0 2px 10px rgba(82, 67, 50, 0.05);
}

.warning-card {
    padding: 1rem;
    border-radius: 18px;
    border: 1px solid #D9A441;
    background-color: #FFF4D8;
    margin-bottom: 1rem;
}

.escalation-card {
    padding: 1rem;
    border-radius: 18px;
    border: 1px solid #D9A19A;
    background-color: #FBEDEA;
    margin-bottom: 0.75rem;
}

.action-box {
    padding: 1rem;
    border-radius: 18px;
    border: 1px solid #B7D4D1;
    background-color: #EFF8F6;
    margin-bottom: 0.75rem;
}

.education-box {
    padding: 1rem;
    border-radius: 18px;
    border: 1px solid #CFC3A8;
    background-color: #F8F1E5;
    margin-bottom: 0.75rem;
}

.ai-box {
    padding: 1rem;
    border-radius: 18px;
    border: 1px solid #AFC9C3;
    background-color: #EAF5F3;
    margin-bottom: 0.75rem;
}

button[kind="primary"] {
    background-color: #7A9E7E !important;
    border-color: #7A9E7E !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 999px;
    background-color: #EFE8DD;
    padding: 8px 14px;
}

.stTabs [aria-selected="true"] {
    background-color: #DCEBD8;
    color: #355E3B;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# SESSION STATE
# ==================================================

def set_phase(n):
    st.session_state.phase = n

if "phase" not in st.session_state:
    st.session_state.phase = 1

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "feedback_done" not in st.session_state:
    st.session_state.feedback_done = False

defaults = {
    "visual_mouth": "unknown",
    "visual_posture": "unknown",
    "visual_ears": "unknown",
    "visual_tail": "unknown",
    "visual_eyes": "unknown",
    "visual_hiding": "unknown",
    "image_available": "no",
    "image_ai_confidence": None,
    "image_ai_reason": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

CLASSES = ["anxiety", "boredom", "overstimulation", "recovery", "neutral", "needs observation"]

# ==================================================
# HYBRID PREDICTION LOGIC
# ==================================================

def evidence_scores(inputs):
    scores = {c: 0.2 for c in CLASSES}

    behavior = inputs["behavior"]
    activity = inputs["activity"]
    environment = inputs["environment"]
    duration = inputs["duration"]
    age = inputs["age_group"]
    energy = inputs["energy"]
    sensitivity = inputs["sensitivity"]

    mouth = inputs["visual_mouth"]
    posture = inputs["visual_posture"]
    ears = inputs["visual_ears"]
    tail = inputs["visual_tail"]
    eyes = inputs["visual_eyes"]
    hiding = inputs["visual_hiding"]

    stress_cues = 0
    relaxed_cues = 0

    if posture in ["tense", "crouched"]:
        scores["anxiety"] += 2.2
        scores["needs observation"] += 0.8
        stress_cues += 1

    if ears == "back/pinned":
        scores["anxiety"] += 1.8
        stress_cues += 1

    if tail == "tucked":
        scores["anxiety"] += 2.0
        scores["needs observation"] += 0.6
        stress_cues += 1

    if eyes == "wide-eyed / whale eye":
        scores["anxiety"] += 2.4
        stress_cues += 1

    if hiding == "yes":
        scores["anxiety"] += 2.3
        scores["needs observation"] += 0.7
        stress_cues += 1

    if posture == "relaxed":
        scores["neutral"] += 1.7
        scores["recovery"] += 0.8
        relaxed_cues += 1

    if ears == "neutral":
        scores["neutral"] += 0.8
        relaxed_cues += 1

    if tail == "relaxed":
        scores["neutral"] += 1.0
        relaxed_cues += 1

    if eyes == "relaxed":
        scores["neutral"] += 1.0
        relaxed_cues += 1

    if mouth == "closed mouth":
        scores["neutral"] += 0.9
        scores["anxiety"] -= 0.4

    if mouth == "open mouth / panting":
        scores["recovery"] += 0.9
        scores["overstimulation"] += 0.6

    if behavior == "panting":
        scores["recovery"] += 1.0
        scores["overstimulation"] += 0.7
        if activity == "none" and duration == "long":
            scores["needs observation"] += 1.2

    if behavior == "pacing":
        scores["anxiety"] += 1.1
        scores["overstimulation"] += 0.8
        scores["boredom"] += 0.7

    if behavior == "whining":
        scores["anxiety"] += 1.4
        scores["needs observation"] += 0.5

    if behavior == "hiding":
        scores["anxiety"] += 2.0
        scores["needs observation"] += 0.7

    if behavior == "toy-seeking":
        scores["boredom"] += 2.2
        scores["neutral"] += 0.5

    if behavior == "resting":
        scores["neutral"] += 2.2
        scores["anxiety"] -= 0.7
        scores["overstimulation"] -= 0.5

    if activity == "high":
        scores["recovery"] += 1.3
        scores["overstimulation"] += 1.3
        if stress_cues >= 2:
            scores["anxiety"] += 0.8

    if activity == "low":
        scores["neutral"] += 0.4
        scores["boredom"] += 0.4

    if activity == "none":
        scores["boredom"] += 0.8
        scores["recovery"] -= 0.8
        if behavior in ["pacing", "toy-seeking"]:
            scores["boredom"] += 1.2

    if environment == "warm":
        scores["recovery"] += 1.2
        if mouth == "open mouth / panting" or behavior == "panting":
            scores["recovery"] += 1.0

    if environment == "noisy":
        scores["anxiety"] += 1.4
        scores["overstimulation"] += 0.5

    if environment in ["indoor", "outdoor"] and stress_cues == 0:
        scores["neutral"] += 0.5

    if duration == "short":
        scores["neutral"] += 0.5
        scores["recovery"] += 0.4
        scores["needs observation"] -= 0.6

    if duration == "medium":
        scores["overstimulation"] += 0.3
        scores["recovery"] += 0.3

    if duration == "long":
        scores["anxiety"] += 0.8
        scores["needs observation"] += 1.4
        scores["neutral"] -= 0.7

    if sensitivity == "high":
        scores["anxiety"] += 0.9
    elif sensitivity == "low":
        scores["anxiety"] -= 0.5
        scores["neutral"] += 0.3

    if energy == "high":
        scores["boredom"] += 0.5
        scores["overstimulation"] += 0.5
    elif energy == "low":
        scores["neutral"] += 0.4
        scores["overstimulation"] -= 0.5

    if age == "senior" and duration == "long":
        scores["needs observation"] += 1.0

    if relaxed_cues >= 3 and stress_cues == 0:
        scores["neutral"] += 2.0
        scores["anxiety"] -= 1.3
        scores["overstimulation"] -= 0.6

    if stress_cues >= 3:
        scores["anxiety"] += 2.0
        scores["neutral"] -= 1.2

    if stress_cues == 0 and behavior == "resting" and mouth == "closed mouth":
        scores["neutral"] += 1.8

    for key in scores:
        scores[key] = max(0.01, scores[key])

    total = sum(scores.values())
    probs = {k: v / total for k, v in scores.items()}
    return probs, scores


def hybrid_predict(inputs, ml_model):
    input_df = pd.DataFrame([inputs])

    ml_classes = list(ml_model.named_steps["rf"].classes_)
    ml_raw_probs = ml_model.predict_proba(input_df)[0]
    ml_probs = {c: 0.0 for c in CLASSES}

    for c, p in zip(ml_classes, ml_raw_probs):
        ml_probs[c] = float(p)

    evidence_probs, raw_scores = evidence_scores(inputs)

    final_probs = {}
    for c in CLASSES:
        final_probs[c] = (0.78 * evidence_probs[c]) + (0.22 * ml_probs.get(c, 0))

    total = sum(final_probs.values())
    final_probs = {k: v / total for k, v in final_probs.items()}

    prob_df = pd.DataFrame({
        "Behavior State": list(final_probs.keys()),
        "Probability": list(final_probs.values())
    }).sort_values("Probability", ascending=False)

    pred = prob_df.iloc[0]["Behavior State"]
    return pred, prob_df, evidence_probs, ml_probs, raw_scores


# ==================================================
# DATA + MODEL
# ==================================================

@st.cache_data
def generate_data(n=2200):
    np.random.seed(42)

    rows = []
    values = {
        "behavior": ["panting", "pacing", "whining", "hiding", "toy-seeking", "resting"],
        "activity": ["none", "low", "high"],
        "environment": ["indoor", "outdoor", "warm", "noisy"],
        "duration": ["short", "medium", "long"],
        "assumption": ["anxiety", "boredom", "overstimulation", "recovery", "unsure"],
        "age_group": ["puppy", "adult", "senior"],
        "size": ["small", "medium", "large"],
        "energy": ["low", "moderate", "high"],
        "sensitivity": ["low", "moderate", "high"],
        "image_available": ["yes", "no"],
        "visual_mouth": ["unknown", "closed mouth", "open mouth / panting"],
        "visual_posture": ["unknown", "relaxed", "alert", "tense", "crouched"],
        "visual_ears": ["unknown", "neutral", "back/pinned"],
        "visual_tail": ["unknown", "relaxed", "tucked", "high"],
        "visual_eyes": ["unknown", "relaxed", "wide-eyed / whale eye"],
        "visual_hiding": ["unknown", "yes", "no"]
    }

    for _ in range(n):
        row = {k: np.random.choice(v) for k, v in values.items()}

        probs, _ = evidence_scores(row)
        label = max(probs, key=probs.get)

        rows.append({**row, "label": label})

    return pd.DataFrame(rows)


@st.cache_resource
def train_model(df):
    X = df.drop("label", axis=1)
    y = df["label"]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), X.columns)
    ])

    model = Pipeline([
        ("prep", preprocessor),
        ("rf", RandomForestClassifier(n_estimators=260, random_state=42, class_weight="balanced"))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0)
    }

    return model, metrics


df = generate_data()
model, metrics = train_model(df)

# ==================================================
# HELPERS
# ==================================================

def get_openai_key():
    try:
        return st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        return os.environ.get("OPENAI_API_KEY")


def analyze_image_with_ai(uploaded_image):
    api_key = get_openai_key()

    if not api_key:
        return {"error": "No OpenAI API key found. Add OPENAI_API_KEY in Streamlit Secrets."}

    client = OpenAI(api_key=api_key)

    image_bytes = uploaded_image.getvalue()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    mime_type = uploaded_image.type or "image/jpeg"

    prompt = """
You are analyzing an uploaded image for a dog behavior interpretation prototype.

Determine whether the image clearly shows a dog. If not, return image_valid=false.
If it shows a dog, infer visible body-language cues using only the allowed values.

Return valid JSON only:
{
  "image_valid": true or false,
  "reason": "short explanation",
  "visual_mouth": "unknown | closed mouth | open mouth / panting",
  "visual_posture": "unknown | relaxed | alert | tense | crouched",
  "visual_ears": "unknown | neutral | back/pinned",
  "visual_tail": "unknown | relaxed | tucked | high",
  "visual_eyes": "unknown | relaxed | wide-eyed / whale eye",
  "visual_hiding": "unknown | yes | no",
  "confidence": 0.0 to 1.0
}

Be conservative. If a cue is not clearly visible, use "unknown".
"""

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": f"data:{mime_type};base64,{image_base64}"}
                    ]
                }
            ]
        )
        return json.loads(response.output_text)
    except Exception as e:
        return {"error": f"Image analysis failed: {str(e)}"}


def visual_cue_completeness(image_available, *visuals):
    known = sum(v != "unknown" for v in visuals)
    base = known / len(visuals)
    if image_available == "yes":
        return min(1.0, base + 0.15)
    return base * 0.75


def confidence_label(prob):
    if prob >= 0.75:
        return "High confidence"
    if prob >= 0.55:
        return "Moderate confidence"
    return "Low confidence"


def quality_label(score):
    if score >= 0.75:
        return "High"
    if score >= 0.4:
        return "Medium"
    return "Low"


def confidence_color(prob):
    if prob >= 0.75:
        return "🟢"
    if prob >= 0.55:
        return "🟡"
    return "🔴"


def get_feedback_count():
    try:
        feedback = pd.read_csv("feedback_log.csv")
        return len(feedback)
    except FileNotFoundError:
        return 0


def recommendation_for(pred):
    return {
        "recovery": {
            "summary": "Likely recovery. Support rest, hydration, and reassessment.",
            "do_now": ["Offer water.", "Move your dog to a calm, comfortable area.", "Pause additional play, running, or training for now."],
            "observe": ["Breathing and posture should gradually normalize.", "Panting should decrease as the dog rests and cools down.", "Energy should return to baseline after recovery."],
            "avoid": ["Do not continue intense activity immediately.", "Do not assume panting always means anxiety without considering recent activity or heat."]
        },
        "anxiety": {
            "summary": "Possible anxiety or stress. Reduce triggers and support calm observation.",
            "do_now": ["Lower noise and stimulation.", "Create distance from the trigger if possible.", "Give your dog space instead of forcing interaction."],
            "observe": ["Watch for repeated pacing, hiding, trembling, lip licking, or inability to settle.", "Track whether the behavior improves after the environment becomes calmer."],
            "avoid": ["Avoid punishment or forcing exposure.", "Avoid overwhelming the dog with attention if they are trying to retreat."]
        },
        "boredom": {
            "summary": "Possible unmet stimulation need. Use low-pressure enrichment.",
            "do_now": ["Offer a puzzle toy, sniff game, or short training session.", "Try calm engagement before escalating intensity.", "Use a structured activity rather than random excitement."],
            "observe": ["Behavior should decrease after mental stimulation.", "Toy-seeking or pacing may reduce when the dog has an appropriate outlet."],
            "avoid": ["Avoid only using high-arousal play if the dog is already restless.", "Avoid reinforcing demand behavior every time without structure."]
        },
        "overstimulation": {
            "summary": "Possible overstimulation. Pause activity and help your dog decompress.",
            "do_now": ["Stop exciting play or training.", "Move to a quieter area.", "Use calm routines and allow decompression."],
            "observe": ["Look for reduction in pacing, panting, jumping, or inability to settle.", "Reassess after 10–15 minutes of calm."],
            "avoid": ["Avoid more chase, fetch, or intense play immediately.", "Avoid adding more stimulation to an already aroused state."]
        },
        "neutral": {
            "summary": "No strong concern detected. Continue observing without over-intervening.",
            "do_now": ["Let your dog continue resting or behaving normally.", "Monitor for changes."],
            "observe": ["Look for sudden shifts in posture, breathing, responsiveness, or appetite.", "Reassess if the behavior becomes prolonged or unusual."],
            "avoid": ["Avoid unnecessary intervention if the dog appears relaxed.", "Avoid assuming every behavior requires correction."]
        },
        "needs observation": {
            "summary": "Unclear or potentially elevated concern. Observe closely and consider professional guidance if it persists.",
            "do_now": ["Reduce activity and keep the dog comfortable.", "Monitor breathing, posture, appetite, responsiveness, and duration.", "Document what you are seeing."],
            "observe": ["Watch whether the behavior resolves or escalates.", "Track whether it repeats under similar conditions."],
            "avoid": ["Avoid ignoring persistent or unusual behavior.", "Avoid relying on this prototype for medical decisions."]
        }
    }[pred]


def education_for(pred):
    return {
        "recovery": {
            "what_it_means": "Recovery means the behavior may be a normal response after activity, excitement, or warmth.",
            "common_cues": ["Panting after activity", "Short duration", "Relaxed or neutral body posture"],
            "watch_for": ["Panting that does not reduce with rest", "Labored breathing", "Lethargy or collapse"]
        },
        "anxiety": {
            "what_it_means": "Anxiety means the dog may be responding to stress, uncertainty, fear, or environmental triggers.",
            "common_cues": ["Pacing", "Hiding", "Pinned ears", "Tucked tail", "Whale eye"],
            "watch_for": ["Escalating avoidance", "Repeated inability to settle", "Frequent fear responses"]
        },
        "boredom": {
            "what_it_means": "Boredom means the dog may need mental stimulation, engagement, or structured activity.",
            "common_cues": ["Toy-seeking", "Restlessness", "Low recent activity", "Attention-seeking"],
            "watch_for": ["Demand behavior increasing", "Destructive behavior", "Difficulty settling"]
        },
        "overstimulation": {
            "what_it_means": "Overstimulation means the dog may be over-aroused and needs decompression rather than more activity.",
            "common_cues": ["Pacing after activity", "Panting after excitement", "Inability to settle"],
            "watch_for": ["Jumping, mouthing, or frantic behavior", "Escalation after more play", "Difficulty calming down"]
        },
        "neutral": {
            "what_it_means": "Neutral means the available signals do not strongly suggest stress, boredom, overstimulation, or concern.",
            "common_cues": ["Relaxed posture", "Short duration", "Resting", "No obvious stressors"],
            "watch_for": ["Sudden behavior changes", "Persistent unusual behavior", "Changes in appetite or responsiveness"]
        },
        "needs observation": {
            "what_it_means": "Needs observation means the signals are unclear or potentially elevated, so the safest response is monitoring and cautious escalation.",
            "common_cues": ["Long duration", "Senior dog", "Unclear cause", "Incomplete visual cues"],
            "watch_for": ["Symptoms worsening", "Repeated episodes", "Any medical warning signs"]
        }
    }[pred]


def escalation_guidance():
    return [
        "Seek veterinary care immediately if your dog shows labored breathing, collapse, pale gums, vomiting, severe distress, or other emergency signs.",
        "Consult a veterinarian if symptoms persist, worsen, or are unusual for your dog.",
        "Consult a certified trainer or veterinary behavior professional if anxiety-related behaviors are frequent, escalating, or interfering with daily life.",
        "This tool is for behavioral decision support only and does not replace professional medical or behavioral evaluation."
    ]


def factors(inputs):
    risk, protective, missing = [], [], []

    for key, label in [
        ("visual_mouth", "mouth / panting visibility"),
        ("visual_posture", "body posture"),
        ("visual_ears", "ear position"),
        ("visual_tail", "tail position"),
        ("visual_eyes", "eye expression"),
        ("visual_hiding", "hiding / avoidance visibility")
    ]:
        if inputs[key] == "unknown":
            missing.append(label)

    if inputs["behavior"] in ["pacing", "whining", "hiding"]:
        risk.append(f"Observed behavior: {inputs['behavior']}")
    if inputs["visual_posture"] in ["tense", "crouched"]:
        risk.append(f"Visual posture cue: {inputs['visual_posture']}")
    if inputs["visual_ears"] == "back/pinned":
        risk.append("Ears back/pinned")
    if inputs["visual_tail"] == "tucked":
        risk.append("Tail tucked")
    if inputs["visual_eyes"] == "wide-eyed / whale eye":
        risk.append("Wide-eyed / whale eye")
    if inputs["visual_hiding"] == "yes":
        risk.append("Hiding or avoidance visible")
    if inputs["environment"] == "noisy":
        risk.append("Noisy environment")
    if inputs["duration"] == "long":
        risk.append("Long duration")
    if inputs["sensitivity"] == "high":
        risk.append("High stress sensitivity")
    if inputs["age_group"] == "senior":
        risk.append("Senior age group")

    if inputs["activity"] == "high" and (
        inputs["behavior"] == "panting" or inputs["visual_mouth"] == "open mouth / panting"
    ):
        protective.append("Panting after high activity can indicate recovery")
    if inputs["duration"] == "short":
        protective.append("Short duration")
    if inputs["visual_mouth"] == "closed mouth":
        protective.append("Closed mouth")
    if inputs["visual_posture"] == "relaxed":
        protective.append("Relaxed posture")
    if inputs["visual_ears"] == "neutral":
        protective.append("Neutral ears")
    if inputs["visual_tail"] == "relaxed":
        protective.append("Relaxed tail")
    if inputs["visual_eyes"] == "relaxed":
        protective.append("Relaxed eyes")
    if inputs["sensitivity"] in ["low", "moderate"]:
        protective.append(f"{inputs['sensitivity'].title()} stress sensitivity")

    if not risk:
        risk.append("No major risk factors identified")
    if not protective:
        protective.append("Limited protective signals provided")

    return risk, protective, missing


def build_reasoning(inputs, image_available, completeness):
    reasons = []

    if image_available == "yes":
        reasons.append("AI-extracted image cues were included, so the interpretation uses both owner context and visible body-language signals.")
    else:
        reasons.append("No image was used in this run. The interpretation is based on profile and behavior context only.")

    if inputs["visual_posture"] == "relaxed" and inputs["visual_eyes"] == "relaxed":
        reasons.append("Relaxed posture and relaxed eyes reduce concern for anxiety.")
    if inputs["visual_eyes"] == "wide-eyed / whale eye":
        reasons.append("Whale eye is a strong stress cue and increases concern for anxiety.")
    if inputs["visual_tail"] == "tucked" or inputs["visual_ears"] == "back/pinned":
        reasons.append("Tucked tail or pinned ears increase stress likelihood.")
    if inputs["visual_mouth"] == "closed mouth":
        reasons.append("Closed mouth can lower concern for panting-related recovery or overstimulation.")
    if inputs["activity"] == "high":
        reasons.append("Recent high activity increases likelihood of recovery or overstimulation.")
    if inputs["activity"] == "none":
        reasons.append("No recent activity makes recovery less likely and increases relevance of boredom or stress.")
    if inputs["duration"] == "long":
        reasons.append("Long duration raises concern because the behavior is less likely to be brief or incidental.")
    if inputs["duration"] == "short":
        reasons.append("Short duration lowers concern and may indicate a temporary state.")

    reasons.append(f"Interpretation quality is {quality_label(completeness)} based on available visual and contextual cues.")
    return reasons


def confidence_drivers(inputs, completeness, adjusted_prob):
    drivers = []

    drivers.append("Image cues included" if inputs["image_available"] == "yes" else "No image cues included")

    if completeness >= 0.75:
        drivers.append("High input completeness")
    elif completeness >= 0.4:
        drivers.append("Moderate input completeness")
    else:
        drivers.append("Low input completeness")

    if inputs["assumption"] != "unsure":
        drivers.append("Owner baseline assumption available")
    else:
        drivers.append("No owner baseline assumption")

    if adjusted_prob < 0.55:
        drivers.append("Low confidence: signals are mixed or incomplete")
    elif adjusted_prob >= 0.75:
        drivers.append("High confidence: signals are internally consistent")
    else:
        drivers.append("Moderate confidence: some ambiguity remains")

    return drivers


def what_would_change_prediction(inputs):
    suggestions = []

    if inputs["image_available"] == "no":
        suggestions.append("Uploading a clear dog image could add body-language cues and improve interpretation quality.")
    if inputs["visual_posture"] == "unknown":
        suggestions.append("Knowing whether posture is relaxed, tense, or crouched would help distinguish anxiety from recovery or neutrality.")
    if inputs["visual_eyes"] == "unknown":
        suggestions.append("Eye expression could strongly affect anxiety scoring, especially whale eye versus relaxed eyes.")
    if inputs["visual_tail"] == "unknown" or inputs["visual_ears"] == "unknown":
        suggestions.append("Ear and tail position could shift the interpretation toward or away from stress.")
    if inputs["behavior"] == "panting":
        suggestions.append("If panting decreases after rest, recovery becomes more likely; if it persists without heat/activity, concern increases.")
    if inputs["behavior"] == "pacing":
        suggestions.append("If pacing decreases after enrichment, boredom becomes more likely; if it persists with stress cues, anxiety becomes more likely.")

    if not suggestions:
        suggestions.append("More follow-up feedback would improve future interpretation.")

    return suggestions


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:
    st.header("Prototype Model Metrics")
    st.metric("Accuracy", f"{metrics['accuracy']:.0%}")
    st.metric("Weighted Precision", f"{metrics['precision']:.0%}")
    st.metric("Weighted Recall", f"{metrics['recall']:.0%}")

    with st.expander("How AI is used", expanded=True):
        st.write(
            """
            This prototype uses a hybrid AI approach:
            
            1. A supervised ML classifier learns behavior patterns from synthetic labeled examples.
            2. A behavioral evidence scoring layer calibrates strong cues like whale eye, tucked tail, relaxed posture, and activity context.
            3. An optional vision model can validate dog images and pre-fill visual cues.
            
            The app is preventive behavioral decision support, not diagnosis.
            """
        )

    with st.expander("Data transparency"):
        st.write(
            """
            The current model uses synthetic training data and structured behavioral logic.
            Feedback is logged for future retraining but does not automatically retrain the model after each submission.
            """
        )


# ==================================================
# HEADER
# ==================================================

st.title("🐶 Dog Behavior Interpreter")
st.write(
    "A preventive AI decision-support prototype that helps owners understand ambiguous dog behavior using profile, context, and optional image cues."
)

st.markdown("""
<div class="disclaimer-card">
⚠️ <b>Important:</b> This prototype provides behavioral guidance only. It is <b>not a medical or diagnostic tool</b>.
Seek veterinary care immediately if your dog shows labored breathing, collapse, pale gums, vomiting, severe distress,
or symptoms that are unusual, persistent, or worsening.
</div>
""", unsafe_allow_html=True)

steps = ["Profile", "Context", "Visual Cues", "Interpretation", "Feedback"]

step_html = "<div class='stepper'>"
for i, step in enumerate(steps, start=1):
    if i < st.session_state.phase:
        cls = "step-complete"
    elif i == st.session_state.phase:
        cls = "step-active"
    else:
        cls = "step-inactive"
    step_html += f"<span class='{cls}'>{i}. {step}</span>"
step_html += "</div>"
st.markdown(step_html, unsafe_allow_html=True)


# ==================================================
# PHASE 1
# ==================================================

if st.session_state.phase == 1:
    st.markdown("## Phase 1 — Dog Profile")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.session_state.dog_name = st.text_input("Dog name", value=st.session_state.get("dog_name", "Sadie"), help="Used to personalize the output only.")
        st.session_state.age_group = st.selectbox("Age group", ["puppy", "adult", "senior"], index=["puppy", "adult", "senior"].index(st.session_state.get("age_group", "adult")))

    with c2:
        st.session_state.size = st.selectbox("Size category", ["small", "medium", "large"], index=["small", "medium", "large"].index(st.session_state.get("size", "medium")))
        st.session_state.energy = st.selectbox("Typical energy level", ["low", "moderate", "high"], index=["low", "moderate", "high"].index(st.session_state.get("energy", "high")))

    with c3:
        st.session_state.sensitivity = st.selectbox("Known stress sensitivity", ["low", "moderate", "high"], index=["low", "moderate", "high"].index(st.session_state.get("sensitivity", "moderate")))
        st.session_state.assumption = st.selectbox("Your initial assumption", ["unsure", "anxiety", "boredom", "overstimulation", "recovery"], index=["unsure", "anxiety", "boredom", "overstimulation", "recovery"].index(st.session_state.get("assumption", "unsure")))

    nav1, nav2, spacer = st.columns([1, 1, 3])
    with nav1:
        st.button("Next: Context", type="primary", on_click=set_phase, args=(2,), use_container_width=True)


# ==================================================
# PHASE 2
# ==================================================

elif st.session_state.phase == 2:
    st.markdown("## Phase 2 — Current Behavior Context")

    c1, c2 = st.columns(2)

    with c1:
        st.session_state.behavior = st.selectbox("Observed behavior", ["panting", "pacing", "whining", "hiding", "toy-seeking", "resting"], index=["panting", "pacing", "whining", "hiding", "toy-seeking", "resting"].index(st.session_state.get("behavior", "panting")))
        st.session_state.activity = st.selectbox("Recent activity", ["none", "low", "high"], index=["none", "low", "high"].index(st.session_state.get("activity", "none")))

    with c2:
        st.session_state.environment = st.selectbox("Environment", ["indoor", "outdoor", "warm", "noisy"], index=["indoor", "outdoor", "warm", "noisy"].index(st.session_state.get("environment", "indoor")))
        st.session_state.duration = st.selectbox("Duration", ["short", "medium", "long"], index=["short", "medium", "long"].index(st.session_state.get("duration", "short")))

    nav1, nav2, spacer = st.columns([1, 1, 3])
    with nav1:
        st.button("Back", on_click=set_phase, args=(1,), use_container_width=True)
    with nav2:
        st.button("Next: Visual Cues", type="primary", on_click=set_phase, args=(3,), use_container_width=True)


# ==================================================
# PHASE 3
# ==================================================

elif st.session_state.phase == 3:
    st.markdown("## Phase 3 — AI Image Cue Review")

    st.write("Upload a dog image if available. The app can use AI to validate the image and pre-fill visible body-language cues. You can review and override the selections.")

    image = st.file_uploader("Upload dog image", type=["jpg", "jpeg", "png"])

    invalid_image = False

    if image:
        st.image(image, caption="Uploaded image for AI cue extraction", width=350)

        if st.button("Analyze Image Cues with AI", type="secondary"):
            with st.spinner("Analyzing image cues..."):
                image_result = analyze_image_with_ai(image)

            if "error" in image_result:
                st.error(image_result["error"])
                st.info("You can still continue without image AI by manually selecting cues.")
                st.session_state.image_available = "no"

            elif not image_result.get("image_valid", False):
                st.error("Invalid image. Please upload an image that clearly shows a dog.")
                st.session_state.image_available = "no"
                invalid_image = True

            else:
                st.success("Dog image validated. Visual cues were auto-filled.")
                st.write(image_result.get("reason", ""))

                st.session_state.image_available = "yes"
                st.session_state.visual_mouth = image_result.get("visual_mouth", "unknown")
                st.session_state.visual_posture = image_result.get("visual_posture", "unknown")
                st.session_state.visual_ears = image_result.get("visual_ears", "unknown")
                st.session_state.visual_tail = image_result.get("visual_tail", "unknown")
                st.session_state.visual_eyes = image_result.get("visual_eyes", "unknown")
                st.session_state.visual_hiding = image_result.get("visual_hiding", "unknown")
                st.session_state.image_ai_confidence = image_result.get("confidence", None)
                st.session_state.image_ai_reason = image_result.get("reason", None)
                st.rerun()
    else:
        st.session_state.image_available = "no"

    c1, c2, c3 = st.columns(3)

    with c1:
        st.session_state.visual_mouth = st.selectbox("Mouth / panting visible", ["unknown", "closed mouth", "open mouth / panting"], index=["unknown", "closed mouth", "open mouth / panting"].index(st.session_state.visual_mouth))
        st.session_state.visual_posture = st.selectbox("Body posture", ["unknown", "relaxed", "alert", "tense", "crouched"], index=["unknown", "relaxed", "alert", "tense", "crouched"].index(st.session_state.visual_posture))

    with c2:
        st.session_state.visual_ears = st.selectbox("Ear position", ["unknown", "neutral", "back/pinned"], index=["unknown", "neutral", "back/pinned"].index(st.session_state.visual_ears))
        st.session_state.visual_tail = st.selectbox("Tail position", ["unknown", "relaxed", "tucked", "high"], index=["unknown", "relaxed", "tucked", "high"].index(st.session_state.visual_tail))

    with c3:
        st.session_state.visual_eyes = st.selectbox("Eye expression", ["unknown", "relaxed", "wide-eyed / whale eye"], index=["unknown", "relaxed", "wide-eyed / whale eye"].index(st.session_state.visual_eyes))
        st.session_state.visual_hiding = st.selectbox("Hiding / avoidance visible", ["unknown", "yes", "no"], index=["unknown", "yes", "no"].index(st.session_state.visual_hiding))

    visuals = [
        st.session_state.visual_mouth,
        st.session_state.visual_posture,
        st.session_state.visual_ears,
        st.session_state.visual_tail,
        st.session_state.visual_eyes,
        st.session_state.visual_hiding
    ]

    completeness = visual_cue_completeness(st.session_state.image_available, *visuals)

    st.markdown("### Input Quality Preview")
    st.write(f"**Dog image used:** {'Yes' if st.session_state.image_available == 'yes' else 'No'}")
    st.write(f"**Visual cue completeness:** {quality_label(completeness)}")
    st.progress(completeness)

    nav1, nav2, spacer = st.columns([1, 1, 3])
    with nav1:
        st.button("Back", on_click=set_phase, args=(2,), use_container_width=True)

    with nav2:
        analyze_clicked = st.button("Analyze", type="primary", use_container_width=True, disabled=invalid_image)

    if analyze_clicked:
        inputs = {
            "behavior": st.session_state.behavior,
            "activity": st.session_state.activity,
            "environment": st.session_state.environment,
            "duration": st.session_state.duration,
            "assumption": st.session_state.assumption,
            "age_group": st.session_state.age_group,
            "size": st.session_state.size,
            "energy": st.session_state.energy,
            "sensitivity": st.session_state.sensitivity,
            "image_available": st.session_state.image_available,
            "visual_mouth": st.session_state.visual_mouth,
            "visual_posture": st.session_state.visual_posture,
            "visual_ears": st.session_state.visual_ears,
            "visual_tail": st.session_state.visual_tail,
            "visual_eyes": st.session_state.visual_eyes,
            "visual_hiding": st.session_state.visual_hiding
        }

        pred, prob_df, evidence_probs, ml_probs, raw_scores = hybrid_predict(inputs, model)

        top_prob = float(prob_df.iloc[0]["Probability"])
        second_state = prob_df.iloc[1]["Behavior State"]
        second_prob = float(prob_df.iloc[1]["Probability"])

        st.session_state.prediction_result = {
            "inputs": inputs,
            "pred": pred,
            "prob_df": prob_df,
            "evidence_probs": evidence_probs,
            "ml_probs": ml_probs,
            "raw_scores": raw_scores,
            "adjusted_prob": top_prob,
            "second_state": second_state,
            "second_prob": second_prob,
            "completeness": completeness
        }

        set_phase(4)
        st.rerun()


# ==================================================
# PHASE 4
# ==================================================

elif st.session_state.phase == 4:
    st.markdown("## Phase 4 — AI Interpretation")

    result = st.session_state.prediction_result

    if result is None:
        st.warning("No analysis has been generated yet.")
        st.button("Go to Visual Cues", on_click=set_phase, args=(3,))
    else:
        inputs = result["inputs"]
        pred = result["pred"]
        prob_df = result["prob_df"]
        adjusted_prob = result["adjusted_prob"]
        second_state = result["second_state"]
        second_prob = result["second_prob"]
        completeness = result["completeness"]
        evidence_probs = result["evidence_probs"]
        ml_probs = result["ml_probs"]

        risk, protective, missing = factors(inputs)
        rec = recommendation_for(pred)
        education = education_for(pred)

        context_only_inputs = inputs.copy()
        context_only_inputs["image_available"] = "no"
        context_only_inputs["visual_mouth"] = "unknown"
        context_only_inputs["visual_posture"] = "unknown"
        context_only_inputs["visual_ears"] = "unknown"
        context_only_inputs["visual_tail"] = "unknown"
        context_only_inputs["visual_eyes"] = "unknown"
        context_only_inputs["visual_hiding"] = "unknown"

        context_pred, context_prob_df, _, _, _ = hybrid_predict(context_only_inputs, model)
        context_confidence = float(context_prob_df.iloc[0]["Probability"])

        r1, r2, r3 = st.columns([1.05, 1.1, 0.85])

        with r1:
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.markdown(f"### {confidence_color(adjusted_prob)} Likely state")
            st.markdown(f"## {pred.title()}")
            st.write(f"**Dog:** {st.session_state.dog_name}")
            st.write(f"**Confidence:** {adjusted_prob:.0%} — {confidence_label(adjusted_prob)}")
            st.progress(adjusted_prob)
            st.write(f"**Secondary:** {second_state.title()} ({second_prob:.0%})")
            st.markdown("</div>", unsafe_allow_html=True)

        with r2:
            st.markdown("<div class='action-hero'>", unsafe_allow_html=True)
            st.markdown("### Recommended action")
            st.write(rec["summary"])
            for step in rec["do_now"][:3]:
                st.write(f"- {step}")
            st.markdown("</div>", unsafe_allow_html=True)

        with r3:
            st.markdown("<div class='quality-card'>", unsafe_allow_html=True)
            st.markdown("### Quality")
            st.write(f"**Input:** {quality_label(completeness)}")
            st.write(f"**Image cues:** {'Used' if inputs['image_available'] == 'yes' else 'Not used'}")
            st.write(f"**Missing cues:** {len(missing)}")
            st.progress(completeness)
            st.markdown("</div>", unsafe_allow_html=True)

        if adjusted_prob < 0.55:
            st.markdown("<div class='warning-card'>", unsafe_allow_html=True)
            st.warning("Low confidence means the model found mixed or incomplete signals. Use this as a prompt for observation, not a final answer.")
            st.markdown("</div>", unsafe_allow_html=True)

        st.info(
            f"This does not mean {st.session_state.dog_name} is definitely {pred}. "
            f"It means the available signals are most consistent with **{pred}**, while some uncertainty may remain."
        )

        tabs = st.tabs(["Overview", "Why this result", "Image impact", "Learn", "Details"])

        with tabs[0]:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown("### Clear action plan")
            a1, a2, a3, a4 = st.columns(4)

            with a1:
                st.markdown("#### Do now")
                for step in rec["do_now"]:
                    st.write(f"- {step}")
            with a2:
                st.markdown("#### Observe next")
                for step in rec["observe"]:
                    st.write(f"- {step}")
            with a3:
                st.markdown("#### Avoid")
                for step in rec["avoid"]:
                    st.write(f"- {step}")
            with a4:
                st.markdown("#### Escalate if")
                st.write("- Labored breathing")
                st.write("- Collapse or pale gums")
                st.write("- Vomiting or severe distress")
                st.write("- Symptoms worsen or persist")
            st.markdown("</div>", unsafe_allow_html=True)

        with tabs[1]:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown("### AI reasoning summary")
            for reason in build_reasoning(inputs, inputs["image_available"], completeness):
                st.write(f"- {reason}")

            st.markdown("### Confidence drivers")
            for driver in confidence_drivers(inputs, completeness, adjusted_prob):
                st.write(f"- {driver}")

            st.markdown("### What could change this prediction?")
            for item in what_would_change_prediction(inputs):
                st.write(f"- {item}")
            st.markdown("</div>", unsafe_allow_html=True)

        with tabs[2]:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown("### Image cue impact comparison")
            c1, c2 = st.columns(2)

            with c1:
                st.markdown("#### Context only")
                st.write("Profile + behavior context")
                st.write(f"**Prediction:** {context_pred.title()}")
                st.write(f"**Confidence:** {context_confidence:.0%}")

            with c2:
                st.markdown("#### Context + dog image cues")
                st.write("Profile + context + visual cues")
                st.write(f"**Prediction:** {pred.title()}")
                st.write(f"**Confidence:** {adjusted_prob:.0%}")

            if inputs["image_available"] == "yes":
                if context_pred != pred:
                    st.success("The dog image cues changed the predicted state.")
                else:
                    st.info("The dog image cues did not change the top state, but may still affect confidence.")
            else:
                st.warning("Because no dog image was used, the system relied mostly on structured context.")
            st.markdown("</div>", unsafe_allow_html=True)

        with tabs[3]:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown("### Behavior education")
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown("#### What this means")
                st.write(education["what_it_means"])
            with c2:
                st.markdown("#### Common cues")
                for cue in education["common_cues"]:
                    st.write(f"- {cue}")
            with c3:
                st.markdown("#### Watch for")
                for cue in education["watch_for"]:
                    st.write(f"- {cue}")
            st.markdown("</div>", unsafe_allow_html=True)

        with tabs[4]:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown("### Risk, protective, and missing signals")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("#### Risk factors")
                for item in risk:
                    st.write(f"- {item}")
            with c2:
                st.markdown("#### Protective factors")
                for item in protective:
                    st.write(f"- {item}")
            with c3:
                st.markdown("#### Missing signals")
                if missing:
                    for item in missing:
                        st.write(f"- {item}")
                else:
                    st.write("- No major visual cues missing")

            st.markdown("### Prediction breakdown")
            breakdown = pd.DataFrame({
                "Behavior State": CLASSES,
                "Evidence Score Probability": [evidence_probs[c] for c in CLASSES],
                "ML Model Probability": [ml_probs[c] for c in CLASSES],
            })
            st.dataframe(breakdown, use_container_width=True)

            with st.expander("Probability distribution"):
                st.bar_chart(prob_df.set_index("Behavior State"))

            with st.expander("When to seek professional help"):
                for item in escalation_guidance():
                    st.write(f"- {item}")
            st.markdown("</div>", unsafe_allow_html=True)

        st.write("The interpretation is complete. Feedback helps improve future versions but is not required.")

        nav1, nav2, spacer = st.columns([1, 1, 3])
        with nav1:
            st.button("Back", on_click=set_phase, args=(3,), use_container_width=True)
        with nav2:
            st.button("Continue to Feedback", type="primary", on_click=set_phase, args=(5,), use_container_width=True)


# ==================================================
# PHASE 5
# ==================================================

elif st.session_state.phase == 5:
    st.markdown("## Phase 5 — Feedback")
    st.write("Feedback helps demonstrate how the prototype could learn from real-world outcomes. You can also start a new interpretation without submitting feedback.")

    result = st.session_state.prediction_result

    if result is None:
        st.warning("No prediction available.")
        st.button("Start Over", on_click=set_phase, args=(1,))
    else:
        inputs = result["inputs"]
        pred = result["pred"]
        adjusted_prob = result["adjusted_prob"]
        second_state = result["second_state"]
        second_prob = result["second_prob"]
        completeness = result["completeness"]

        feedback_count = get_feedback_count()

        st.markdown("### Learning Loop Status")
        ll1, ll2, ll3 = st.columns(3)

        with ll1:
            st.metric("Feedback records collected", feedback_count)
        with ll2:
            st.metric("Current training source", "Synthetic + hybrid scoring")
        with ll3:
            st.metric("Retraining mode", "Manual / future batch")

        st.info(
            "This prototype logs feedback for future dataset improvement. It does not automatically retrain after each submission because production systems typically retrain after enough validated feedback is collected."
        )

        with st.form("feedback_form"):
            fb1, fb2, fb3 = st.columns(3)

            with fb1:
                accurate = st.radio("Was this accurate?", ["Yes", "No", "Unsure"], horizontal=True)
            with fb2:
                action_taken = st.radio("Did you take the action?", ["Yes", "No", "Partially"], horizontal=True)
            with fb3:
                improved = st.radio("Did behavior improve?", ["Yes", "No", "Not sure"], horizontal=True)

            feedback_submitted = st.form_submit_button("Submit Feedback", use_container_width=True)

        if feedback_submitted:
            feedback_row = {
                "timestamp": datetime.now().isoformat(),
                "dog_name": st.session_state.dog_name,
                **inputs,
                "prediction": pred,
                "confidence": round(adjusted_prob, 3),
                "secondary_state": second_state,
                "secondary_probability": round(second_prob, 3),
                "input_completeness": round(completeness, 3),
                "accurate_feedback": accurate,
                "action_taken": action_taken,
                "behavior_improved": improved
            }

            feedback_df = pd.DataFrame([feedback_row])

            try:
                existing = pd.read_csv("feedback_log.csv")
                updated = pd.concat([existing, feedback_df], ignore_index=True)
            except FileNotFoundError:
                updated = feedback_df

            updated.to_csv("feedback_log.csv", index=False)

            st.session_state.feedback_done = True
            st.success("Feedback saved for future dataset improvement and retraining.")

            with st.expander("Feedback record"):
                st.dataframe(feedback_df)

        nav1, nav2, spacer = st.columns([1, 1, 3])
        with nav1:
            st.button("Back", on_click=set_phase, args=(4,), use_container_width=True)
        with nav2:
            if st.button("Start New", use_container_width=True):
                for key in [
                    "prediction_result",
                    "feedback_done",
                    "phase",
                    "image_available",
                    "visual_mouth",
                    "visual_posture",
                    "visual_ears",
                    "visual_tail",
                    "visual_eyes",
                    "visual_hiding",
                    "image_ai_confidence",
                    "image_ai_reason"
                ]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
