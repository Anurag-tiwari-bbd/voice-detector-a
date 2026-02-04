import numpy as np
import librosa

def analyze_voice(y, sr):
    # ------------------
    # Feature extraction
    # ------------------
    pitch = librosa.yin(y, fmin=50, fmax=300)
    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    rms = librosa.feature.rms(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)

    # ------------------
    # Statistics
    # ------------------
    pitch_var = np.nanvar(pitch)
    flatness_mean = np.mean(spectral_flatness)
    rms_var = np.var(rms)
    zcr_var = np.var(zcr)

    # ------------------
    # Heuristic scoring
    # ------------------
    score = 0
    explanation = []

    if pitch_var < 15:
        score += 0.35
        explanation.append("Unnatural pitch stability")

    if flatness_mean > 0.18:
        score += 0.35
        explanation.append("Spectral smoothness typical of AI")

    if rms_var < 0.008:
        score += 0.45
        explanation.append("Low energy variation typical of synthetic speech")


    if zcr_var < 0.00001:
        score += 0.2
        explanation.append("Overly smooth signal transitions")

    # ------------------
    # Confidence and classification
    # ------------------
    confidence = min(score, 1.0)
    classification = "AI_GENERATED" if confidence >= 0.35 else "HUMAN"

    if "synthetic" in " ".join(explanation).lower() and confidence < 0.35:
        classification = "AI_GENERATED"
        confidence = max(confidence, 0.35)


    if not explanation:
        explanation.append("Natural human speech variations detected")

    return classification, confidence, "; ".join(explanation)
