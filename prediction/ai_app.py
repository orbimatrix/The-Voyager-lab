import os
from dotenv import load_dotenv
import gradio as gr
import joblib
import pandas as pd
import numpy as np
from openai import OpenAI

# Load .env from project root
load_dotenv()  # <-- must be called before reading os.getenv

API_KEY = os.getenv("API_KEY")          # matches the name in .env
BASE_URL = os.getenv("BASE_URL", "https://api.aimlapi.com/v1")

if not API_KEY:
    raise RuntimeError("API_KEY not found in environment. Did you create a .env and call load_dotenv()?")

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)

# Load the trained model and preprocessing objects
model = joblib.load('exoplanet_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

def generate_llm_explanation(prediction, input_data, engineered_features, probabilities):
    """
    Generate natural language explanation using LLM
    """
    try:
        # Prepare context for the LLM
        result_type = "an exoplanet" if prediction == 1 else "a non-exoplanet"
        confidence = probabilities[prediction] if probabilities is not None else "N/A"
        
        prompt = f"""You are an expert astronomer analyzing exoplanet detection data. The Machine learning model predicts how much it is exoplanet and now you need to explain why the machine learning model classified this as {result_type}.

            Prediction: {result_type.upper()}
            Confidence: {confidence:.2%} if isinstance(confidence, (int, float)) else confidence

            Input Parameters 
            - Orbital Period: {input_data['tce_period']:.2f} days
            - Transit Epoch: {input_data['tce_time0bk']:.2f} BJD
            - Impact Parameter: {input_data['tce_impact']:.3f} (0=center, 1=grazing)
            - Transit Duration: {input_data['tce_duration']:.2f} hours
            - Transit Depth: {input_data['tce_depth']:.1f} ppm
            - Signal-to-Noise Ratio: {input_data['tce_model_snr']:.2f}
            - Planet Radius: {input_data['tce_prad']:.2f} Earth radii
            - Equilibrium Temperature: {input_data['tce_eqt']:.1f} K
            - Insolation: {input_data['tce_insol']:.2f} Earth flux
            - Stellar Temperature: {input_data['tce_steff']:.1f} K
            - Stellar Surface Gravity: {input_data['tce_slogg']:.2f} log(g)
            - Stellar Radius: {input_data['tce_sradius']:.2f} Solar radii

            Explanation Guidelines:
            - Use the input parameters to explain the classification.
            - How often the dip in starlight repeats: every {input_data['tce_period']:.2f} days
            - How much the star dims during transit: {input_data['tce_depth']:.1f} parts per million
            - How long each dimming event lasts: {input_data['tce_duration']:.2f} hours
            - Signal quality/clarity: {input_data['tce_model_snr']:.2f} (higher = clearer signal)
            - Where the object crosses the star: {input_data['tce_impact']:.3f} (0 = dead center, 1 = barely grazing edge)
            - Estimated size of the object: {input_data['tce_prad']:.2f} times Earth's size
            - Temperature of the object: {input_data['tce_eqt']:.1f} Kelvin
            - How much starlight hits it: {input_data['tce_insol']:.2f} times what Earth gets
            - Temperature of the host star: {input_data['tce_steff']:.1f} Kelvin
            - Size of the host star: {input_data['tce_sradius']:.2f} times our Sun
            - Star's surface gravity: {input_data['tce_slogg']:.2f} (tells us if it's a normal star or something else)

            Engineered Features:
            - Temperature/Radius Ratio: {engineered_features['temp_radius_ratio']:.2f}
            - Period/Duration Ratio: {engineered_features['period_duration_ratio']:.2f}
            - SNR × Depth Product: {engineered_features['snr_depth_product']:.2f}

            YOUR TASK: Write a clear, conversational explanation (5-7 sentences) that a middle schooler could understand. Explain:

        1. WHAT we're looking at: Describe what causes the star to dim (don't just say "transit" - explain it's like an object passing in front)

        2. WHY the model thinks it's {result_type}:
        - If EXOPLANET: Explain what specific measurements look RIGHT for a planet (e.g., "The size is reasonable for a planet", "It crosses the star regularly like a planet in orbit would", "The signal is strong and clear, not messy like false alarms")
        - If FALSE POSITIVE: Explain what looks WRONG (e.g., "The object is way too big to be a planet - it's probably a small star", "The signal is too weak and noisy", "The way it crosses the star is weird for a planet")

        3. THE KEY EVIDENCE: Pick 2-3 specific measurements that most strongly support this conclusion and explain WHY they matter in plain English

        4. THE VERDICT: Wrap up by saying whether this is likely a real planet or not and why we can/can't trust this

        DO NOT:
        - Use technical jargon without explaining it
        - Just list numbers - explain what they MEAN
        - Say "the model detected" - explain the actual physics/reasons
        - Use bullet points or lists

        BE CONVERSATIONAL and use analogies where helpful (e.g., "imagine Earth passing in front of the Sun from far away")

        """

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"LLM explanation unavailable: {str(e)}"

def predict_exoplanet(period, time0bk, impact, duration, depth, snr, prad, 
                      eqt, insol, steff, slogg, sradius):
    """
    Predict if the input represents an exoplanet or false positive
    """
    try:
        # Create input dictionary
        input_data = {
            'tce_period': float(period),
            'tce_time0bk': float(time0bk),
            'tce_impact': float(impact),
            'tce_duration': float(duration),
            'tce_depth': float(depth),
            'tce_model_snr': float(snr),
            'tce_prad': float(prad),
            'tce_eqt': float(eqt),
            'tce_insol': float(insol),
            'tce_steff': float(steff),
            'tce_slogg': float(slogg),
            'tce_sradius': float(sradius)
        }
        
        # Create DataFrame
        df = pd.DataFrame([input_data])
        
        # Add engineered features
        df['temp_radius_ratio'] = df['tce_steff'] / (df['tce_sradius'] + 1e-6)
        df['period_duration_ratio'] = df['tce_period'] / (df['tce_duration'] + 1e-6)
        df['snr_depth_product'] = df['tce_model_snr'] * df['tce_depth']
        
        engineered_features = {
            'temp_radius_ratio': df['temp_radius_ratio'].values[0],
            'period_duration_ratio': df['period_duration_ratio'].values[0],
            'snr_depth_product': df['snr_depth_product'].values[0]
        }
        
        # Ensure correct feature order
        df = df[feature_names]
        
        # Scale features
        df_scaled = scaler.transform(df)
        
        # Make prediction
        prediction = model.predict(df_scaled)[0]
        
        # Get probability if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(df_scaled)[0]
            confidence = probabilities[prediction]
            prob_exoplanet = probabilities[1]
            prob_false_positive = probabilities[0]
        else:
            confidence = None
            prob_exoplanet = prediction
            prob_false_positive = 1 - prediction
            probabilities = None
        
        # Generate LLM explanation
        llm_explanation = generate_llm_explanation(
            prediction, input_data, engineered_features, probabilities
        )
        
        # Format result
        result = "🪐 EXOPLANET DETECTED" if prediction == 1 else "❌ FALSE POSITIVE"
        
        # Create detailed output
        details = f"""
### Prediction: {result}

**Confidence:** {f"{confidence:.2%}" if confidence else 'N/A'}

**Probability Breakdown:**
- Exoplanet: {prob_exoplanet:.2%}
- False Positive: {prob_false_positive:.2%}

---

### 🤖 AI Explanation:

{llm_explanation}

---

### 📊 Input Summary:
- **Orbital Period:** {period} days
- **Transit Depth:** {depth} ppm
- **Signal-to-Noise Ratio:** {snr}
- **Planet Radius:** {prad} Earth radii
- **Stellar Temperature:** {steff} K
- **Impact Parameter:** {impact}

### 🔧 Engineered Features:
- **Temp/Radius Ratio:** {engineered_features['temp_radius_ratio']:.2f}
- **Period/Duration Ratio:** {engineered_features['period_duration_ratio']:.2f}
- **SNR × Depth:** {engineered_features['snr_depth_product']:.2f}
        """
        
        return details
        
    except Exception as e:
        return f"Error: {str(e)}\n\nPlease check your input values."

# Example datasets for quick testing
examples = [
    # Confirmed exoplanet-like parameters
    [3.5, 135.0, 0.3, 2.5, 800, 20.0, 2.0, 450, 100, 5800, 4.5, 1.0],
    # False positive-like parameters
    [50.0, 200.0, 0.9, 15.0, 50, 5.0, 15.0, 300, 5, 4500, 4.8, 0.5],
    # Hot Jupiter-like
    [0.8, 140.0, 0.1, 1.2, 15000, 50.0, 11.0, 1500, 5000, 6200, 4.3, 1.2],
]

# Create Gradio interface
with gr.Blocks(title="Exoplanet Detection System", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🌌 Exoplanet Detection System with AI Explanations
        
        This AI-powered system predicts whether a transit signal represents a genuine exoplanet 
        or a false positive based on Kepler space telescope data, with intelligent explanations.
        
        ### How to use:
        1. Enter the transit and stellar parameters below
        2. Click "Detect Exoplanet" to get prediction
        3. Read the AI-generated explanation of why this classification was made
        4. Or try example cases for quick testing
        """
    )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Transit Parameters")
            period = gr.Number(label="Orbital Period (days)", value=10.0, 
                             info="Time between successive transits")
            time0bk = gr.Number(label="Transit Epoch (BJD - 2454833)", value=135.0,
                              info="Time of first transit")
            impact = gr.Slider(0, 1, value=0.5, label="Impact Parameter",
                             info="0 = center crossing, 1 = grazing")
            duration = gr.Number(label="Transit Duration (hours)", value=3.0,
                               info="How long the transit lasts")
            depth = gr.Number(label="Transit Depth (ppm)", value=500.0,
                            info="Fractional decrease in brightness")
            snr = gr.Number(label="Signal-to-Noise Ratio", value=15.0,
                          info="Quality of the detection")
            
        with gr.Column():
            gr.Markdown("### Planet & Stellar Parameters")
            prad = gr.Number(label="Planet Radius (Earth radii)", value=2.5,
                           info="Size relative to Earth")
            eqt = gr.Number(label="Equilibrium Temperature (K)", value=400.0,
                          info="Estimated planet temperature")
            insol = gr.Number(label="Insolation (Earth flux)", value=50.0,
                            info="Stellar flux received")
            steff = gr.Number(label="Stellar Temperature (K)", value=5500.0,
                            info="Host star temperature")
            slogg = gr.Slider(2, 5, value=4.5, label="Stellar Surface Gravity (log g)",
                            info="Indicates star type")
            sradius = gr.Number(label="Stellar Radius (Solar radii)", value=1.0,
                              info="Size of host star")
    
    detect_btn = gr.Button("🔍 Detect Exoplanet", variant="primary", size="lg")
    
    output = gr.Markdown(label="Prediction Results")
    
    detect_btn.click(
        fn=predict_exoplanet,
        inputs=[period, time0bk, impact, duration, depth, snr, prad, 
                eqt, insol, steff, slogg, sradius],
        outputs=output
    )
    
    gr.Markdown("### 📋 Example Cases")
    gr.Examples(
        examples=examples,
        inputs=[period, time0bk, impact, duration, depth, snr, prad, 
                eqt, insol, steff, slogg, sradius],
        label="Try these example transit signals"
    )
    
    gr.Markdown(
        """
        ---
        **Model Information:**
        - Trained on Kepler mission data
        - Uses ensemble machine learning techniques
        - Features: Transit timing, depth, stellar parameters, and more
        - AI explanations powered by GPT-4o
        
        **Note:** This is a demonstration tool. Real exoplanet confirmation requires 
        extensive follow-up observations and validation.
        """
    )

if __name__ == "__main__":
    demo.launch(share=True)