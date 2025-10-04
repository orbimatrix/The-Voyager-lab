import gradio as gr
import joblib
import pandas as pd
import numpy as np

# Load the trained model and preprocessing objects
model = joblib.load('exoplanet_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

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
        
        # Format result
        result = "🪐 EXOPLANET DETECTED" if prediction == 1 else "❌ FALSE POSITIVE"
        
        # Create detailed output
        details = f"""
        ### Prediction: {result}
        
        **Confidence:** {confidence:.2%} if confidence else 'N/A'
        
        **Probability Breakdown:**
        - Exoplanet: {prob_exoplanet:.2%}
        - False Positive: {prob_false_positive:.2%}
        
        **Input Summary:**
        - Orbital Period: {period} days
        - Transit Depth: {depth} ppm
        - Signal-to-Noise Ratio: {snr}
        - Planet Radius: {prad} Earth radii
        - Stellar Temperature: {steff} K
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
        # 🌌 Exoplanet Detection System
        
        This AI-powered system predicts whether a transit signal represents a genuine exoplanet 
        or a false positive based on Kepler space telescope data.
        
        ### How to use:
        1. Enter the transit and stellar parameters below
        2. Click "Detect Exoplanet" to get prediction
        3. Or try example cases for quick testing
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
        
        **Note:** This is a demonstration tool. Real exoplanet confirmation requires 
        extensive follow-up observations and validation.
        """
    )

if __name__ == "__main__":
    demo.launch(share=True)