
# streamlit_app.py
import streamlit as st
import librosa
import numpy as np
from voice_recognition_for_streamlit import streamlit_voice_recognition

def main():
    st.title("Voice Recognition - Fixed Confidence Issue")
    
    # File upload
    uploaded_file = st.file_uploader("Upload audio file", type=['wav', 'mp3'])
    
    if uploaded_file is not None:
        # Load audio
        audio_data, sr = librosa.load(uploaded_file, sr=22050)
        
        # Get prediction
        result = streamlit_voice_recognition(audio_data, sr)
        
        # Display results
        if result['status'] == 'success':
            st.success(f"Speaker: {result['speaker']}")
            st.info(f"Confidence: {result['confidence']:.1%}")
            st.info(f"Quality: {result['prediction_quality']}")
            
            # Show details in expander
            with st.expander("Show Details"):
                st.json(result)
        else:
            st.error(f"Error: {result.get('error_message', 'Unknown error')}")

if __name__ == "__main__":
    main()
