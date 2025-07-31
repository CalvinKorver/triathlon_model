import streamlit as st
from PIL import Image
import os

st.title("🏊‍♀️🚴‍♀️🏃‍♀️ Triathlon Stage Classifier")

# Check if model exists
model_path = 'models/triathlon_stage_model.pkl'
if not os.path.exists(model_path):
    st.error("Model file not found! Please ensure triathlon_stage_model.pkl is in the models/ directory.")
    st.stop()

# Load FastAI model
model_loaded = False
learn = None

try:
    from fastai.vision.all import load_learner
    
    # Load model (cache for performance)
    @st.cache_resource
    def load_model():
        return load_learner(model_path)
    
    learn = load_model()
    model_loaded = True
    st.success("🎉 FastAI model loaded successfully!")
    
except ImportError as e:
    st.error(f"FastAI import error: {e}")
    st.info("Please install FastAI with: `mamba install -c conda-forge fastai`")
    model_loaded = False
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.info("Make sure you're using the correct .pkl file exported from FastAI")
    model_loaded = False

uploaded_file = st.file_uploader("Choose a triathlon image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    if model_loaded and learn is not None:
        try:
            # Convert image to RGB if needed (in case it's RGBA or grayscale)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            st.info(f"Image info: Mode={image.mode}, Size={image.size}")
            
            # Make prediction with FastAI
            pred_class, pred_idx, probs = learn.predict(image)
            
            st.write(f"**Prediction: {pred_class}**")
            st.write(f"**Confidence: {probs[pred_idx]:.1%}**")
            
            # Show confidence for all classes
            st.write("**All class probabilities:**")
            for i, class_name in enumerate(learn.dls.vocab):
                confidence = probs[i]
                st.write(f"  {class_name}: {confidence:.1%}")
                
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.error("Debug info:")
            st.write(f"- Image mode: {image.mode}")
            st.write(f"- Image size: {image.size}")
            st.write(f"- Model vocab: {learn.dls.vocab}")
            st.write(f"- Model vocab length: {len(learn.dls.vocab)}")
            
            # Show full traceback for debugging
            import traceback
            st.code(traceback.format_exc())
    else:
        st.warning("Model not loaded. Cannot make predictions.")
        st.info("To enable predictions, install FastAI: `mamba install -c conda-forge fastai`")