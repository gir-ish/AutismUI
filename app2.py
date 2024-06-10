import streamlit as st
import os
import numpy as np
import torchaudio
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
import time
import psutil
import streamlit.components.v1 as components
import random

# model_path = 'C:/Users/giris/Downloads/AutismUI/TrillsonFeature_model' 
# m = hub.load(model_path)
m = hub.KerasLayer('https://tfhub.dev/google/nonsemantic-speech-benchmark/trillsson4/1')
class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.01, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim)])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super(TransformerEncoder, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate
        })
        return config

model = load_model('C:/Users/giris/Downloads/AutismUI/autism_detection_model3.h5', custom_objects={'TransformerEncoder': TransformerEncoder})

def extract_features(path):
    sample_rate = 16000
    array, fs = torchaudio.load(path)

    array = np.array(array)
    if array.shape[0] > 1:
        array = np.mean(array, axis=0, keepdims=True)

    embeddings = m(array)['embedding']
    embeddings.shape.assert_is_compatible_with([None, 1024])
    embeddings = np.squeeze(np.array(embeddings), axis=0)

    return embeddings
# st.markdown('Streamlit is **_really_ cool**.')
st.markdown('<span style="color:black; font-size: 48px; font-weight: bold;">Neu</span> <span style="color:black; font-size: 48px; font-weight: bold;">RO:</span> <span style="color:black; font-size: 48px; font-weight: bold;">An Application for Code-Switched Autism Detection in Children</span>', unsafe_allow_html=True)
# def set_background():
#     # HTML code with CSS styling for background image
#     page_bg_img = '''
#     <style>
#     body {
#         background-image: url("https://example.com/background_image.jpg");
#         background-size: cover;
#     }
#     </style>
#     '''
#     st.markdown(page_bg_img, unsafe_allow_html=True)

# # Call the function to set background image
# set_background()

# def random_color():
#     r = random.randint(0, 255)
#     g = random.randint(0, 255)
#     b = random.randint(0, 255)
#     return f'rgb({r}, {g}, {b})'

# text = "NeuRO: An Application for Code-Switched Autism Detection in Children"
# words = text.split()

# styled_text = ""
# for word in words:
#     color = random_color()
#     styled_text += f'<span style="color:{color}; font-size: 48px; font-weight: bold;">{word}</span> '

# st.markdown(styled_text, unsafe_allow_html=True)

#st.title('NeuRO: An Application for Code-Switched Autism Detection in Children')

# option = st.radio("Choose an option:", ("Upload an audio file", "Record audio"))
option = st.radio("**Choose an option:**", ["Upload an audio file", "Record audio"])

if option == "Upload an audio file":
    uploaded_file = st.file_uploader("Upload an audio file (.wav)", type=["wav"])
    if uploaded_file is not None:
        start_time = time.time()  # Record start time
        with st.spinner('Extracting features...'):
            # Process the uploaded file
            with open("temp_audio.wav", "wb") as f:
                f.write(uploaded_file.getbuffer())
            features = extract_features("temp_audio.wav")
            os.remove("temp_audio.wav")

            # Display prediction probabilities
            prediction = model.predict(np.expand_dims(features, axis=0))
            autism_probability = prediction[0][1]
            normal_probability = prediction[0][0]

            st.subheader("Prediction Probabilities:")

            if autism_probability > normal_probability:
                st.markdown(
                    f'<div style="background-color:#658EA9;padding:20px;border-radius:10px;margin-bottom:40px;">'
                    f'<h3 style="color:black;">Autism: {autism_probability}</h3>'
                    '</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<div style="background-color:#ADD8E6;padding:20px;border-radius:10px;margin-bottom:40px;">'
                    f'<h3 style="color:black;">Normal: {normal_probability}</h3>'
                    '</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div style="background-color:#658EA9;padding:20px;border-radius:10px;margin-bottom:40px;">'
                    f'<h3 style="color:black;">Normal: {normal_probability}</h3>'
                    '</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<div style="background-color:#ADD8E6;padding:20px;border-radius:10px;margin-bottom:40px;">'
                    f'<h3 style="color:black;">Autism: {autism_probability}</h3>'
                    '</div>',
                    unsafe_allow_html=True
                )

        elapsed_time = round(time.time() - start_time, 2)
        st.write(f"Elapsed Time: {elapsed_time} seconds")

else:  # Option is "Record audio"
    #st.write('<iframe src="http://localhost:8000/index.html" width="300" height="250"></iframe>', unsafe_allow_html=True)
    st.components.v1.iframe("http://localhost:8000/index.html", width=500, height=250)

    if st.button("Click to Predict"):
        # Run the ffmpeg command to convert the recorded audio
        os.system('ffmpeg -i C:/Users/giris/Downloads/recorded_audio.wav -acodec pcm_s16le -ar 16000 -ac 1 C:/Users/giris/Downloads/recorded_audio2.wav')
        
        # Process the converted audio file
        features = extract_features("C:/Users/giris/Downloads/recorded_audio2.wav")

        # Display prediction probabilities
        prediction = model.predict(np.expand_dims(features, axis=0))
        autism_probability = prediction[0][1]
        normal_probability = prediction[0][0]

        st.subheader("Prediction Probabilities:")

        if autism_probability > normal_probability:
            st.markdown(
                f'<div style="background-color:#658EA9;padding:20px;border-radius:10px;margin-bottom:40px;">'
                f'<h3 style="color:black;">Autism: {autism_probability}</h3>'
                '</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div style="background-color:#ADD8E6;padding:20px;border-radius:10px;margin-bottom:40px;">'
                f'<h3 style="color:black;">Normal: {normal_probability}</h3>'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div style="background-color:#658EA9;padding:20px;border-radius:10px;margin-bottom:40px;">'
                f'<h3 style="color:black;">Normal: {normal_probability}</h3>'
                '</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div style="background-color:#ADD8E6;padding:20px;border-radius:10px;margin-bottom:40px;">'
                f'<h3 style="color:black;">Autism: {autism_probability}</h3>'
                '</div>',
                unsafe_allow_html=True
            )

        # Try to delete the first audio file
        try:
            os.remove("C:/Users/giris/Downloads/recorded_audio.wav")
        except Exception as e:
            print(f"Error deleting 'recorded_audio.wav': {e}")

        # Try to delete the second audio file
        try:
            os.remove("C:/Users/giris/Downloads/recorded_audio2.wav")
        except Exception as e:
            print(f"Error deleting 'recorded_audio2.wav': {e}")
