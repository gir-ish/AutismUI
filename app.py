import streamlit as st
import os
import numpy as np
import torchaudio
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
import time
import streamlit.components.v1 as components

# Set GPU memory growth for TensorFlow (optional, for environments with GPUs)
try:
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
except Exception as e:
    st.warning(f"Could not set GPU memory growth: {e}")

# Load TensorFlow Hub layer
m = hub.KerasLayer('https://tfhub.dev/google/nonsemantic-speech-benchmark/trillsson4/1')

# Define TransformerEncoder (custom layer used in the model)
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

# Load the Autism detection model
@st.cache_resource
def load_autism_model():
    try:
        return load_model('autism_detection_model3.h5', custom_objects={'TransformerEncoder': TransformerEncoder})
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_autism_model()

# Function to extract features from an audio file
def extract_features(path):
    sample_rate = 16000
    array, fs = torchaudio.load(path)

    array = np.array(array)
    if array.shape[0] > 1:
        array = np.mean(array, axis=0, keepdims=True)

    # Truncate audio to 10 seconds for efficiency
    array = array[:, :sample_rate * 10]

    embeddings = m(array)['embedding']
    embeddings.shape.assert_is_compatible_with([None, 1024])
    embeddings = np.squeeze(np.array(embeddings), axis=0)

    return embeddings

# Run prediction and display results
def run_prediction(features):
    try:
        prediction = model.predict(np.expand_dims(features, axis=0))
        autism_probability = prediction[0][1]
        normal_probability = prediction[0][0]

        st.subheader("Prediction Probabilities:")
        if autism_probability > normal_probability:
            st.markdown(
                f'<div style="background-color:#658EA9;padding:20px;border-radius:10px;margin-bottom:40px;">'
                f'<h3 style="color:white;">Autism: {autism_probability:.2f}</h3>'
                '</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div style="background-color:#ADD8E6;padding:20px;border-radius:10px;margin-bottom:40px;">'
                f'<h3 style="color:black;">Normal: {normal_probability:.2f}</h3>'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div style="background-color:#658EA9;padding:20px;border-radius:10px;margin-bottom:40px;">'
                f'<h3 style="color:white;">Normal: {normal_probability:.2f}</h3>'
                '</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div style="background-color:#ADD8E6;padding:20px;border-radius:10px;margin-bottom:40px;">'
                f'<h3 style="color:black;">Autism: {autism_probability:.2f}</h3>'
                '</div>',
                unsafe_allow_html=True
            )
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Streamlit app layout
st.markdown('<span style="color:black; font-size: 48px; font-weight: bold;">Neu</span> <span style="color:black; font-size: 48px; font-weight: bold;">RO:</span> <span style="color:black; font-size: 48px; font-weight: bold;">An Application for Code-Switched Autism Detection in Children</span>', unsafe_allow_html=True)

option = st.radio("Choose an option:", ["Upload an audio file", "Record audio"])

if option == "Upload an audio file":
    uploaded_file = st.file_uploader("Upload an audio file (.wav)", type=["wav"])
    if uploaded_file is not None:
        start_time = time.time()  # Record start time
        with st.spinner('Extracting features...'):
            temp_audio_path = os.path.join(".", "temp_audio.wav")
            with open(temp_audio_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            features = extract_features(temp_audio_path)
            os.remove(temp_audio_path)
            run_prediction(features)
        elapsed_time = round(time.time() - start_time, 2)
        st.write(f"Elapsed Time: {elapsed_time} seconds")
else:
    st.markdown("### Audio Recorder")
    st.markdown("To record your audio, please **allow microphone access** in your browser.")
    components.html(
        """
        <div style="text-align: center;">
            <h1>Audio Recorder</h1>
            <button id="startRecording">Start Recording</button>
            <button id="stopRecording" disabled>Stop Recording</button>
            <div id="timer">00:00</div>
            <div id="permission-message" style="color: red; font-size: 14px;"></div>
        </div>
        <script>
            let recorder;
            let audioChunks = [];
            let startTime;
            let timerInterval;

            function updateTime() {
                const elapsedTime = Math.floor((Date.now() - startTime) / 1000);
                const minutes = Math.floor(elapsedTime / 60);
                const seconds = elapsedTime % 60;
                const formattedTime = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
                document.getElementById('timer').textContent = formattedTime;
            }

            navigator.mediaDevices.getUserMedia({ audio: true })
                .then(stream => {
                    document.getElementById('permission-message').textContent = '';
                    recorder = new MediaRecorder(stream);

                    recorder.ondataavailable = e => {
                        audioChunks.push(e.data);
                    };

                    recorder.onstart = () => {
                        startTime = Date.now();
                        timerInterval = setInterval(updateTime, 1000);
                    };

                    recorder.onstop = () => {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                        const audioUrl = URL.createObjectURL(audioBlob);
                        const a = document.createElement('a');
                        a.href = audioUrl;
                        a.download = 'recorded_audio.wav';
                        document.body.appendChild(a);
                        a.click();
                        audioChunks = [];
                        clearInterval(timerInterval);
                    };
                })
                .catch(err => {
                    document.getElementById('permission-message').textContent = 'Permission to access the microphone is required!';
                    console.error('Permission to access microphone denied:', err);
                });

            document.getElementById('startRecording').addEventListener('click', () => {
                recorder.start();
                document.getElementById('startRecording').disabled = true;
                document.getElementById('stopRecording').disabled = false;
                setTimeout(() => {
                    recorder.stop();
                    document.getElementById('startRecording').disabled = false;
                    document.getElementById('stopRecording').disabled = true;
                }, 15000); // 15 seconds
            });

            document.getElementById('stopRecording').addEventListener('click', () => {
                recorder.stop();
                document.getElementById('startRecording').disabled = false;
                document.getElementById('stopRecording').disabled = true;
            });
        </script>
        """,
        height=300,
    )
