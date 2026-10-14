import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import plotly.express as px

st.set_page_config(
    page_title="NeuralVision AI",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1220, #172554);
    color: white;
}
h1, h2, h3, p, label {
    color: white !important;
}
div[data-testid="stMetric"] {
    background: #1e293b;
    padding: 22px;
    border-radius: 15px;
    border: 1px solid #334155;
}
div[data-testid="stMetricValue"] {
    color: #38bdf8;
}
.stButton > button {
    background: #2563eb;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 25px;
}
.stButton > button:hover {
    background: #1d4ed8;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🧠 NeuralVision AI")
st.sidebar.write("Deep Learning Dashboard")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Digit Recognition", "About Project"]
)

@st.cache_resource
def load_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    (x_train, y_train), (x_test, y_test) = (
        tf.keras.datasets.mnist.load_data()
    )

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    model.fit(
        x_train, y_train,
        epochs=3,
        batch_size=128,
        validation_split=0.1,
        verbose=0
    )

    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return model, accuracy


if page == "Dashboard":
    st.title("🧠 NeuralVision AI")
    st.write("Handwritten Digit Recognition using Deep Learning")
    st.markdown("---")

    with st.spinner("Preparing your AI model..."):
        model, accuracy = load_model()

    c1, c2, c3 = st.columns(3)

    c1.metric("Model Accuracy", f"{accuracy * 100:.2f}%")
    c2.metric("Digit Classes", "10")
    c3.metric("Model Type", "CNN")

    st.subheader("Welcome to NeuralVision AI")
    st.write("""
    This application uses a Convolutional Neural Network (CNN)
    to recognize handwritten digits from 0 to 9.

    **How it works:**
    1. Upload a handwritten digit image.
    2. The CNN model processes the image.
    3. The model predicts the digit.
    4. View prediction confidence and probability chart.
    """)

    st.info("Select 'Digit Recognition' from the sidebar to test the model.")


elif page == "Digit Recognition":
    st.title("🔢 Digit Recognition")
    st.write("Upload an image containing one handwritten digit.")

    with st.spinner("Preparing your AI model..."):
        model, accuracy = load_model()

    uploaded_file = st.file_uploader(
        "Upload PNG or JPG",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("L")

        left, right = st.columns(2)

        with left:
            st.image(image, caption="Uploaded Image", width=250)

        processed = ImageOps.invert(image)
        processed = ImageOps.fit(processed, (28, 28))

        arr = np.array(processed).astype("float32") / 255.0
        arr = arr[np.newaxis, ..., np.newaxis]

        if st.button("✨ Predict Digit"):
            probabilities = model.predict(arr, verbose=0)[0]
            digit = int(np.argmax(probabilities))
            confidence = float(np.max(probabilities)) * 100

            with right:
                st.success(f"Predicted Digit: {digit}")
                st.metric("Confidence", f"{confidence:.2f}%")

            chart = px.bar(
                x=list(range(10)),
                y=probabilities * 100,
                labels={"x": "Digit", "y": "Probability (%)"},
                title="Prediction Probability"
            )

            chart.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )

            st.plotly_chart(chart, use_container_width=True)


else:
    st.title("📘 About the Project")

    st.subheader("Project Information")

    st.write("""
    **Project Name:** NeuralVision AI

    **Domain:** Deep Learning

    **Algorithm:** Convolutional Neural Network (CNN)

    **Dataset:** MNIST Handwritten Digits

    **Framework:** TensorFlow / Keras

    **Interface:** Streamlit

    **Programming Language:** Python
    """)

    st.subheader("Objective")
    st.write("""
    The objective is to build a deep learning application that
    recognizes handwritten digits and displays its prediction
    confidence.
    """)

    st.warning(
        "Predictions may be inaccurate for unclear images or "
        "images that differ from the MNIST dataset."
    )

st.markdown("---")
st.caption("NeuralVision AI | Deep Learning Student Project")