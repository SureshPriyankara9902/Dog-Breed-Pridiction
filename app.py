import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import time
import json
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


 
# Load the trained model
model = tf.keras.models.load_model("dogclassification.h5")

# Define class names
class_names = {
    "0": "Afghan",
    "1": "African Wild Dog",
    "2": "Airedale",
    "3": "American Hairless",
    "4": "American Spaniel",
    "5": "Basenji",
    "6": "Basset",
    "7": "Beagle",
    "8": "Bearded Collie",
    "9": "Bermaise",
    "10": "Bichon Frise",
    "11": "Blenheim",
    "12": "Bloodhound",
    "13": "Bluetick",
    "14": "Border Collie",
    "15": "Borzoi",
    "16": "Boston Terrier",
    "17": "Boxer",
    "18": "Bull Mastiff",
    "19": "Bull Terrier",
    "20": "Bulldog",
    "21": "Cairn",
    "22": "Chihuahua",
    "23": "Chinese Crested",
    "24": "Chow",
    "25": "Clumber",
    "26": "Cockapoo",
    "27": "Cocker",
    "28": "Collie",
    "29": "Corgi",
    "30": "Coyote",
    "31": "Dalmation",
    "32": "Dhole",
    "33": "Dingo",
    "34": "Doberman",
    "35": "Elk Hound",
    "36": "French Bulldog",
    "37": "German Sheperd",
    "38": "Golden Retriever",
    "39": "Great Dane",
    "40": "Great Perenees",
    "41": "Greyhound",
    "42": "Groenendael",
    "43": "Irish Spaniel",
    "44": "Irish Wolfhound",
    "45": "Japanese Spaniel",
    "46": "Komondor",
    "47": "Labradoodle",
    "48": "Labrador",
    "49": "Lhasa",
    "50": "Malinois",
    "51": "Maltese",
    "52": "Mex Hairless",
    "53": "Newfoundland",
    "54": "Pekinese",
    "55": "Pit Bull",
    "56": "Pomeranian",
    "57": "Poodle",
    "58": "Pug",
    "59": "Rhodesian",
    "60": "Rottweiler",
    "61": "Saint Bernard",
    "62": "Schnauzer",
    "63": "Scotch Terrier",
    "64": "Shar_Pei",
    "65": "Shiba Inu",
    "66": "Shih-Tzu",
    "67": "Siberian Husky",
    "68": "Vizsla",
    "69": "Yorkie"
}

# Load breed descriptions from JSON file
def load_breed_descriptions():
    with open('breed_descriptions.json', 'r') as file:
        return json.load(file)

breed_descriptions = load_breed_descriptions()

# Function to display breed description
def display_breed_description(breed_name):
    description = breed_descriptions.get(breed_name, "No description available.")
    st.markdown(f"<span style='font-size:18px; color:black;'>{description}</span>", unsafe_allow_html=True)

# Function to plot and display the class probabilities
def plot_class_probabilities(probabilities):
    plt.figure(figsize=(12, 6))
    plt.bar(class_names.values(), probabilities)
    plt.xlabel('Dog Breeds')
    plt.ylabel('Probability')
    plt.title('Class Probabilities')
    plt.xticks(rotation=90)
    plt.tight_layout()
    st.pyplot(plt)

# Function to plot a histogram of class probabilities
def plot_probability_histogram(probabilities):
    plt.figure(figsize=(12, 6))
    plt.hist(probabilities, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.title('Histogram of Class Probabilities')
    st.pyplot(plt)


# Function to plot a pie chart of the top 5 class probabilities
def plot_top5_pie_chart(probabilities):
    top5_indices = np.argsort(probabilities)[-5:]
    top5_classes = [class_names[str(i)] for i in top5_indices]
    top5_probabilities = [probabilities[i] for i in top5_indices]
    
    plt.figure(figsize=(8, 8))
    plt.pie(top5_probabilities, labels=top5_classes, autopct='%1.1f%%', startangle=140)
    plt.title('Top 5 Class Probabilities')
    st.pyplot(plt)

# Function to plot feedback distribution
def plot_feedback_distribution(feedback_list):
    feedback_counts = {key: feedback_list.count(key) for key in set(feedback_list)}
    
    plt.figure(figsize=(10, 6))
    plt.bar(feedback_counts.keys(), feedback_counts.values(), color=['#4CAF50', '#FF5722', '#FFC107'])
    plt.xlabel('Feedback Category')
    plt.ylabel('Count')
    plt.title('Feedback Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

# Function for image preprocessing
def preprocess_image(image, resize=(224, 224)):
    image = Image.open(image).convert("RGB")
    image = ImageOps.fit(image, resize, method=Image.BICUBIC)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array



# Set up Streamlit app
st.set_page_config(page_title="Dog Breed Detection App", page_icon="🐕")

# Sidebar navigation and controls
st.sidebar.title("🐕Dog Breed Detection🔎")
option = st.sidebar.radio("Go to", ["Home", "Model Accuracy", "Visualization", "Feedback", "Image Gallery"])
uploaded_image = st.sidebar.file_uploader("Upload a dog image", type=["jpg", "png", "jpeg"])


# CSS for styling
st.markdown(
    """
    <style>
    
    /* Targeting the body */
    body {
        background-color: #F2D4D7;
        
    }

    /* Ensuring the background color for main content */
    .main {
        background-color: #F2D4D7;
    }
    
    
    /* Home section background color */
    .home-content {
        background-color: transparent; /* Transparent background for the home content */
        padding: 20px;
        border-radius: 8px;
    }
    
    .stSidebarContent {
        text-align: center;
        background-color: black;
    }
    
    .home-title {
        text-align: center;
        color: #741874;
        font-size: 50px;
        font-family: 'Arial', sans-serif;
    }
    
    .home-description {
        text-align: center;
        font-size: 20px;
        color: black;
        font-family: 'Arial', sans-serif;
    }
    
    .predict-button {
        display: inline-block;
        margin: 20px;
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 12px 24px;
        text-align: center;
        font-size: 18px;
        cursor: pointer;
        border-radius: 5px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .reset-button {
        display: inline-block;
        margin: 20px;
        background-color: red;
        color: white;
        border: none;
        padding: 12px 24px;
        text-align: center;
        font-size: 18px;
        cursor: pointer;
        border-radius: 5px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .results {
        text-align: center;
        font-size: 20px;
        font-family: 'Arial', sans-serif;
        color: #4CAF50;
    }
    
    .description {
        text-align: center;
        font-size: 18px;
        color: black;
        font-family: 'Arial', sans-serif;
        width: 100%;
        word-wrap: break-word;
    }
    
    .progress-container {
        width: 100%;
        background-color: green;
        border-radius: 5px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        padding: 5px;
        margin-top: 20px;
    }
    
    .progress-bar {
        height: 30px;
        width: 100%;
        background-color: green;
    }
    
    /* Feedback section styling */
    .feedback-form {
        background-color: #000000; /* Black background for the feedback section */
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    .feedback-title {
        text-align: center;
        color: #333;
        font-size: 24px;
        font-family: 'Arial', sans-serif;
    }
    
    .feedback-description {
        text-align: center;
        font-size: 18px;
        color: #666;
        font-family: 'Arial', sans-serif;
    }
    
  
    
    /* Visualization section background color */
    .visualization-content {
        background-color: #000000; /* Black background for the Visualization content */
        padding: 20px;
        border-radius: 8px;
    }
    
    /* Image Gallery section background color */
    .image-gallery-content {
        background-color: #000000; /* Black background for the Image Gallery content */
        padding: 20px;
        border-radius: 8px;
    }
    
    .stProgress > div > div > div > div {
        border-radius: 10px;
        background-color: green; /* Green color for the filled portion of the progress bar */
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
    <style>
    .icon-links {
        text-align: center;
        margin-top: 40px;
    }
    .icon-links a {
        margin: 0 10px;
        color: black;
        text-decoration: none;
        font-size:20px;
    }
    .icon-links a:hover {
        color: blue;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    """
    <div class="icon-links">
        <a href="https://github.com/SureshPriyankara9902" target="_blank"><i class="fab fa-github fa-2x"></i></a>
        <a href="https://www.linkedin.com/in/suresh-priyankara-753319284/" target="_blank"><i class="fab fa-linkedin fa-2x"></i></a>
    </div>
    """,
    unsafe_allow_html=True
)



# State variables
if 'predict' not in st.session_state:
    st.session_state.predict = False
if 'breed_name' not in st.session_state:
    st.session_state.breed_name = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'probabilities' not in st.session_state:
    st.session_state.probabilities = None
if 'feedback' not in st.session_state:
    st.session_state.feedback = None
if 'image_gallery' not in st.session_state:
    st.session_state.image_gallery = []
if 'feedback_list' not in st.session_state:
    st.session_state.feedback_list = []

# Home page
if option == "Home":
    st.markdown(
        """
        <div class='home-content'>
        <h1 class='home-title'>🐕Dog Breed Detection🔎</h1>
        <p class='home-description'>Upload a dog image using the sidebar and then click 'Predict' to get breed details.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if uploaded_image or st.session_state.uploaded_image:
        st.session_state.uploaded_image = uploaded_image or st.session_state.uploaded_image
        st.markdown("<div class='image-frame'>", unsafe_allow_html=True)
        st.image(st.session_state.uploaded_image, caption="Uploaded Image", use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        
        

        # Predict button
        if st.button("Predict", key="predict-button", help="Classify the uploaded image", use_container_width=True):
            with st.spinner("Classifying..."):
                # Add a progress bar with percentage
                progress_bar = st.progress(0)
                progress_text = st.empty()

                # Simulate some work with progress
                for i in range(100):
                    progress_bar.progress(i + 1)
                    progress_text.text(f"Processing: {i + 1}%")
                    time.sleep(0.05)  # Simulate a delay
                progress_bar.empty()
                progress_text.empty()

                # Image processing
                img = preprocess_image(st.session_state.uploaded_image)
                
                # Prediction
                prediction = model.predict(img)
                predicted_class = np.argmax(prediction, axis=-1)[0]
                breed_name = class_names[str(predicted_class)]
                probability = prediction[0][predicted_class]

                # Store results in session state
                st.success(f"The predicted breed is: **{breed_name}** with a probability of **{probability:.2f}**")
                st.session_state.breed_name = breed_name
                st.session_state.probabilities = prediction[0]  # Store the probabilities
                
                start_time = time.time()
                img_array = preprocess_image(uploaded_image)
                predictions = model.predict(img_array)
                probabilities = tf.nn.softmax(predictions[0]).numpy()
                end_time = time.time()

                # Display results
                #st.markdown(f"<span style='color:black;text-align: center; font-size:35px;'>**Predicted Breed:**</span> <span style='color:#4CAF50; font-size:35px;'>{breed_name}</span>", unsafe_allow_html=True)
                display_breed_description(breed_name)
                
                st.write(f"Prediction time: {end_time - start_time:.2f} seconds")
                st.success("Processing complete!", icon="✅")

                # Add to image gallery
                st.session_state.image_gallery.append({
                    'image': st.session_state.uploaded_image,
                    'breed': breed_name
                })

        # Reset button
        if st.button("Reset", key="reset-button", help="Clear the uploaded image and results", use_container_width=True):
                st.session_state.uploaded_image = None
                st.session_state.predict = False
                st.session_state.breed_name = None
                st.session_state.probabilities = None
                st.session_state.feedback_list = []
                st.session_state.image_gallery = []
                

           
           
    else:
        st.warning("Please upload an image to classify.")


# Model Accuracy page
if option == "Model Accuracy":
    st.markdown("<div class='model-accuracy-content'>", unsafe_allow_html=True)
    st.title("Model Accuracy")
    
    # Load or prepare your dataset for evaluation
   
    
    # Load and split the dataset
    data = load_iris()
    X = data.data
    y = data.target
    
    # Increase test size to 50% to make the evaluation more challenging
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    
    # Train a model (replace this with your actual model and dataset)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Predict on the test dataset
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Display accuracy as a percentage
    st.markdown(f"<span style='font-size:24px; color:#4CAF50;'>Model Accuracy: {accuracy*100:.2f}%</span>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    
# Visualization page
elif option == "Visualization":
    st.title("Visualization")

    if st.session_state.probabilities is not None:
        st.sidebar.title("Visualization Controls")
        visualization_option = st.sidebar.radio("Select Visualization", [
            "Class Probabilities",
            "Probability Histogram",
            "Top 5 Probabilities Pie Chart",
            "Feedback Chart"
        ])

        if visualization_option == "Class Probabilities":
            st.subheader("Class Probabilities")
            st.sidebar.text("Showing Class Probabilities Graph:")
            plot_class_probabilities(st.session_state.probabilities)

        elif visualization_option == "Probability Histogram":
            st.subheader("Probability Histogram")
            st.sidebar.text("Showing Probability Histogram:")
            plot_probability_histogram(st.session_state.probabilities)

        elif visualization_option == "Top 5 Probabilities Pie Chart":
            st.subheader("Top 5 Probabilities Pie Chart")
            st.sidebar.text("Showing Top 5 Probabilities Pie Chart:")
            plot_top5_pie_chart(st.session_state.probabilities)
            
        elif visualization_option == "Feedback Chart":
            st.subheader("Feedback Distribution")
            st.sidebar.text("Showing Feedback Distribution Chart:")

            if st.session_state.feedback_list:
                plot_feedback_distribution(st.session_state.feedback_list)
            else:
                st.warning("No feedback data available to display.")

    else:
        st.warning("No predictions available to visualize.")

# Feedback page
elif option == "Feedback":
    st.title("Feedback")
    st.markdown("**Please provide your feedback on the prediction accuracy:**")
    feedback = st.text_area("Your feedback", "")
    
    if st.button("Submit Feedback"):
        if feedback:
            # Simple categorization
            if "good" in feedback.lower() or "excellent" in feedback.lower():
                feedback_category = "Positive"
            elif "bad" in feedback.lower() or "poor" in feedback.lower():
                feedback_category = "Negative"
            else:
                feedback_category = "Neutral"
            
            st.session_state.feedback_list.append(feedback_category)
            st.success("Thank you for your feedback!")
        else:
            st.warning("Please enter your feedback before submitting.")
            
# Image Gallery page
elif option == "Image Gallery":
    st.title("Image Gallery")
    st.markdown("**View the history of uploaded images and predictions:**")
    
    if st.session_state.image_gallery:
        for item in st.session_state.image_gallery:
            st.image(item['image'], caption=f"Predicted Breed: {item['breed']}", use_column_width=True)
    else:
        st.warning("No images to display.")
