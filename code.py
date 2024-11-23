import streamlit as st
from PIL import Image
import pytesseract
import pyttsx3
import os
import google.generativeai as genai  # Google Generative AI
import base64
from io import BytesIO
import numpy as np
import cv2

# LangChain imports
from langchain.chat_models import ChatOpenAI  # LangChain Chat Model
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool, AgentType

# Set up API Key for Google Generative AI
genai.configure(api_key="AIzaSyBBHCAvlrZPYKBoHUhgJJXeUsXE6j1RII4")

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

# Streamlit page setup with custom title
st.set_page_config(page_title="AssistAI", layout="centered")

# Custom Title with Logo and Compact Layout
st.markdown(f"""
    <style>
        .title {{
            font-family: 'Arial', sans-serif;
            font-size: 32px;
            color: #3498db;
            text-align: center;
            font-weight: bold;
            background-color: #ecf0f1;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }}
        .footer {{
            font-family: 'Arial', sans-serif;
            font-size: 12px;
            text-align: center;
            margin-top: 10px;
            color: #7f8c8d;
        }}
        .buttons-container {{
            display: flex;
            justify-content: space-evenly;
            gap: 5px;
            margin-bottom: 10px;
        }}
    </style>
    <div class="title">
        AssistAI - AI Assistant for Visually Impaired
    </div>
""", unsafe_allow_html=True)

# Sidebar setup
st.sidebar.title("🔧 Features")
st.sidebar.markdown("""
- Scene Understanding
- Text-to-Speech
- Object & Obstacle Detection
- Personalized Assistance
""")

# Function to extract text from image using OCR
def extract_text_from_image(image):
    """Extracts text from the given image using OCR."""
    text = pytesseract.image_to_string(image)
    return text

# Function for Text-to-Speech conversion
def text_to_speech(text):
    """Converts the given text to speech."""
    engine.say(text)
    engine.runAndWait()

# Function to convert an image to base64
def image_to_base64(image):
    """Converts the image to base64 encoding for sending to the API."""
    image = image.convert("RGB")
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Improved function to generate scene description
def generate_scene_description(input_prompt, image_data):
    """Generates a scene description from an image, handling both direct images and base64-encoded images."""
    try:
        if isinstance(image_data, str) and image_data.startswith("data:image"):
            image_data = base64.b64decode(image_data.split(',')[1])
            image = Image.open(BytesIO(image_data))
        elif isinstance(image_data, bytes):
            image = Image.open(BytesIO(image_data))
        else:
            image = image_data

        model = genai.GenerativeModel()  
        response = model.generate_content([input_prompt, image_data])
        return response.text
    except Exception as e:
        return f"Error: {e}"

# Object and Obstacle Detection function
def detect_objects(image_file):
    try:
        image_array = np.array(Image.open(image_file))
        img = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

        with open("coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]

        height, width, _ = img.shape
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        detected_objects = []
        if len(indexes) > 0:
            for i in indexes.flatten():
                label = str(classes[class_ids[i]])
                detected_objects.append(label)
                x, y, w, h = boxes[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        output_path = "output.jpg"
        cv2.imwrite(output_path, img)
        return detected_objects, output_path
    except Exception as e:
        return f"Error in object detection: {e}", None

# Function to improve Personalized Assistance
def provide_personalized_assistance(extracted_text, detected_objects):
    """Provides personalized assistance based on detected text and objects."""
    assistance = ""

    if extracted_text.strip():  # If text is detected, provide the extracted text as assistance
        assistance += f"Text detected in the image: {extracted_text}\n"
    else:
        assistance += "No text detected in the image.\n"

    if detected_objects:
        objects_description = ", ".join(detected_objects)
        assistance += f"Detected objects: {objects_description}.\n"
    else:
        assistance += "No objects detected in the image.\n"

    return assistance

# Main functionality of the Streamlit app
def main():
    st.text("Upload an image to get AI-powered assistance.")
    
    # Layout for buttons before image upload
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        describe_button = st.button("🏞️ Describe Scene")
    with col2:
        read_text_button = st.button("📝 Read Text")
    with col3:
        detect_objects_button = st.button("🔍 Detect Objects")
    with col4:
        personalized_assistance_button = st.button("💬 Personalized Assistance")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

    # Check button functionality
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        # Show functionality based on button clicks
        if describe_button:
            st.write("Analyzing the image...")
            image_data = Image.open(uploaded_file)
            description = generate_scene_description("Describe the scene", image_data)
            st.write(f"Scene Description: {description}")

        if read_text_button:
            st.write("Extracting text from the image...")
            image_data = Image.open(uploaded_file)
            extracted_text = extract_text_from_image(image_data)
            st.write(f"Extracted Text: {extracted_text}")
            st.write("Converting text to speech...")
            text_to_speech(extracted_text)

        if detect_objects_button:
            st.write("Detecting objects in the image...")
            detected_objects, output_path = detect_objects(uploaded_file)
            if output_path:
                st.image(output_path, caption="Objects Detected", use_container_width=True)
            st.write(f"Detected Objects: {', '.join(detected_objects) if detected_objects else 'None'}")

        if personalized_assistance_button:
            st.write("Analyzing the image for personalized assistance...")
            image_data = Image.open(uploaded_file)
            extracted_text = extract_text_from_image(image_data)
            detected_objects, _ = detect_objects(uploaded_file)
            assistance = provide_personalized_assistance(extracted_text, detected_objects)
            st.write(f"Personalized Assistance: {assistance}")

    # Footer with "Project made by Nithin Appala"
    st.markdown("<footer class='footer'>Project made by Nithin Appala</footer>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
