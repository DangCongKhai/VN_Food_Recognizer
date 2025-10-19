import streamlit as st
import pandas as pd
import pickle
import torch.nn as nn
import torch
from torchvision import transforms
from PIL import Image
from dotenv import load_dotenv
import os
from google import genai
import tempfile

load_dotenv()

LABELS = ["Banh mi", "Mi Quang", "Pho"]
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
ROOT_DIR = os.path.abspath(".")
MODEL_WEIGHT_PATH = os.path.join(ROOT_DIR, "model_weights.pth")
INGREDIENTS_EXTRATION_INSTRUCTION = """
    From the given image, please describe the ingredients that you see and return them in a valid json format as followed:
    {
        "ingredients" : ['beef', 'pepper', 'egg']
    }
    Remember not to make up any ingredients that you don't find in the image
"""


class FoodCNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(32, len(LABELS)),
        )

    def forward(self, X):
        return self.sequential(X)


# Set up deep learning model
@st.cache_resource
def load_model():
    model = FoodCNN()
    state_dict = torch.load(MODEL_WEIGHT_PATH, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


model = load_model()


# Set up gemini client
@st.cache_resource
def load_gemini_client():
    client = genai.Client(api_key=GEMINI_API_KEY)
    return client


client = load_gemini_client()


def get_ingredients_from_food_image(image_file: str):
    """This function is used to extract ingredients from uploaded food image using gemini

    Args:
        image_file (str): Image file path

    Return:
        (List[str]):  A list of ingredients like ['beef', 'pepper', 'egg']}
    """
    try:

        upload_file = client.files.upload(file=image_file)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[upload_file, INGREDIENTS_EXTRATION_INSTRUCTION],
            config={"response_mime_type": "application/json"},
        )

        content = response.candidates[0].content.parts[0].text

        ingredients = eval(content)["ingredients"]
    except Exception as e:
        print("Error: ", e)
        ingredients = []
    return ingredients


# UI
st.markdown("# Vietnamse food classification and ingredients extraction")
st.markdown(
    "This web app can used to classify 3 dishes such as Pho, Mi Quang and Banh Mi. It can then extract ingredients that the model sees in the image!"
)


uploaded_file = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width="stretch")

    # Create columns for layout
    col1, col2, col3 = st.columns(3)
    with col2:
        predict_button = st.button("🔍 Predict", width="stretch")

    if predict_button:
        # Preprocess
        input_tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = nn.Softmax(dim=1)(outputs)
            top_prob, top_catid = torch.topk(probs, 1)

        pred_label = LABELS[top_catid[0].item()]
        st.success(
            f"Prediction: **{pred_label}** ({top_prob.item()*100:.2f}% confidence)"
        )
        status_placeholder = st.empty()

        # Show temporary status message
        status_placeholder.write("Currently extracting ingredients from image...")
        with tempfile.TemporaryDirectory() as tmpdir:

            img_path = os.path.join(tmpdir, uploaded_file.name)
            with open(img_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            ingredients = get_ingredients_from_food_image(img_path)
            status_placeholder.empty()
            if len(ingredients) > 0:
                st.success(
                    f"Ingredients in {pred_label}: **{', '.join(ingredients)}** "
                )
            else:
                st.error(f"Fail to extract ingredients in the image!")
