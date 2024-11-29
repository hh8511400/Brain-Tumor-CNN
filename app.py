import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

class TumorClassifier(nn.Module):
    def __init__(self, num_classes):
        super(TumorClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

@st.cache_resource  
def load_model():
    model = TumorClassifier(num_classes=4)
    model = torch.load("/media/husnain/nain/Programming/Python/UNI/Brain-Tumor/best-model-brain-tumor.pt")  
    model.eval()  
    return model

CLASS_LABELS = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),         
        transforms.Normalize([0.5], [0.5])  
    ])
    return transform(image).unsqueeze(0)  



def main():
    st.title("Brain Tumor Classification")
    st.write("Upload an MRI image to classify it into one of the following categories:")
    st.write(", ".join(CLASS_LABELS))


    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image",  use_container_width =True)
        input_tensor = preprocess_image(image)

        model = load_model()

        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            predicted_class = CLASS_LABELS[predicted.item()]

        st.write(f"### Predicted Category: **{predicted_class}**")

if __name__ == "__main__":
    main()
