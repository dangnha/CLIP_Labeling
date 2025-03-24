import streamlit as st
import torch
import clip
from PIL import Image
import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import io
import base64
import torch.nn.functional as F
import shutil
import uuid
from transformers import BlipProcessor, BlipForConditionalGeneration

# Set page configuration
st.set_page_config(
    page_title="CLIP Vision App",
    page_icon="🖼️",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'db_images' not in st.session_state:
    st.session_state.db_images = {}  # {id: {'path': path, 'caption': caption, 'embedding': embedding}}

# Create necessary directories
DATA_DIR = "database_images"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Load CLIP model
@st.cache_resource
def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

# Load BLIP model for image captioning
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# Get image embedding
def get_image_embedding(model, preprocess, image, device):
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
    return image_features / image_features.norm(dim=-1, keepdim=True)

# Get text embedding
def get_text_embedding(model, text, device):
    text_inputs = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
    return text_features / text_features.norm(dim=-1, keepdim=True)

# Generate caption using BLIP
def generate_caption(image, blip_processor, blip_model):
    inputs = blip_processor(image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption

# Zero-shot classification
def classify_image(model, preprocess, image, device, categories):
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_inputs = clip.tokenize(categories).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarity
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
    values, indices = similarity[0].topk(min(5, len(categories)))
    
    results = []
    for value, index in zip(values, indices):
        results.append((categories[index], value.item()))
    
    return results

# Save image to database
def save_to_database(image, embedding, caption, label):
    # Generate unique ID
    img_id = str(uuid.uuid4())
    
    # Save image
    img_path = os.path.join(DATA_DIR, f"{img_id}.jpg")
    image.save(img_path)
    
    # Store in session state
    st.session_state.db_images[img_id] = {
        'path': img_path,
        'caption': caption,
        'label': label,
        'embedding': embedding.cpu().numpy(),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return img_id

# Find similar images
def find_similar_images(query_embedding, top_k=5):
    if not st.session_state.db_images:
        return []
    
    similarities = []
    for img_id, data in st.session_state.db_images.items():
        db_embedding = torch.tensor(data['embedding'])
        similarity = F.cosine_similarity(query_embedding.cpu(), db_embedding, dim=-1)
        similarities.append((img_id, similarity.item()))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top-k results
    return similarities[:top_k]

# Main function
def main():
    # Sidebar
    st.sidebar.title("CLIP Vision App")
    app_mode = st.sidebar.selectbox(
        "Choose a task", 
        ["Image Captioning", "Image Classification", "Image Retrieval", "Database Management"]
    )
    
    # Load models
    with st.spinner("Loading models..."):
        clip_model, preprocess, device = load_clip_model()
        blip_processor, blip_model = load_blip_model()
    
    # Define some common categories for classification
    default_categories = [
        "a photo of a dog", "a photo of a cat", "a photo of a car", 
        "a photo of a bird", "a photo of a person", "a landscape photo",
        "a photo of food", "a photo of a building", "a photo of a flower"
    ]
    
    # Image Captioning
    if app_mode == "Image Captioning":
        st.title("Image Captioning")
        st.write("Upload an image to generate a caption using BLIP model.")
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)
            
            with col2:
                with st.spinner("Generating caption..."):
                    # Generate caption
                    caption = generate_caption(image, blip_processor, blip_model)
                    st.write("**Caption:**", caption)
                    
                    # Get image embedding
                    embedding = get_image_embedding(clip_model, preprocess, image, device)
                    
                    # Option to save to database
                    if st.button("Save to Database"):
                        # Use first classification as label
                        label_results = classify_image(clip_model, preprocess, image, device, default_categories)
                        top_label = label_results[0][0] if label_results else "unknown"
                        
                        save_to_database(image, embedding, caption, top_label)
                        st.success("Image saved to database!")
                    
                    # Option to download captioned image
                    buffered = io.BytesIO()
                    fig, ax = plt.subplots(figsize=(10, 10))
                    ax.imshow(np.array(image))
                    ax.text(10, image.height - 30, caption, fontsize=12, color='white', 
                            bbox=dict(facecolor='black', alpha=0.7))
                    ax.axis('off')
                    plt.tight_layout()
                    plt.savefig(buffered, format="jpg")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    
                    download_link = f'<a href="data:file/jpg;base64,{img_str}" download="captioned_image.jpg">Download Image with Caption</a>'
                    st.markdown(download_link, unsafe_allow_html=True)
    
    # Image Classification
    elif app_mode == "Image Classification":
        st.title("Zero-Shot Image Classification")
        st.write("Upload an image to classify it using CLIP's zero-shot classification.")
        
        # Input for custom categories
        custom_categories = st.text_area(
            "Enter custom categories (one per line)", 
            "\n".join(default_categories)
        )
        categories = [cat.strip() for cat in custom_categories.split("\n") if cat.strip()]
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)
            
            with col2:
                with st.spinner("Classifying image..."):
                    # Classify image
                    results = classify_image(clip_model, preprocess, image, device, categories)
                    
                    st.write("**Classification Results:**")
                    
                    # Create a bar chart for visualization
                    labels = [result[0].replace("a photo of ", "") for result in results]
                    scores = [result[1] * 100 for result in results]
                    
                    fig, ax = plt.subplots()
                    bars = ax.barh(labels, scores, color='skyblue')
                    ax.set_xlabel('Confidence (%)')
                    ax.set_xlim(0, 100)
                    for i, v in enumerate(scores):
                        ax.text(v + 1, i, f"{v:.1f}%", va='center')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Get image embedding and caption
                    embedding = get_image_embedding(clip_model, preprocess, image, device)
                    caption = generate_caption(image, blip_processor, blip_model)
                    
                    # Option to save to database
                    if st.button("Save to Database"):
                        top_label = results[0][0] if results else "unknown"
                        save_to_database(image, embedding, caption, top_label)
                        st.success("Image saved to database!")
    
    # Image Retrieval
    elif app_mode == "Image Retrieval":
        st.title("Image Retrieval")
        
        retrieval_mode = st.radio("Retrieval Method", ["Image-to-Image", "Text-to-Image"])
        
        if retrieval_mode == "Image-to-Image":
            st.write("Upload an image to find similar images in the database.")
            
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                query_image = Image.open(uploaded_file).convert("RGB")
                
                st.image(query_image, caption="Query Image", width=300)
                
                with st.spinner("Finding similar images..."):
                    # Get image embedding
                    query_embedding = get_image_embedding(clip_model, preprocess, query_image, device)
                    
                    # Find similar images
                    similar_images = find_similar_images(query_embedding)
                    
                    if similar_images:
                        st.write("**Similar Images:**")
                        
                        cols = st.columns(min(len(similar_images), 3))
                        for i, (img_id, similarity) in enumerate(similar_images):
                            col = cols[i % 3]
                            img_data = st.session_state.db_images[img_id]
                            img = Image.open(img_data["path"])
                            
                            with col:
                                st.image(img, width=200)
                                st.write(f"Similarity: {similarity:.2f}")
                                st.write(f"Caption: {img_data['caption']}")
                                st.write(f"Label: {img_data['label']}")
                    else:
                        st.write("No images in database yet!")
        
        else:  # Text-to-Image
            st.write("Enter text to find similar images in the database.")
            
            query_text = st.text_input("Enter query text")
            
            if query_text and st.button("Search"):
                with st.spinner("Finding similar images..."):
                    # Get text embedding
                    query_embedding = get_text_embedding(clip_model, query_text, device)
                    
                    # Find similar images
                    similar_images = find_similar_images(query_embedding)
                    
                    if similar_images:
                        st.write("**Similar Images:**")
                        
                        cols = st.columns(min(len(similar_images), 3))
                        for i, (img_id, similarity) in enumerate(similar_images):
                            col = cols[i % 3]
                            img_data = st.session_state.db_images[img_id]
                            img = Image.open(img_data["path"])
                            
                            with col:
                                st.image(img, width=200)
                                st.write(f"Similarity: {similarity:.2f}")
                                st.write(f"Caption: {img_data['caption']}")
                                st.write(f"Label: {img_data['label']}")
                    else:
                        st.write("No images in database yet!")
    
    # Database Management
    elif app_mode == "Database Management":
        st.title("Database Management")
        
        if not st.session_state.db_images:
            st.write("Database is empty. Add images through other features.")
        else:
            st.write(f"Database contains {len(st.session_state.db_images)} images.")
            
            # Filter options
            filter_option = st.selectbox(
                "Filter by", 
                ["All", "Caption", "Label"]
            )
            
            filtered_images = {}
            
            if filter_option == "All":
                filtered_images = st.session_state.db_images
            elif filter_option == "Caption":
                search_term = st.text_input("Search in captions")
                if search_term:
                    filtered_images = {
                        img_id: data for img_id, data in st.session_state.db_images.items()
                        if search_term.lower() in data['caption'].lower()
                    }
                else:
                    filtered_images = st.session_state.db_images
            elif filter_option == "Label":
                labels = list(set(data['label'] for data in st.session_state.db_images.values()))
                selected_label = st.selectbox("Select label", ["All"] + labels)
                
                if selected_label == "All":
                    filtered_images = st.session_state.db_images
                else:
                    filtered_images = {
                        img_id: data for img_id, data in st.session_state.db_images.items()
                        if data['label'] == selected_label
                    }
            
            # Display images in grid
            cols_per_row = 3
            for i in range(0, len(filtered_images), cols_per_row):
                cols = st.columns(cols_per_row)
                
                for j in range(cols_per_row):
                    if i + j < len(filtered_images):
                        img_id = list(filtered_images.keys())[i + j]
                        img_data = filtered_images[img_id]
                        
                        with cols[j]:
                            img = Image.open(img_data["path"])
                            st.image(img, width=200)
                            st.write(f"Caption: {img_data['caption']}")
                            st.write(f"Label: {img_data['label']}")
                            st.write(f"Date: {img_data['timestamp']}")
                            
                            # Download button
                            buffered = io.BytesIO()
                            fig, ax = plt.subplots(figsize=(10, 10))
                            ax.imshow(np.array(img))
                            ax.text(10, img.height - 30, img_data['caption'], fontsize=12, color='white', 
                                    bbox=dict(facecolor='black', alpha=0.7))
                            ax.axis('off')
                            plt.tight_layout()
                            plt.savefig(buffered, format="jpg")
                            img_str = base64.b64encode(buffered.getvalue()).decode()
                            
                            download_link = f'<a href="data:file/jpg;base64,{img_str}" download="image_{img_id[:8]}.jpg">Download</a>'
                            st.markdown(download_link, unsafe_allow_html=True)
                            
                            # Delete button
                            if st.button(f"Delete", key=f"del_{img_id}"):
                                if os.path.exists(img_data["path"]):
                                    os.remove(img_data["path"])
                                del st.session_state.db_images[img_id]
                                st.experimental_rerun()
            
            # Export database
            if st.button("Export Database"):
                # Create CSV with metadata
                data = []
                for img_id, img_data in st.session_state.db_images.items():
                    data.append({
                        'id': img_id,
                        'path': img_data['path'],
                        'caption': img_data['caption'],
                        'label': img_data['label'],
                        'timestamp': img_data['timestamp']
                    })
                
                df = pd.DataFrame(data)
                csv = df.to_csv(index=False)
                csv_bytes = csv.encode()
                
                st.download_button(
                    label="Download CSV",
                    data=csv_bytes,
                    file_name="image_database.csv",
                    mime="text/csv"
                )
            
            # Clear database
            if st.button("Clear Database"):
                # Delete all files
                for img_data in st.session_state.db_images.values():
                    if os.path.exists(img_data["path"]):
                        os.remove(img_data["path"])
                
                # Clear session state
                st.session_state.db_images = {}
                st.success("Database cleared!")
                st.experimental_rerun()

if __name__ == "__main__":
    main()