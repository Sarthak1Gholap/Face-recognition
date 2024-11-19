import streamlit as st
from PIL import Image
import boto3
import os

# Title of the application
st.title('Face Recognition using AWS')

# File uploader for images
img_file = st.file_uploader('Upload Face Image', type=['png', 'jpg', 'jpeg'])

# Function to load and display an image
def load_image(img):
    return Image.open(img)

# Function to convert images to supported formats (JPEG/PNG)
def convert_to_jpg(image_path):
    try:
        img = Image.open(image_path)
        if img.format not in ['JPEG', 'PNG']:
            output_path = image_path.rsplit('.', 1)[0] + ".jpg"
            img.convert('RGB').save(output_path, "JPEG")
            return output_path
        return image_path
    except Exception as e:
        st.write(f"Error converting {image_path}: {e}")
        return None

# Function to resize images if they are too large
def resize_image(image_path):
    try:
        with Image.open(image_path) as img:
            if img.width > 4096 or img.height > 4096:  # AWS Rekognition limits
                img = img.resize((4096, 4096), Image.ANTIALIAS)
                resized_path = image_path.rsplit('.', 1)[0] + "_resized.jpg"
                img.save(resized_path, "JPEG")
                return resized_path
        return image_path
    except Exception as e:
        st.write(f"Error resizing {image_path}: {e}")
        return None

# Function to compare faces using AWS Rekognition
def compare_faces(source_path, target_path):
    client = boto3.client('rekognition', region_name='ap-south-1')  # Replace 'ap-south-1' with your AWS region
    try:
        with open(source_path, 'rb') as source_image:
            with open(target_path, 'rb') as target_image:
                response = client.compare_faces(
                    SimilarityThreshold=70,  # You can adjust the threshold if needed
                    SourceImage={'Bytes': source_image.read()},
                    TargetImage={'Bytes': target_image.read()}
                )
        return response
    except Exception as e:
        st.write(f"Error comparing {source_path} with {target_path}: {e}")
        return None

# Main logic for the Streamlit app
if img_file is not None:
    # Display file details
    file_details = {
        "name": img_file.name,
        "type": img_file.type,
        "size": img_file.size
    }
    st.write("File Details:", file_details)

    # Save the uploaded image locally
    src_path = "src.jpg"
    with open(src_path, "wb") as f:
        f.write(img_file.getbuffer())
    
    # Display the uploaded image
    st.image(load_image(img_file), caption="Uploaded Image", width=250)

    # Convert and resize the source image
    src_path = convert_to_jpg(src_path)
    if src_path:
        src_path = resize_image(src_path)

    # Folder containing the "database" of known faces
    faces_folder = "faces"  # Ensure this folder exists and contains images

    # Perform face comparison
    if os.path.exists(faces_folder):
        found_match = False  # Flag to track if a match is found
        for face_image in os.listdir(faces_folder):
            target_path = os.path.join(faces_folder, face_image)
            if os.path.isfile(target_path):  # Ensure it's a file
                st.write(f"Comparing with: {face_image}")
                
                # Convert and resize the target image
                target_path = convert_to_jpg(target_path)
                if target_path:
                    target_path = resize_image(target_path)

                # Compare faces
                response = compare_faces(src_path, target_path)
                if response:
                    for match in response.get('FaceMatches', []):
                        similarity = match['Similarity']
                        if similarity > 70:  # If similarity > 70%, we consider it a match
                            st.write(f"Match found with {face_image} with {similarity:.2f}% similarity.")
                            found_match = True
                            break
        
        if not found_match:
            st.write("No matches found in the database.")
    else:
        st.write("Faces folder not found. Please create a 'faces' folder and add images for comparison.")
