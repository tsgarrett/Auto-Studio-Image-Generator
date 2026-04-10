import streamlit as st
from google import genai
from google.genai import types
import urllib.parse
import io
import PIL.Image

# Access the API key securely from Streamlit's secrets
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

st.set_page_config(page_title="Auto Studio Generator", layout="centered")

st.title("🚗 Auto Studio Image Generator")
st.write("Upload a source car and a reference image to transfer the studio lighting and pose!")

# Layout for uploading images side-by-side
col1, col2 = st.columns(2)
with col1:
    source_upload = st.file_uploader("1. Upload Source Car Image", type=["jpg", "jpeg", "png"])
with col2:
    reference_upload = st.file_uploader("2. Upload Reference Studio Image", type=["jpg", "jpeg", "png"])

if source_upload and reference_upload:
    # Display the uploaded images
    st.write("### Inputs")
    col3, col4 = st.columns(2)
    with col3:
        st.image(source_upload, caption="Source Car", use_container_width=True)
    with col4:
        st.image(reference_upload, caption="Reference Style", use_container_width=True)

    if st.button("Generate Studio Shot", type="primary"):
        with st.spinner("Analyzing the source car..."):
            try:
                # Open the images
                source_img = PIL.Image.open(source_upload)
                reference_img = PIL.Image.open(reference_upload)

                # 1. Analyze the car's "DNA"
                analysis_prompt = """
                Analyze this source car image and extract its visual DNA.
                Describe:
                - The base Make and Model (CRITICAL: Avoid using specific trim package names like 'Mach 1', 'GT', or 'SS' as these cause AI to hallucinate factory features. Just use the base model, e.g., '1970 Ford Mustang').
                - The primary paint color and finish.
                - The explicit headlight and grille configuration (e.g., "horizontally side-by-side dual headlights" vs "vertically stacked headlights"). This is critical to prevent the image generator from using the wrong model year's front fascia!
                - Physically describe ONLY the visible decals, stripes, aero pieces (spoilers, splitters), and wheel types.
                
                CRITICAL: YOU MUST NOT hallucinate or guess any features that are not explicitly visible in this specific photo. Do not assume factory defaults. Return the result as a concise, descriptive paragraph.
                """
                
                # Using Gemini 2.5 Pro for the vision analysis (much higher adherence)
                analysis_res = client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=[analysis_prompt, source_img]
                )
                car_description = analysis_res.text
                
                # 2. Analyze the Reference Image's Pose and Lighting
                pose_prompt = """
                Analyze this reference automotive photograph. Your goal is to precisely extract the camera positioning and vehicle orientation so it can be perfectly replicated.
                Describe:
                1. The direction the car's nose is pointing relative to the image frame (e.g., "The car is pointing towards the right edge of the image frame", "The car is pointing towards the left edge of the image frame"). CRITICAL: DO NOT use terms like 'front-left profile' or 'front-right profile' as image generators misinterpret these based on the driver's seat. Only describe which way the nose points in the frame.
                2. The exact camera elevation/pitch relative to the car (Pay minute attention to this: are we looking slightly down at the hood and roof, or is it a low angle? Describe the exact vantage point height).
                3. The studio lighting setup and environment.
                Be obsessively specific about maintaining these minute spatial details and camera height. Return a concise paragraph.
                """
                pose_res = client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=[pose_prompt, reference_img]
                )
                pose_description = pose_res.text

                st.success("Analysis complete! Generating final image...")
                st.info(f"**Extracted DNA:** {car_description}\n\n**Target Pose & Lighting:** {pose_description}")

                # 3. Build and send the final prompt
                final_prompt = f"""
                A professional, photorealistic automotive studio photograph of the following vehicle: 
                {car_description}
                
                CRITICAL EXACT MATCH INSTRUCTION:
                You MUST replicate the minute spatial details of the camera and car orientation exactly as described below. Pay absolute strict attention to the EXACT camera elevation, pitch, and the vehicle's yaw/facing direction:
                {pose_description}
                
                CRITICAL FEATURE INSTRUCTION:
                DO NOT add any spoilers, decals, stripes, factory trim packages, scoops, or modifications that are not explicitly written in the vehicle description above. The vehicle must remain exactly as described physically.
                
                Ensure the background is a seamless studio cyclorama. Do not deviate from the specified camera angle, elevation, or lighting setup.
                """

                # Using Imagen 4 for generating the final image
                # Note: This requires a Google Cloud billing account linked to your API key
                response = client.models.generate_images(
                    model="imagen-4.0-ultra-generate-001",
                    prompt=final_prompt,
                    config=types.GenerateImagesConfig(
                        number_of_images=1,
                        output_mime_type="image/jpeg"
                    )
                )
                
                # 3. Display the result
                for generated_image in response.generated_images:
                    final_image = PIL.Image.open(io.BytesIO(generated_image.image.image_bytes))
                    st.write("### Result")
                    st.image(final_image, caption="Final Studio Shot", use_container_width=True)
                        
            except Exception as e:
                st.error(f"An error occurred: {e}")