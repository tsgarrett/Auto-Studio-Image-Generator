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
                Identify the car in this image. List its: 
                - Make/Model/Year (if identifiable)
                - Primary paint color and finish
                - Specific unique features (e.g., hood scoops, stripes, wheel type, tire branding).
                Return the result as a concise, descriptive paragraph for an image generator.
                """
                
                # Using Gemini 2.5 Flash for the vision analysis
                analysis_res = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[analysis_prompt, source_img]
                )
                car_description = analysis_res.text
                st.success("Analysis complete! Generating final image...")
                st.info(f"**Extracted DNA:** {car_description}")

                # 2. Build and send the final prompt
                final_prompt = f"""
                A professional, high-key automotive cyclorama studio photograph of the following vehicle: {car_description}. 
                The vehicle is depicted in a perfectly duplicated pose, perspective, and orientation, exactly matching the vehicle in the attached reference image. 
                The original background is completely removed and replaced with the seamless white cyclorama and soft, high-key diffused overhead softbox lighting from the reference image. 
                Ensure clean, controlled reflections and a subtle, soft grounding shadow directly beneath the tires, identical to the lighting quality of the reference.
                """

                # Using a free image generation API (Pollinations) as a fallback since the current API key lacks Imagen permissions.
                # We feed it the highly-detailed prompt we just generated using Gemini.
                st.info("Sending prompt to image generator...")
                safe_prompt = urllib.parse.quote(final_prompt.strip())
                image_url = f"https://image.pollinations.ai/prompt/{safe_prompt}?width=1024&height=1024&nologo=true&enhance=false"
                
                # 3. Display the result
                st.write("### Result")
                st.image(image_url, caption="Final Studio Shot (Powered by Pollinations)", use_container_width=True)
                        
            except Exception as e:
                st.error(f"An error occurred: {e}")