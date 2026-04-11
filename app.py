import streamlit as st
from google import genai
import fal_client
import urllib.parse
import io
import PIL.Image
import os

# Access API keys from Streamlit secrets
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# fal_client reads FAL_KEY from environment automatically
os.environ["FAL_KEY"] = st.secrets["FAL_KEY"]

st.set_page_config(page_title="Auto Studio Generator", layout="centered")

st.title("🚗 Auto Studio Image Generator")
st.write("Upload a source car and a reference image to transfer the studio lighting and pose!")

col1, col2 = st.columns(2)
with col1:
    source_upload = st.file_uploader("1. Upload Source Car Image", type=["jpg", "jpeg", "png"])
with col2:
    reference_upload = st.file_uploader("2. Upload Reference Studio Image", type=["jpg", "jpeg", "png"])

if source_upload and reference_upload:
    st.write("### Inputs")
    col3, col4 = st.columns(2)
    with col3:
        st.image(source_upload, caption="Source Car", use_container_width=True)
    with col4:
        st.image(reference_upload, caption="Reference Style", use_container_width=True)

    if st.button("Generate Studio Shot", type="primary"):
        with st.spinner("Analyzing the source car..."):
            try:
                source_img = PIL.Image.open(source_upload)
                reference_img = PIL.Image.open(reference_upload)

                # ── Step 1: Analyze the car's visual DNA ─────────────────────────────
                analysis_prompt = """
                Analyze this source car image and extract its visual DNA.
                - A highly descriptive but generic vehicle baseline (CRITICAL: You MUST NOT name the actual vehicle brand, make, or model. Do not say "Pontiac", "Bonneville", or "Mustang". Naming the specific car triggers 2-door muscle car hallucinations in the final image generator. Instead, use a generic physical descriptor like "A large, luxurious 1960s classic American 4-door sedan").
                - The explicit body style (e.g., 2-door coupe vs 4-door sedan/hardtop) based on the visible door cutlines. If it has 4 doors, emphasize it heavily.
                - The hood geometry (e.g., completely flat hood with NO scoops). Explicitly state if it lacks hood scoops.
                - The primary paint color and finish.
                - The explicit headlight and grille configuration (e.g., "horizontally side-by-side dual headlights" vs "vertically stacked headlights"). This is critical to prevent the image generator from using the wrong model year's front fascia!
                - Physically describe ONLY the visible decals, stripes, aero pieces (spoilers, splitters), and wheel types.
                
                CRITICAL: YOU MUST NOT hallucinate or guess any features that are not explicitly visible in this specific photo. Do not assume factory defaults. Return the result as a concise, descriptive paragraph.
                """

                analysis_res = client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=[analysis_prompt, source_img]
                )
                car_description = analysis_res.text

                # ── Step 2: Analyze the reference image's pose and lighting ──────────
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

                st.success("Analysis complete! Uploading source image for generation...")
                st.info(f"**Extracted DNA:** {car_description}\n\n**Target Pose & Lighting:** {pose_description}")

                # ── Step 3: Upload source image to fal.ai ────────────────────────────
                # FLUX Kontext requires a URL, so we upload the source image bytes first
                source_upload.seek(0)
                image_bytes = source_upload.read()
                mime_type = source_upload.type or "image/jpeg"

                with st.spinner("Uploading source image..."):
                    image_url = fal_client.upload(image_bytes, mime_type)

                # ── Step 4: Build the FLUX Kontext prompt ────────────────────────────
                # FLUX Kontext works best with direct editing instructions rather than
                # descriptive prompts — it preserves the source subject's pixel identity
                # while applying the requested environment/lighting changes.
                final_prompt = f"""
                Transform this car photograph into a professional automotive studio shot.
                
                PRESERVE EXACTLY from the source image:
                - Every detail of the car's body: paint color, finish, number of doors, hood shape, headlights, grille, wheels, and all visible trim. Do not add or remove any features.
                
                APPLY this exact camera angle and lighting:
                {pose_description}
                
                ADDITIONAL CONTEXT about the vehicle for accuracy:
                {car_description}
                
                The background must be a seamless studio cyclorama. Maintain photorealistic quality throughout.
                """.strip()

                # ── Step 5: Generate with FLUX Kontext ───────────────────────────────
                with st.spinner("Generating studio shot with FLUX Kontext... (this may take 30–60 seconds)"):
                    result = fal_client.run(
                        "fal-ai/flux-pro/v1/kontext",
                        arguments={
                            "prompt": final_prompt,
                            "image_url": image_url,
                            "guidance_scale": 3.5,       # Higher = more prompt adherence
                            "num_inference_steps": 28,
                            "num_images": 1,
                            "output_format": "jpeg",
                        }
                    )

                # ── Step 6: Display result ────────────────────────────────────────────
                images = result.get("images", [])
                if images:
                    import requests
                    output_url = images[0]["url"]
                    img_response = requests.get(output_url)
                    final_image = PIL.Image.open(io.BytesIO(img_response.content))
                    st.write("### Result")
                    st.image(final_image, caption="Final Studio Shot (FLUX Kontext)", use_container_width=True)
                    
                    # Download button
                    buf = io.BytesIO()
                    final_image.save(buf, format="JPEG", quality=95)
                    st.download_button(
                        label="⬇️ Download Image",
                        data=buf.getvalue(),
                        file_name="studio_shot.jpg",
                        mime="image/jpeg"
                    )
                else:
                    st.error("No image returned from fal.ai. Check your API key or try again.")

            except Exception as e:
                st.error(f"An error occurred: {e}")