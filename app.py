import streamlit as st
from google import genai
import fal_client
import io
import PIL.Image
import os
import requests

# Access API keys from Streamlit secrets
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
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

                # ── Step 1: Analyze the source car's visual DNA ───────────────────────
                analysis_prompt = """
                Analyze this source car image and extract its visual DNA.
                - A highly descriptive but generic vehicle baseline (CRITICAL: You MUST NOT name the actual vehicle brand, make, or model. Do not say "Pontiac", "Bonneville", or "Mustang". Naming the specific car triggers 2-door muscle car hallucinations in the final image generator. Instead, use a generic physical descriptor like "A large, luxurious 1960s classic American 4-door sedan").
                - The explicit body style (e.g., 2-door coupe vs 4-door sedan/hardtop) based on the visible door cutlines. If it has 4 doors, emphasize it heavily.
                - The hood geometry (e.g., completely flat hood with NO scoops). Explicitly state if it lacks hood scoops.
                - The primary paint color and finish.
                - The explicit headlight and grille configuration (e.g., "horizontally side-by-side dual headlights" vs "vertically stacked headlights"). This is critical to prevent the image generator from using the wrong model year's front fascia!
                - Physically describe ONLY the visible decals, stripes, aero pieces (spoilers, splitters), and wheel types.

                CRITICAL: YOU MUST NOT hallucinate or guess any features not explicitly visible. Return a concise, descriptive paragraph.
                """
                analysis_res = client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=[analysis_prompt, source_img]
                )
                car_description = analysis_res.text

                # ── Step 2: Analyze the reference image pose and lighting ───────────
                # The reference image is the source of truth for the final layout.
                pose_prompt = """
                Analyze this reference automotive photograph. Extract the camera positioning,
                vehicle orientation, and lighting so it can be replicated exactly.
                Describe:
                1. NOSE DIRECTION: State as an absolute screen direction only —
                   "the nose (front/headlights) points toward the LEFT edge of the frame"
                   OR "toward the RIGHT edge of the frame". Be unambiguous.
                2. CAMERA ANGLE: The horizontal angle around the car (e.g., front three-quarter, pure side profile).
                3. CAMERA ELEVATION: Are we looking slightly down at the hood/roof, level, or upward?
                4. LIGHTING & ENVIRONMENT: The studio lighting setup, light source types, shadow quality, background.
                Return a concise but precise paragraph.
                """
                pose_res = client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=[pose_prompt, reference_img]
                )
                pose_description = pose_res.text

                # ── Step 3: Extract nose direction as a hard one-sentence anchor ─────
                # Extracted separately so it can be injected as an unambiguous hard
                # constraint in the final prompt, preventing FLUX Kontext from flipping
                # the car horizontally.
                nose_prompt = """
                Look at this car photo. Reply with exactly one sentence in this exact format:
                "The car's nose points toward the [LEFT/RIGHT] side of the image frame."
                Replace [LEFT/RIGHT] with only the correct word. No other words or punctuation.
                """
                nose_res = client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=[nose_prompt, reference_img]
                )
                nose_direction = nose_res.text.strip()

                st.success("Analysis complete! Uploading source image for generation...")
                st.info(
                    f"**Extracted DNA:** {car_description}\n\n"
                    f"**Target Pose & Lighting:** {pose_description}\n\n"
                    f"**Nose direction lock:** {nose_direction}"
                )

                # ── Step 4: Upload source image to fal.ai ────────────────────────────
                source_upload.seek(0)
                image_bytes = source_upload.read()
                mime_type = source_upload.type or "image/jpeg"

                with st.spinner("Uploading source image..."):
                    image_url = fal_client.upload(image_bytes, mime_type)

                # ── Step 5: Build the FLUX Kontext prompt ────────────────────────────
                final_prompt = f"""
                Transform this car photograph into a professional automotive studio shot.

                PRESERVE EXACTLY from the source image:
                - Every physical detail of the car: paint color, finish, number of doors, hood shape,
                  headlights, grille, wheels, and all visible trim. Do not add or remove any features.

                APPLY this exact pose, camera angle, and lighting from the reference:
                {pose_description}

                CRITICAL — NOSE DIRECTION (HIGHEST PRIORITY — DO NOT VIOLATE):
                {nose_direction}
                The car's front end (headlights/grille) MUST face that exact screen direction.
                DO NOT mirror, flip, or reverse the car horizontally under any circumstances.
                If the nose points LEFT, the headlights must appear on the LEFT side of the final image.
                If the nose points RIGHT, the headlights must appear on the RIGHT side of the final image.

                CRITICAL FEATURE RULES:
                1. If the description says "4-door", physically draw four doors.
                2. If the description says the hood is flat, draw a flat hood with NO HOOD SCOOPS.
                3. Do NOT add spoilers, decals, stripes, or modifications not listed in the vehicle description.

                ADDITIONAL VEHICLE CONTEXT:
                {car_description}

                The background must be a seamless studio cyclorama. Photorealistic quality throughout.
                """.strip()

                # ── Step 6: Generate with FLUX Kontext ───────────────────────────────
                with st.spinner("Generating studio shot with FLUX Kontext... (this may take 30–60 seconds)"):
                    result = fal_client.run(
                        "fal-ai/flux-pro/kontext/max",
                        arguments={
                            "prompt": final_prompt,
                            "image_url": image_url,
                            "guidance_scale": 3.5,
                            "num_inference_steps": 28,
                            "num_images": 1,
                            "output_format": "jpeg",
                        }
                    )

                # ── Step 7: Display result ────────────────────────────────────────────
                images = result.get("images", [])
                if images:
                    output_url = images[0]["url"]
                    img_response = requests.get(output_url)
                    final_image = PIL.Image.open(io.BytesIO(img_response.content))
                    st.write("### Result")
                    st.image(final_image, caption="Final Studio Shot (FLUX Kontext)", use_container_width=True)

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
