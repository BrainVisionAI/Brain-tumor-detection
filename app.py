from PIL import Image
import streamlit as st
from ultralytics import YOLO
import io
import tempfile
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as PDFImage
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime


st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)
#backgroung color
st.markdown(
    """
    <style>
        .stApp {
            background-color: #f4f7fb;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# loading model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

#creating sidebar
st.sidebar.title("Menu")
page = st.sidebar.radio(
    "Choose a page:",
    [
        "Quick Project Summary",
        "🔍 Brain Tumor Detector"
    ]
    
)
# this part is related to "Quick Project Summary"
if page == "Quick Project Summary":

    st.write('# Brain Tumor Detection')  
    st.markdown(
        '<div class="section-title"> AI-powered MRI analysis using YOLO11s</div>',
        unsafe_allow_html=True
    )
    st.subheader("Quick Project Summary")

    st.write(
        'This project is an AI-based brain tumor detection system designed to analyze MRI images using a trained YOLO11s model.'
        'The system detects and classifies four categories:'
 
        '\n1. Glioma'
        '\n2. Meningioma'
        '\n3. No Tumor'
        '\n4. Pituitary'

     '\n\nThe trained model performs object detection by identifying'
     ' the detected region in the MRI image and providing a confidence score for the prediction.'
        )
    
    st.info(
        '* For additional information, please visit and read the '

    )
    st.write('---')
    st.subheader(" How the System Works")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.success("**1. Upload MRI**\n\n Upload an MRI image.")

    with col2:
        st.success("**2. YOLO11s**\n\n The trained model analyzes the image.")

    with col3:
        st.success("**3. Detection**\n\n The tumor region is detected.")

    with col4:
        st.success("**4. Result**\n\n Diagnosis and confidence are displayed.")


    st.warning(
        "⚠️ This application is for educational and research "
        "purposes only and is not a medical diagnosis."
    )





# this part is related to "🔍 Brain Tumor Detector"
if page == "🔍 Brain Tumor Detector":
    st.write('# Brain Tumor Detection') 
    st.markdown(
        '<div class="section-title"> AI-powered MRI analysis using YOLO11s</div>',
        unsafe_allow_html=True
    )
    st.subheader("Brain Tumor Detector")
    st.info(
        '* The dataset used was a brain MRI dataset with bounding box'
        ' annotations in YOLO format, obtained and prepared using Roboflow:\n'
        '[Roboflow]'
        '(https://universe.roboflow.com/mango-qoesz/labeled-mri-brain-tumor-dataset-l8ayj).'
        )
    # creating uploader
    uploaded_file = st.file_uploader(
        " Upload MRI Image",
        type=["jpg", "jpeg", "png"]
    )
    # prettier Analyze button
    st.markdown("""
      <style>
       div.stButton > button {
       background-color: #6c63ff;
       color: white;
       height: 2em;
       width: 100%;
       border-radius: 10px;
       font-size: 16px;
     }
        div[data-testid="stAppViewContainer"] {
        background-color: #fafafa;
    }
     </style>
      """, unsafe_allow_html=True)

    if st.button("analyze"):
        if uploaded_file is not None:
            
            image = Image.open(uploaded_file).convert("RGB") #opening & converting the image
                                                            # to the right format

            with st.spinner("Analyzing MRI..."):

                results = model(image)

                result = results[0] # taking first result

                annotated = result.plot() # adding Bounding Box

                if len(result.boxes) > 0: # if there is any Bounding Box:

                        best_box = result.boxes[0] # choose the first one Temporarily
                        for box in result.boxes:
                            if box.conf > best_box.conf: # choosing the best confidence among all boxed
                                best_box = box

                        class_id = int(best_box.cls.item()) # taking YOLO id
                        confidence = float(best_box.conf.item()) # taking YOlO confidence
                        diagnosis = model.names[class_id]  # taking the name of the id,
                                                           # which is the name of the tumor

                # else:

                #     diagnosis = "No Detection"
                #     confidence = 0

            st.subheader("MRI Result")
            # columns for images
            col1, col2 = st.columns(2)

            with col1:

                st.write("### Original MRI")

                st.image(
                    image,
                    use_container_width=True
                )
            with col2:
                st.write("### Detection Result")

                st.image(
                    annotated,
                    use_container_width=True
                )

            st.divider()
            # columns for diagnosis
            col1, col2 = st.columns(2)
            with col1:

                st.metric(
                    "Diagnosis",
                    diagnosis
                )

            with col2:

                number= f"{confidence * 100:.2f}%"
                st.metric(
                    "Confidence",
                    number
                )


            # PDF
            st.markdown("---")
            st.subheader(" PDF Medical Report")
        
            pdf = io.BytesIO() # creating an empty space
            document = SimpleDocTemplate(  # creating an pdf file called document
              pdf,
              pagesize=A4 
              )
            styles = getSampleStyleSheet()  # pdf style
            content = []     # creating a list for adding the information we want

            content.append(
                 Paragraph(
                    "Brain MRI Analysis Report", # title
                    styles["Title"]
                    )
            )
            
            content.append(
              Spacer(1, 15)    # space
            )

            date = datetime.now().strftime(   # date
               "%Y-%m-%d %H:%M"
            )

            content.append(
              Paragraph(
              f"Analysis Date: {date}",  #adding date
              styles["Normal"]
               )
            )

            content.append(
            Spacer(1, 10)
            )

            content.append(
               Paragraph(
               "Model: YOLO11s",  #adding the name of the model
                styles["Normal"]
                )
            )

            content.append(
                Spacer(1, 10)
            )

            content.append(
                 Paragraph(
                 f"Diagnosis: {diagnosis}", #adding the name of tumor
                  styles["Normal"]
                  )
            )

            content.append(
                Paragraph(
                f"Confidence: {number}", # adding the number of confidence
                styles["Normal"]
                   )
            )
            # Temporary file for image
            original_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".png"
            )

            image.save(
                original_file.name,
                format="PNG"
            )
            # Temporary file for annotated 
            detection_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".png"
            )
            # converting annotated to an image just like above
            detection_image = Image.fromarray(
               annotated
            )

            detection_image.save(
                detection_file.name,
                format="PNG"
            )

            content.append(
               Paragraph(
               "Original MRI",
               styles["Heading2"]
               )
            )

            content.append(
                Spacer(1, 10)
            )

            content.append(
                PDFImage(
                original_file.name,
                width=210,
                height=210
                )
            )

            content.append(
                Spacer(1, 10)
            )

            content.append(
                Paragraph(
                "Detection Result",
                styles["Heading2"]
               )
            )

            content.append(
                Spacer(1, 10)
            )

            content.append(
               PDFImage(
               detection_file.name,
               width=210,
               height=210
               )
            )

            content.append(
            Spacer(1, 20)
            )           

            content.append(
               Paragraph(
               "Disclaimer: This report is intended for "
               "educational and research purposes only and "
               "is not a medical diagnosis.",
               styles["Normal"]
                )

            )

            document.build(content)
            pdf.seek(0)

            #prettier button
            st.markdown("""
              <style>
              .stDownloadButton button {
              background-color: #4CAF50;
              color: white;
                }
               </style>
                """, unsafe_allow_html=True)

            
            st.download_button(
               "Download Report",
                data=pdf,
                file_name="brain_tumor_report.pdf",
                mime="application/pdf"
            )
        

                

                