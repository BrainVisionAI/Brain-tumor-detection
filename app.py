from PIL import Image

import streamlit as st

from ultralytics import YOLO

import io

import tempfile

from textwrap import dedent

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as PDFImage


from reportlab.lib.pagesizes import A4

from reportlab.lib.styles import getSampleStyleSheet

from datetime import datetime


import base64

from pathlib import Path


def get_image_base64(image_path):
    image_path = Path(image_path)
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


st.set_page_config(
    page_title="Brainvision AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown("""


<style>

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    /* ---------- MAIN BACKGROUND ---------- */

    .stApp {
        background: linear-gradient(180deg, #eef2fa 0%, #f6f8fc 100%);
    }

    .main .block-container {
        max-width: 1250px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }


    /* ---------- HEADER ---------- */

    .top-header {
        background: linear-gradient(
            135deg,
            #081c44,
            #132f6e 55%,
            #2a3f8f
        );

        padding: 26px 34px;
        border-radius: 20px;
        margin-bottom: 30px;

        box-shadow:
            0 12px 32px rgba(10, 35, 90, 0.22);
    }

    .brand-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .brand-left {
        display: flex;
        align-items: center;
        gap: 18px;
    }

    .brain-logo {
        width: 56px;
        height: 56px;

        display: flex;
        align-items: center;
        justify-content: center;

        font-size: 32px;

        border-radius: 16px;

        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.18);
        box-shadow: 0 4px 14px rgba(0,0,0,0.15);
    }

    .brain-logo img {
        width: 32px;
        height: 32px;
        object-fit: contain;
    }

    .brand-name {
        color: white;
        font-size: 28px;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.3px;
    }

    .brand-subtitle {
        color: #c3d2ec;
        font-size: 13px;
        margin-top: 3px;
        letter-spacing: 0.2px;
    }

    .header-badges {
        display: flex;
        gap: 10px;
        align-items: center;
    }

    .model-badge {
        background: #3a63d1;
        color: white;
        padding: 8px 17px;
        border-radius: 22px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.3px;
        box-shadow: 0 3px 10px rgba(58, 99, 209, 0.35);
    }

    .ready-badge {
        background: rgba(50, 200, 130, 0.16);
        color: #7cebb2;
        border: 1px solid rgba(50, 200, 130, 0.3);
        padding: 8px 17px;
        border-radius: 22px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.3px;
    }


    /* ---------- MAIN TITLE ---------- */

    .main-title {
        color: #101f3c;
        font-size: 40px;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 6px;
    }

    .main-description {
        color: #6b7794;
        font-size: 15.5px;
        margin-bottom: 26px;
    }


    /* ---------- CARDS ---------- */

.card {
    background: white;
    border: 1px solid #e9edf5;
    border-radius: 16px;
    padding: 24px;
    min-height: 0;

    box-shadow:
        0 4px 14px rgba(20, 50, 100, 0.05);

    transition: box-shadow 0.2s ease, transform 0.2s ease;
}

.card:hover {
    box-shadow:
        0 8px 20px rgba(20, 50, 100, 0.08);
}

.upload-card {
    margin-bottom: 0;
}

    .card-title {
        color: #16264a;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 6px;
        letter-spacing: -0.1px;
    }

    .step-number {
        display: inline-flex;
        width: 28px;
        height: 28px;

        align-items: center;
        justify-content: center;

        background: linear-gradient(
            135deg,
            #2f6bf0,
            #7c4bef
        );

        color: white;
        border-radius: 50%;
        font-size: 13px;
        font-weight: 700;
        margin-right: 8px;
        box-shadow: 0 3px 8px rgba(90, 80, 230, 0.3);
    }

    .card-description {
        color: #808da3;
        font-size: 13px;
        margin-bottom: 18px;
    }


   /* ---------- CARDS WRAPPER ----------
      Wraps each column (Upload MRI Image / MRI Preview) in an
      identical card so the pair shares the same height, border,
      radius, padding and shadow, side by side.                 */

@supports selector(:has(*)) {

    [data-testid="stColumn"]:has(.upload-card),
    [data-testid="stColumn"]:has(.preview-title-block) {
        background: white;
        border: 1px solid #e6ecf5;
        border-radius: 16px;
        padding: 26px;

        box-shadow:
            0 4px 14px rgba(20, 50, 100, 0.05);

        transition: box-shadow 0.2s ease, transform 0.2s ease;
    }

    [data-testid="stColumn"]:has(.upload-card):hover {
        box-shadow:
            0 8px 20px rgba(20, 50, 100, 0.08);
    }

    /* the inner card header no longer needs its own box */
    [data-testid="stColumn"]:has(.upload-card) .card.upload-card {
        background: transparent;
        border: none;
        border-radius: 0;
        box-shadow: none;
        padding: 0;
        margin-bottom: 0;
    }

    [data-testid="stColumn"]:has(.upload-card) .card-description,
    [data-testid="stColumn"]:has(.preview-title-block) .card-description {
        margin-bottom: 20px;
    }

}


   /* ---------- UPLOAD AREA ---------- */

[data-testid="stFileUploader"] {
    width: 100% !important;
    height: 220px !important;

    margin: 0 auto !important;
    padding: 0 !important;

    background: transparent !important;
    border: none !important;

    position: relative;
    z-index: 2;
    overflow: hidden;
}


/* DROPZONE (fixed height so the box never changes size) */
[data-testid="stFileUploaderDropzone"] {
    width: 100% !important;
    height: 220px !important;
    min-height: 0 !important;
    max-height: 220px !important;

    padding: 26px 18px !important;

    background: #f7f9fc !important;

    border: 1.5px dashed #cdd9ec !important;
    border-radius: 12px !important;

    box-sizing: border-box !important;

    display: flex !important;
    flex-direction: column !important;

    align-items: center !important;
    justify-content: center !important;

    gap: 6px !important;

    transition: border-color 0.2s ease, background 0.2s ease;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #8fabe8 !important;
    background: #f2f5fb !important;
}


/* همه محتوای داخل Dropzone */
[data-testid="stFileUploaderDropzone"] > div {
    width: 100% !important;

    display: flex !important;
    flex-direction: column !important;

    align-items: center !important;
    justify-content: center !important;

    text-align: center !important;
    gap: 4px !important;
}


/* متن Drag & Drop */
[data-testid="stFileUploaderDropzone"] label {
    width: 100% !important;

    display: block !important;

    text-align: center !important;

    color: #33455f !important;

    font-size: 15px !important;
    font-weight: 600 !important;
}


/* آیکون */
[data-testid="stFileUploaderDropzone"] svg {
    width: 42px !important;
    height: 42px !important;
    margin-bottom: 6px !important;

    color: #5b8def !important;
    fill: #5b8def !important;
}


/* اطلاعات فایل (داخل Dropzone) — hide Streamlit's own hint;
   no file-type caption is shown in the upload area anymore  */
[data-testid="stFileUploaderDropzone"] small {
    display: none !important;
}


/* دکمه Upload (نمایش فقط "Upload") */
[data-testid="stFileUploaderDropzone"] button {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;

    margin:10px auto 0 auto !important;

    padding: 8px 28px !important;

    border: 1px solid #c8d6ec !important;

    border-radius: 8px !important;

    background: white !important;

    color: transparent !important;

    font-size: 0 !important;

    font-weight: 600 !important;

    cursor: pointer !important;

    transition: box-shadow 0.2s ease, transform 0.2s ease;
}

/* hide any native text Streamlit places inside the button (and any
   child element of the button), so the word appears only once and
   the "upload" text sits exactly centered in the button           */
[data-testid="stFileUploaderDropzone"] button {
    font-size: 0 !important;
    color: transparent !important;
    text-shadow: none !important;
}

[data-testid="stFileUploaderDropzone"] button * {
    display: none !important;
    font-size: 0 !important;
    color: transparent !important;
    text-shadow: none !important;
}

/* replace Streamlit's default button text with "upload" */
[data-testid="stFileUploaderDropzone"] button::after {
    content: "upload";
    font-size: 13.5px;
    font-weight: 600;
    color: #3159bd;
    line-height: 1;
}

/* hide any folder icon Streamlit may place inside the button */
[data-testid="stFileUploaderDropzone"] button svg {
    display: none !important;
}

[data-testid="stFileUploaderDropzone"] button:hover {
    box-shadow: 0 3px 10px rgba(49, 89, 189, 0.18) !important;
    transform: translateY(-1px);
}


/* ---------- SELECTED FILE ----------
   The dropzone must stay visually identical after a file is
   selected, so Streamlit's file-info row is hidden entirely.
   The dropzone itself is forced visible with a fixed size, so
   the box never grows, shrinks, or moves.                     */

[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
    display: none !important;
}

/* hide any widget label/caption Streamlit may render outside
   the dropzone, so only the styled inner drag text shows       */
[data-testid="stFileUploader"] [data-testid="stWidgetLabel"],
[data-testid="stFileUploader"] > [data-testid="stCaptionContainer"] {
    display: none !important;
}


/* ---------- CLEAR (X) BUTTON ----------
   Shown in the upload box's top-right corner only while a file
   is selected. Absolutely positioned so it never changes the
   size or position of the fixed upload box.                    */

@supports selector(:has(*)) {

    /* the container that holds the uploader + X button */
    [data-testid="stVerticalBlock"]:has([data-testid="stFileUploader"]):not(:has(.upload-card)) {
        position: relative;
        margin: 0 !important;
        padding: 0 !important;
    }

    [data-testid="stVerticalBlock"]:has([data-testid="stFileUploader"]):not(:has(.upload-card)) [data-testid="stButton"] {
    position: absolute !important;
    top: -143px !important;
    left: 130px !important;
    z-index: 100 !important;
}


    [data-testid="stVerticalBlock"]:has([data-testid="stFileUploader"]):not(:has(.upload-card)) [data-testid="stButton"] button {
        width: 30px !important;
        height: 30px !important;
        min-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        border: 1px solid #e0e7f2 !important;
        border-radius: 8px !important;
        background: rgba(255, 255, 255, 0.92) !important;
        box-shadow: 0 2px 8px rgba(20, 50, 100, 0.10) !important;
        color: #8b97aa !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        line-height: 1 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        transition: color 0.2s ease, background 0.2s ease, box-shadow 0.2s ease;
    }

    [data-testid="stVerticalBlock"]:has([data-testid="stFileUploader"]):not(:has(.upload-card)) [data-testid="stButton"] button:hover {
        background: #fdecec !important;
        color: #d64545 !important;
        transform: none !important;
        box-shadow: 0 3px 10px rgba(214, 69, 69, 0.20) !important;
    }

}

    /* ---------- BUTTON ---------- */

div.stButton > button {
    width: 100%;
    height: 44px;
    margin-top: 14px;

    border: none;
    border-radius: 9px;

    background: linear-gradient(
        90deg,
        #3462f0,
        #7c4bef
    );

    color: white;
    font-size: 14.5px;
    font-weight: 600;
    letter-spacing: 0.2px;

    padding: 9px 18px;

    box-shadow: 0 4px 12px rgba(90, 70, 220, 0.18);

    transition: all 0.2s ease;
}

div.stButton > button:hover {
    transform: translateY(-1px);

    box-shadow:
        0 6px 16px rgba(75, 70, 220, 0.28);
}


    /* ---------- MRI PREVIEW ----------
       Fixed-size preview box with a thin solid light-gray border.
       It keeps its exact dimensions whether empty or showing the
       uploaded MRI, so the layout never shifts.                */

@supports selector(:has(*)) {

    [data-testid="stColumn"]:has(.preview-title-block) .preview-title-block {
        margin-bottom: 6px;
    }

    /* uploaded image is contained inside the fixed preview box */
    [data-testid="stColumn"]:has(.preview-title-block) [data-testid="stImage"] {
        width: 100% !important;
        height: 220px !important;
        box-sizing: border-box !important;

         border: none !important;
    background: transparent !important;
    border-radius: 0 !important;

        overflow: hidden !important;

        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    [data-testid="stColumn"]:has(.preview-title-block) [data-testid="stImageContainer"] {
        width: 100% !important;
        height: 100% !important;
        padding: 0 !important;
        margin: 0 !important;
        overflow: hidden !important;

        display: flex !important;
        align-items: center !important;
        justify-content: flex-end !important;
        
    }

    /* image keeps its aspect ratio, never stretches or crops, and is
       centered both horizontally and vertically inside the box.
       Flex centering (not absolute positioning) keeps it anchored to
       the box itself, so it can never be displaced or hidden.
       min-width/min-height: 0 lets the flex item shrink below its
       intrinsic size so it fits and centers instead of overflowing. */
    [data-testid="stColumn"]:has(.preview-title-block) [data-testid="stImage"] img,
    [data-testid="stColumn"]:has(.preview-title-block) [data-testid="stImageContainer"] img {
        max-width: 100% !important;
        max-height: 100% !important;
        width: auto !important;
        height: auto !important;
        object-fit: contain !important;
        display: block !important;
        margin-left: 170px !important;
margin-right: 0 !important;
        min-width: 0 !important;
        min-height: 0 !important;
    }

}

    /* placeholder box matches the preview container size */
    .preview-empty {
        height: 220px;
        box-sizing: border-box;

        border: 1px solid #e3e9f3;
        background: #fafbfd;
        border-radius: 12px;

        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;

        text-align: center;
        color: #75839a;
    }

    .preview-icon {
        font-size: 58px;
        margin-bottom: 14px;
        opacity: 0.85;
    }

    .preview-icon img {
        width: 58px;
        height: 58px;
        object-fit: contain;
        display: block;
        margin: 0 auto;
    }

    .preview-title {
        color: #35445d;
        font-size: 15.5px;
        font-weight: 700;
    }

    .preview-text {
        color: #8b97aa;
        font-size: 12.5px;
        margin-top: 6px;
    }


    /* ---------- PIPELINE ---------- */

.pipeline-card {
    background: white;
    border: 1px solid #e6ebf4;
    border-radius: 16px;
    padding: 18px 22px;
    margin-top: 18px;

    box-shadow:
        0 5px 16px rgba(20, 50, 100, 0.05);
}

.pipeline-title {
    color: #16264a;
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 18px;

    display: flex;
    align-items: center;
    gap: 8px;
}

.pipeline-title img {
    width: 24px;
    height: 24px;
    object-fit: contain;
}

.pipeline {
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.pipeline-item {
    text-align: center;
    flex: 1;
    position: relative;
}

.pipeline-icon {
    width: 38px;
    height: 38px;

    margin: auto;

    display: flex;
    align-items: center;
    justify-content: center;

    border-radius: 50%;

    background: #f1f5ff;
    border: 1px solid #d9e4f9;

    overflow: hidden;

    transition: box-shadow 0.2s ease;
}

.pipeline-item:hover .pipeline-icon {
    box-shadow: 0 4px 12px rgba(60, 90, 200, 0.18);
}

.pipeline-icon img {
    width: 22px;
    height: 22px;

    object-fit: contain;
    display: block;
}

.pipeline-label {
    color: #6b7998;
    font-size: 11px;
    font-weight: 500;
    margin-top: 7px;
}

.pipeline-line {
    height: 1px;
    background: linear-gradient(90deg, #dbe3ef, #c7d3ea, #dbe3ef);
    flex: 1;
}

    /* ---------- RESULTS ---------- */

    .result-card {
        background: white;
        border: 1px solid #e3e9f3;
        border-radius: 16px;
        padding: 22px;

        box-shadow:
            0 5px 18px rgba(20, 50, 100, 0.05);
    }

    .result-label {
        color: #748198;
        font-size: 12px;
        margin-bottom: 5px;
    }

    .result-value {
        color: #15294d;
        font-size: 24px;
        font-weight: 700;
    }


    /* ---------- SECTION ---------- */

    .section-title {
        color: #16264a;
        font-size: 23px;
        font-weight: 800;
        letter-spacing: -0.3px;
        margin-top: 30px;
        margin-bottom: 16px;
    }


    /* ---------- DOWNLOAD BUTTON ---------- */

    .stDownloadButton button {
        width: 100%;
        border-radius: 11px;
        background: #16264a;
        color: white;
        font-weight: 700;
        border: none;
        padding: 12px;
        letter-spacing: 0.2px;
        transition: all 0.2s ease;
    }

    .stDownloadButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 18px rgba(20, 40, 90, 0.28);
    }
   
    
/* =========================================================
    RESPONSIVE DESIGN
   ========================================================= */

/* ---------- TABLET ---------- */
@media (max-width: 900px) {

    .main .block-container {
        padding-left: 20px !important;
        padding-right: 20px !important;
    }

    .top-header {
        padding: 22px 24px;
    }

    .brand-name {
        font-size: 24px;
    }

    .brand-subtitle {
        font-size: 12px;
    }

    .brain-logo {
        width: 50px;
        height: 50px;
    }

    .main-title {
        font-size: 34px;
    }

    .pipeline {
        gap: 8px;
    }

    .pipeline-label {
        font-size: 10px;
    }
}


/* ---------- MOBILE ---------- */
@media (max-width: 768px) {

    .main .block-container {
        width: 100% !important;
        max-width: 100% !important;
        padding-left: 12px !important;
        padding-right: 12px !important;
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
    }

    /* HEADER */
    .top-header {
        padding: 18px !important;
        border-radius: 16px !important;
        margin-bottom: 20px !important;
    }

    .brand-container {
        flex-direction: column;
        align-items: flex-start;
        gap: 16px;
    }

    .brand-left {
        gap: 12px;
    }

    .brain-logo {
        width: 46px;
        height: 46px;
        border-radius: 13px;
    }

    .brain-logo img {
        width: 27px;
        height: 27px;
    }

    .brand-name {
        font-size: 22px;
        line-height: 1.2;
    }

    .brand-subtitle {
        font-size: 11px;
        line-height: 1.4;
    }

    .header-badges {
        width: 100%;
        justify-content: flex-start;
        flex-wrap: wrap;
        gap: 8px;
    }

    .model-badge,
    .ready-badge {
        padding: 7px 12px;
        font-size: 11px;
    }


    /* MAIN TITLE */
    .main-title {
        font-size: 28px !important;
        line-height: 1.2 !important;
    }

    .main-description {
        font-size: 13px !important;
        line-height: 1.5 !important;
        margin-bottom: 20px !important;
    }


    /* CARDS */
    .card,
    [data-testid="stColumn"]:has(.upload-card),
    [data-testid="stColumn"]:has(.preview-title-block) {
        padding: 16px !important;
        border-radius: 14px !important;
    }

    .card-title {
        font-size: 16px;
    }

    .card-description {
        font-size: 12px;
    }


    /* UPLOAD BOX */
    [data-testid="stFileUploader"] {
        height: 200px !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        height: 200px !important;
        max-height: 200px !important;
        padding: 20px 12px !important;
    }

    [data-testid="stFileUploaderDropzone"] label {
        font-size: 13px !important;
    }

    [data-testid="stFileUploaderDropzone"] svg {
        width: 36px !important;
        height: 36px !important;
    }


    /* CLEAR X BUTTON
       Remove the fixed desktop positioning */
    [data-testid="stVerticalBlock"]:has([data-testid="stFileUploader"]):not(:has(.upload-card))
    [data-testid="stButton"] {
        top: 8px !important;
        left: auto !important;
        right: 8px !important;
    }


    /* MRI PREVIEW */
[data-testid="stColumn"]:has(.preview-title-block)
[data-testid="stImage"],
[data-testid="stColumn"]:has(.preview-title-block)
[data-testid="stImageContainer"] {
    width: 100% !important;
    height: 200px !important;

    display: flex !important;
    align-items: center !important;
    justify-content: center !important;

    overflow: hidden !important;
}

[data-testid="stColumn"]:has(.preview-title-block)
[data-testid="stImage"] img,
[data-testid="stColumn"]:has(.preview-title-block)
[data-testid="stImageContainer"] img {
    display: block !important;

    width: auto !important;
    height: auto !important;

    max-width: 100% !important;
    max-height: 100% !important;

    margin: 0 !important;

    object-fit: contain !important;
}


    /* EMPTY PREVIEW */
    .preview-empty {
        height: 200px !important;
        padding: 15px !important;
    }

    .preview-icon {
        font-size: 45px;
        margin-bottom: 10px;
    }

    .preview-icon img {
        width: 48px;
        height: 48px;
    }

    .preview-title {
        font-size: 14px;
    }

    .preview-text {
        font-size: 11px;
    }


    /* PIPELINE */
    .pipeline-card {
        padding: 16px !important;
        overflow-x: auto !important;
    }

    .pipeline {
        min-width: 620px;
        justify-content: flex-start;
        gap: 4px;
    }

    .pipeline-item {
        min-width: 95px;
    }

    .pipeline-line {
        min-width: 22px;
    }


    /* RESULTS */
    .result-card {
        padding: 16px;
    }

    .section-title {
        font-size: 20px;
        margin-top: 24px;
    }


    /* DOWNLOAD */
    .stDownloadButton button {
        font-size: 13px !important;
        padding: 11px !important;
    }


    /* REPOSITORY BUTTON */
    .project-resources-title,
    .developed-title {
        font-size: 18px !important;
    }
}


/* ---------- SMALL MOBILE ---------- */
@media (max-width: 480px) {

    .main .block-container {
        padding-left: 8px !important;
        padding-right: 8px !important;
    }

    .top-header {
        padding: 15px !important;
    }

    .brand-name {
        font-size: 20px;
    }

    .brand-subtitle {
        font-size: 10px;
    }

    .main-title {
        font-size: 24px !important;
    }

    .main-description {
        font-size: 12px !important;
    }

    .step-number {
        width: 25px;
        height: 25px;
        font-size: 11px;
    }

    [data-testid="stFileUploaderDropzone"] {
        padding: 16px 8px !important;
    }

    .pipeline-item {
        min-width: 85px;
    }

    .pipeline-label {
        font-size: 9px;
    }
}






</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return YOLO("best.pt")


model = load_model()


header_icon = get_image_base64("icons/header-icon.png")


st.html(dedent(f"""
<div class="top-header">

    <div class="brand-container">

        <div class="brand-left">

            <div class="brain-logo">
                <img src="data:image/png;base64,{header_icon}" alt="Brainvision AI">
            </div>

            <div>

                <div class="brand-name">
                    Brainvision AI
                </div>

                <div class="brand-subtitle">
                    AI-Powered Brain MRI Analysis
                </div>

            </div>

        </div>

        <div class="header-badges">

            <div class="model-badge">
                YOLO11s
            </div>

            <div class="ready-badge">
                ● Ready
            </div>

        </div>

    </div>

</div>
"""))


st.html('<div class="main-title">Brain Tumor Detection</div>')

st.html('<div class="main-description">Detect and localize brain tumors in MRI scans using YOLO11s</div>')


def clear_upload():
    """Rotate the uploader key so Streamlit discards the
    currently selected file and resets the upload box."""
    st.session_state["upload_cycle"] = (
        st.session_state.get("upload_cycle", 0) + 1
    )


col_upload, col_preview = st.columns(
    [1, 1],
    gap="large",
)

with col_upload:

    st.html(dedent("""
    <div class="card upload-card">

        <div class="card-title">
            <span class="step-number">1</span>
            Upload MRI Image
        </div>

        <div class="card-description">
            Upload a brain MRI image for
            AI-powered tumor detection.
        </div>

    </div>
    """))

    # A dedicated container holds the fixed-size upload box and the
    # small X overlay, so the X can be anchored to the box's own
    # top-right corner without affecting the surrounding layout.
    uploader_box = st.container()

    with uploader_box:

        uploaded_file = st.file_uploader(
            "Drag & Drop your MRI here",
            ["jpg", "jpeg", "png"],
            label_visibility="visible",
            key=f"mri_uploader_{st.session_state.get('upload_cycle', 0)}"
        )

        if uploaded_file is not None:

            st.button(
                "✕",
                key="clear_mri",
                help="Remove selected image",
                on_click=clear_upload
            )

    analyze = st.button(
        "⌕  Analyze MRI"
    )

with col_preview:

    st.html(dedent("""
    <div class="card-title preview-title-block">
        <span class="step-number">2</span>
        MRI Preview
    </div>

    <div class="card-description">
        Your uploaded MRI will appear here.
    </div>
    """))

    if uploaded_file is None:

        mri_preview_icon = get_image_base64("icons/mri-preview-icon.png")

        st.html(dedent(f"""
        <div class="preview-empty">

            <div class="preview-icon">
                <img src="data:image/png;base64,{mri_preview_icon}" alt="MRI Preview">
            </div>

            <div class="preview-title">
                No image uploaded yet
            </div>

            <div class="preview-text">
                Your MRI scan will be shown here
                before and after analysis.
            </div>

        </div>
        """))

    else:

        preview_image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            preview_image,
            use_container_width=True
        )


brain_icon = get_image_base64("icons/brain.png")
upload_icon = get_image_base64("icons/upload.png")

preprocessing_icon = get_image_base64("icons/preprocessing.png")

yolo_icon = get_image_base64("icons/yolo.png")

detection_icon = get_image_base64("icons/detection.png")

results_icon = get_image_base64("icons/results.png")


st.html(f"""
<div class="pipeline-card">

    <div class="pipeline-title">
    <img
        src="data:image/png;base64,{brain_icon}"
        alt="AI"
    >
    <span>AI Analysis Pipeline</span>
</div>

    <div class="pipeline">

        <!-- UPLOAD -->

        <div class="pipeline-item">

            <div class="pipeline-icon">
                <img
                    src="data:image/png;base64,{upload_icon}"
                    alt="Upload"
                >
            </div>

            <div class="pipeline-label">
                Upload
            </div>

        </div>


        <div class="pipeline-line"></div>


        <!-- PREPROCESSING -->

        <div class="pipeline-item">

            <div class="pipeline-icon">
                <img
                    src="data:image/png;base64,{preprocessing_icon}"
                    alt="Preprocessing"
                >
            </div>

            <div class="pipeline-label">
                Preprocessing
            </div>

        </div>


        <div class="pipeline-line"></div>


        <!-- YOLO -->

        <div class="pipeline-item">

            <div class="pipeline-icon">
                <img
                    src="data:image/png;base64,{yolo_icon}"
                    alt="YOLO11s"
                >
            </div>

            <div class="pipeline-label">
                YOLO11s<br>
                Inference
            </div>

        </div>


        <div class="pipeline-line"></div>


        <!-- DETECTION -->

        <div class="pipeline-item">

            <div class="pipeline-icon">
                <img
                    src="data:image/png;base64,{detection_icon}"
                    alt="Detection"
                >
            </div>

            <div class="pipeline-label">
                Detection
            </div>

        </div>


        <div class="pipeline-line"></div>


        <!-- RESULTS -->

        <div class="pipeline-item">

            <div class="pipeline-icon">
                <img
                    src="data:image/png;base64,{results_icon}"
                    alt="Results"
                >
            </div>

            <div class="pipeline-label">
                Results
            </div>

        </div>

    </div>

</div>
""")


if analyze:

    if uploaded_file is None:

        st.warning(
            "Please upload an MRI image first."
        )

    else:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        with st.spinner("Analyzing MRI with YOLO11s..."):

            results = model(image)

            result = results[0]

            annotated = result.plot()

            if len(result.boxes) > 0:

                best_box = result.boxes[0]

                for box in result.boxes:

                    if box.conf.item() > best_box.conf.item():

                        best_box = box

                class_id = int(
                    best_box.cls.item()
                )

                confidence = float(
                    best_box.conf.item()
                )

                diagnosis = model.names[
                    class_id
                ]

                detection_count = len(
                    result.boxes
                )

            else:

                diagnosis = "No Detection"

                confidence = 0

                detection_count = 0


        st.html('<div class="section-title">MRI Analysis Results</div>')

        result_col1, result_col2 = st.columns(
            2,
            gap="large",
        )

        with result_col1:

            st.html('<div class="result-card"><b>Original MRI</b></div>')

            st.image(
                image,
                use_container_width=True
            )

        with result_col2:

            st.html('<div class="result-card"><b>Detection Result</b></div>')

            st.image(
                annotated,
                use_container_width=True
            )

        st.html('<div class="section-title">Detection Summary</div>')

        metric1, metric2, metric3 = st.columns(
            3
        )

        with metric1:

            st.metric(
                "Diagnosis",
                diagnosis
            )

        with metric2:

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )

        with metric3:

            st.metric(
                "Detections",
                detection_count
            )

        st.html('<div class="section-title">📄 Analysis Report</div>')

        pdf = io.BytesIO()

        document = SimpleDocTemplate(
            pdf,
            pagesize=A4,
        )

        styles = getSampleStyleSheet()

        content = []

        content.append(
            Paragraph(
                "Brain MRI Analysis Report",
                styles["Title"]
            )
        )

        content.append(
            Spacer(1, 15)
        )

        date = datetime.now().strftime(
            "%Y-%m-%d %H:%M"
        )

        content.append(
            Paragraph(
                f"Analysis Date: {date}",
                styles["Normal"]
            )
        )

        content.append(
            Spacer(1, 10)
        )

        content.append(
            Paragraph(
                "Model: YOLO11s",
                styles["Normal"]
            )
        )

        content.append(
            Spacer(1, 10)
        )

        content.append(
            Paragraph(
                f"Diagnosis: {diagnosis}",
                styles["Normal"]
            )
        )

        content.append(
            Paragraph(
                f"Confidence: {confidence * 100:.2f}%",
                styles["Normal"]
            )
        )

        content.append(
            Spacer(1, 15)
        )

        original_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".png",
        )

        image.save(
            original_file.name,
            format="PNG",
        )

        original_file.close()

        detection_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".png",
        )

        detection_image = Image.fromarray(
            annotated
        )

        detection_image.save(
            detection_file.name,
            format="PNG",
        )

        detection_file.close()

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
                height=210,
            )
        )

        content.append(
            Spacer(1, 15)
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
                height=210,
            )
        )

        content.append(
            Spacer(1, 20)
        )

        content.append(
            Paragraph(
                "Disclaimer: This report is intended for educational and research purposes only and is not a medical diagnosis.",
                styles["Normal"]
            )
        )

        document.build(
            content
        )

        pdf.seek(0)

        st.download_button(
            "⬇️ Download Analysis Report",
            pdf,
            "brain_tumor_report.pdf",
            "application/pdf",
        )
st.markdown("---")
 

st.markdown("""
<style>
.project-resources-title {
    color: #1E3A8A;
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="project-resources-title">Project Resources</div>',
    unsafe_allow_html=True
)

st.write(
    "For the complete source code, model results, evaluation metrics, "
    "implementation details, and project documentation, "
    "please visit our organization repository."
)

st.link_button(
    "View Project Repository →",
    "https://github.com/BrainVisionAI/Brain-tumor-detection"
)
st.markdown("""
<style>
.developed-title {
    color: #1E3A8A;
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="developed-title">Developed By</div>',
    unsafe_allow_html=True
)








import streamlit as st

col1, col2 = st.columns([0.7, 2])

with col1:
    st.link_button(
        "Shaghayegh Arzani",
        "https://github.com/Shaghayegha76",
        icon=":material/code:",
    )

with col2:
    st.link_button(
        "Mahshid Helaleh",
        "https://github.com/Mahshid04",
        icon=":material/code:",
    )
    st.markdown("""
<style>
a {
    color: #60A5FA !important;
}

a:hover {
    color:#60A5FA !important;
}
</style>
""", unsafe_allow_html=True)

 