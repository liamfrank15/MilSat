import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



# Streamlit app
st.title("Deep Learning for Military Aircraft Recognition in Satellite Imagery")

# Load the model
model = tf.keras.models.load_model('MilSat224.keras')

# Create a mapping for class indices to class names
class_indices = {
    0: 'A-10',  
    1: 'B-1',
    2: 'B-2',
    3: 'B-52',
    4: 'Bareland',
    5: 'C-130',
    6: 'K/C-135',
    7: 'C-17',
    8: 'C-5',
    9: 'E-3',
    10: 'KC-10',
}

#################################################################

ellsworth = """
<iframe src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d38111.699620673004!2d-103.09450059999999!3d44.14832345!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x877d642f49c32e97%3A0x4f8203477acdad6!2sEllsworth%20AFB%2C%20SD!5e1!3m2!1sen!2sus!4v1730681563874!5m2!1sen!2sus" width="600" height="450" style="border:0;" allowfullscreen="" loading="lazy" referrerpolicy="no-referrer-when-downgrade"></iframe>
"""
macdill = """
<iframe src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d5870.309900677768!2d-82.51399483965027!3d27.8498222!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x88c2dc129ffb9399%3A0x67dc96635a5e8d09!2sMacDill%20Air%20Force%20Base!5e1!3m2!1sen!2sus!4v1730687999201!5m2!1sen!2sus" width="600" height="450" style="border:0;" allowfullscreen="" loading="lazy" referrerpolicy="no-referrer-when-downgrade"></iframe>
"""

whiteman = """
<iframe src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d5870.309900677768!2d-82.51399483965027!3d27.8498222!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x87c3f05b0abeb0a3%3A0xd38015c25b32833!2sWhiteman%20AFB%2C%20MO!5e1!3m2!1sen!2sus!4v1730688271515!5m2!1sen!2sus" width="600" height="450" style="border:0;" allowfullscreen="" loading="lazy" referrerpolicy="no-referrer-when-downgrade"></iframe>
"""


elmendorf = """
<iframe src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d10452.239473291558!2d-80.06180593161758!3d32.89275153629513!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x56c895e596b4a2cd%3A0xaecac3abe99899fb!2sElmendorf%20AFB%2C%20Anchorage%2C%20AK%2099506!5e1!3m2!1sen!2sus!4v1730688540286!5m2!1sen!2sus" width="600" height="450" style="border:0;" allowfullscreen="" loading="lazy" referrerpolicy="no-referrer-when-downgrade"></iframe>
"""

minot = """
<iframe src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d35250.26725818198!2d-101.38694567874468!3d48.41992206058344!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x52dedf5d72ee0f1b%3A0x3b5e7fddf7ae4d7b!2sMinot%20AFB%2C%20ND!5e1!3m2!1sen!2sus!4v1730767886822!5m2!1sen!2sus" width="600" height="450" style="border:0;" allowfullscreen="" loading="lazy" referrerpolicy="no-referrer-when-downgrade"></iframe>
"""

ramstein = """
<iframe src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d2158.5909217394637!2d7.593861675701477!3d49.43985965961235!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x479674d1b564a3b7%3A0x8ab9f92dc7234016!2sRamstein%20Air%20Base!5e1!3m2!1sen!2sus!4v1730767929043!5m2!1sen!2sus" width="600" height="450" style="border:0;" allowfullscreen="" loading="lazy" referrerpolicy="no-referrer-when-downgrade"></iframe>
"""

mcguire = """
<iframe src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d20331.293925726335!2d-74.60153618589806!3d40.04268493103097!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x89c169d2e1d25913%3A0xeb78f09a99dfff40!2sMcGuire%20AFB%2C%20NJ!5e1!3m2!1sen!2sus!4v1730767979872!5m2!1sen!2sus" width="600" height="450" style="border:0;" allowfullscreen="" loading="lazy" referrerpolicy="no-referrer-when-downgrade"></iframe>
"""

#######################
eielson = """
<iframe src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d4164.99394648169!2d-147.1003754338114!3d64.67123668249974!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x5133ae29c272882b%3A0x9ccf7099f6eaa0bc!2sEielson%20AFB%2C%20AK!5e1!3m2!1sen!2sus!4v1730788026094!5m2!1sen!2sus" width="600" height="450" style="border:0;" allowfullscreen="" loading="lazy" referrerpolicy="no-referrer-when-downgrade"></iframe>
"""

beale = """
<iframe src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d8612.611688386025!2d-121.44063420780192!3d39.14948325482037!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x809b5ab98e960ed9%3A0x5bbd09d077cfb2f6!2sBeale%20AFB%20Altitude%20Chamber!5e1!3m2!1sen!2sus!4v1730788271606!5m2!1sen!2sus" width="600" height="450" style="border:0;" allowfullscreen="" loading="lazy" referrerpolicy="no-referrer-when-downgrade"></iframe>
"""

edwards = """
<iframe src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d87111.36435643482!2d-117.99890997355708!3d34.91176101683871!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x80c24ab03a29c3eb%3A0xdf7023bbd91346e8!2sEdwards%20AFB%2C%20CA!5e1!3m2!1sen!2sus!4v1730788324987!5m2!1sen!2sus" width="600" height="450" style="border:0;" allowfullscreen="" loading="lazy" referrerpolicy="no-referrer-when-downgrade"></iframe>
"""

vandenberg = """
<iframe src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d2727.8463002884205!2d-120.57501532516636!3d34.742031080995545!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x80ec3ce573ed3d19%3A0xd7ea9c09f025e90f!2sVandenberg%20Air%20Force%20Base!5e1!3m2!1sen!2sus!4v1730788447544!5m2!1sen!2sus" width="600" height="450" style="border:0;" allowfullscreen="" loading="lazy" referrerpolicy="no-referrer-when-downgrade"></iframe>
"""

andrews = """
<iframe src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d24737.659739065628!2d-157.96942103849318!3d21.332504520803475!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x89b7bdb52dcb20fb%3A0xb31f9c33c23769e!2sJoint%20Base%20Andrews%2C%20MD!5e1!3m2!1sen!2sus!4v1730788760091!5m2!1sen!2sus" width="600" height="450" style="border:0;" allowfullscreen="" loading="lazy" referrerpolicy="no-referrer-when-downgrade"></iframe>
"""

cannon = """
<iframe src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d24737.659739065628!2d-157.96942103849318!3d21.332504520803475!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x8702da336a2841a9%3A0x13f68e8433731950!2sCannon%20AFB%2C%20NM!5e1!3m2!1sen!2sus!4v1730788969631!5m2!1sen!2sus" width="600" height="450" style="border:0;" allowfullscreen="" loading="lazy" referrerpolicy="no-referrer-when-downgrade"></iframe>
"""

altus = """
<iframe src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d21741.41248789711!2d-106.58661928916014!3d35.0488506!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x87ab648321064ad7%3A0x8bf734cfb6edbb2b!2sAltus%20AFB%2C%20Altus%2C%20OK!5e1!3m2!1sen!2sus!4v1730789210719!5m2!1sen!2sus" width="600" height="450" style="border:0;" allowfullscreen="" loading="lazy" referrerpolicy="no-referrer-when-downgrade"></iframe>
"""

andersen = """
<iframe src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d242.8070931381933!2d-99.84954660273804!3d32.42180675234142!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x671f85cea800343b%3A0xf160f647c746ca24!2sAndersen%20AFB%2C%20Guam!5e1!3m2!1sen!2sus!4v1730789518680!5m2!1sen!2sus" width="600" height="450" style="border:0;" allowfullscreen="" loading="lazy" referrerpolicy="no-referrer-when-downgrade"></iframe>
"""


col4 = st.columns(1)
with col4[0]: 
    st.subheader("Aircraft Types Included in Model Training:")
    st.markdown("""
    - A-10
    - B-1
    - B-2
    - B-52
    - C-130
    - K/C-135
    - K/C-10
    - C-17
    - C-5
    - E-3
    - Bare Terrain
    """)

    st.subheader("Detailed Workflow:")
    st.write("1. Select an available base from the dropdown menu or navigate to a base using google maps")
    st.write("2. Identify a suitable aircraft parked on the ramp")
    st.write("3. Use a capture tool to screenshot a satellite image of selected singular aircraft")
    st.write("4. Upload your captured image")
    st.write("5. View model prediction of aircraft type")
    selected_base = st.selectbox("Select a base to view satellite imagery:", ["Ellsworth AFB","MacDill AFB","Whiteman AFB",
                                                                         "Elmendorf AFB","Minot AFB","Ramstein AB",
                                                                         "McGuire AFB","Eielson AFB","Beale AFB",
                                                                         "Edwards AFB","Vandenberg AFB","Andrews AFB",
                                                                         "Cannon AFB","Altus AFB","Andersen (Guam)"])

    # Display the selected map
    if selected_base == "Ellsworth AFB":
        st.markdown(ellsworth, unsafe_allow_html=True)
    elif selected_base == "MacDill AFB":
        st.markdown(macdill, unsafe_allow_html=True)
    elif selected_base == "Whiteman AFB":
        st.markdown(whiteman, unsafe_allow_html=True)
    elif selected_base == "Elmendorf AFB":
        st.markdown(elmendorf, unsafe_allow_html=True)
    elif selected_base == "Minot AFB":
        st.markdown(minot, unsafe_allow_html=True)
    elif selected_base == "Ramstein AB":
        st.markdown(ramstein, unsafe_allow_html=True)
    elif selected_base == "McGuire AFB":
        st.markdown(mcguire, unsafe_allow_html=True)
    elif selected_base == "Eielson AFB":
        st.markdown(eielson, unsafe_allow_html=True)
    elif selected_base == "Beale AFB":
        st.markdown(beale, unsafe_allow_html=True)
    elif selected_base == "Edwards AFB":
        st.markdown(edwards, unsafe_allow_html=True)
    elif selected_base == "Vandenberg AFB":
        st.markdown(vandenberg, unsafe_allow_html=True)
    elif selected_base == "Andrews AFB":
        st.markdown(andrews, unsafe_allow_html=True)
    elif selected_base == "Cannon AFB":
        st.markdown(cannon, unsafe_allow_html=True)
    elif selected_base == "Altus AFB":
        st.markdown(altus, unsafe_allow_html=True)
    elif selected_base == "Andersen (Guam)":
        st.markdown(andersen, unsafe_allow_html=True)

    # Upload image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open and display the uploaded image
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption='Uploaded Image', width=300)

        # Preprocess the image
        img = img.resize((224, 224))  # Resize to 128x128
        img_array = image.img_to_array(img)  # Convert to array
        img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions
        img_array /= 255.0  # Normalize if needed

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_name = class_indices.get(predicted_class_index, "Unknown Class")

        # Display the result
        st.metric("Predicted Aircraft:", predicted_class_name)
