import streamlit as st
import pickle
import numpy as np
import requests

st.set_page_config(page_title="AI Crop Prediction", layout="wide")

# -------- LOAD MODEL FILES --------
model = pickle.load(open("model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))
encoder = pickle.load(open("encoder.pkl","rb"))

# -------- WEATHER API --------
API_KEY = "615b0efa6840460fabd7514923384ecd"

def get_weather():
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q=Rajkot&appid={API_KEY}&units=metric"
        data = requests.get(url).json()

        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        rain = data.get("rain",{}).get("1h",0)

        return temp, humidity, rain
    except:
        return "N/A","N/A","N/A"

temp_auto, hum_auto, rain_auto = get_weather()

# -------- UI DESIGN --------
st.markdown("""
<style>
.stApp{
background-image: url("https://images.unsplash.com/photo-1500382017468-9049fed747ef");
background-size: cover;
background-position: center;
background-attachment: fixed;
}

.title{
font-size:60px;
font-weight:bold;
color:white;
text-align:center;
text-shadow:2px 2px 10px black;
}

.weather{
font-size:22px;
color:white;
text-align:center;
margin-bottom:25px;
}

.result{
background:rgba(0,0,0,0.6);
padding:25px;
border-radius:20px;
color:white;
text-align:center;
font-size:26px;
margin:15px;
}
</style>
""",unsafe_allow_html=True)

st.markdown('<div class="title">🌾 AI Crop Prediction</div>',unsafe_allow_html=True)

# -------- WEATHER DISPLAY --------
st.markdown(f'<div class="weather">🌡 Temp: {temp_auto}°C | 💧 Humidity: {hum_auto}% | 🌧 Rainfall: {rain_auto} mm</div>',unsafe_allow_html=True)

st.write("## Enter Field Values")

# -------- INPUT (NO + - BUTTONS) --------
c1,c2,c3 = st.columns(3)

with c1:
    n = st.text_input("Nitrogen")
    p = st.text_input("Phosphorus")
    k = st.text_input("Potassium")

with c2:
    temp = st.text_input("Temperature")
    hum = st.text_input("Humidity")
    ph = st.text_input("Soil pH")

with c3:
    rain = st.text_input("Rainfall")

st.write("")

# -------- PREDICTION --------
if st.button("🌱 Predict Best Crop", use_container_width=True):

    if n and p and k and temp and hum and ph and rain:
        try:
            values = np.array([[float(n),float(p),float(k),
                                float(temp),float(hum),float(ph),float(rain)]])
            scaled = scaler.transform(values)

            probs = model.predict_proba(scaled)[0]
            top3 = np.argsort(probs)[-3:][::-1]

            crops = encoder.inverse_transform(top3)
            conf = probs[top3]*100

            st.write("## 🏆 Top Recommended Crops")

            col1,col2,col3 = st.columns(3)

            for i,(crop,cf,col) in enumerate(zip(crops,conf,[col1,col2,col3])):
                with col:
                    st.markdown(f"""
                    <div class="result">
                    Rank {i+1}<br>
                    <b>{crop}</b><br>
                    {cf:.2f}% Suitable
                    </div>
                    """,unsafe_allow_html=True)

        except:
            st.error("Enter valid numeric values")
    else:
        st.error("Please enter all values")
