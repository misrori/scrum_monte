import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Scrum Monte Carlo Forecast", layout="wide")
st.title("🌀 Monte Carlo Forecast for Scrum Masters")

st.markdown("""
Ez az eszköz előrejelzi, hány sprint szükséges a projekt befejezéséhez. Figyelembe veszi:
- történelmi sebességet,
- bizonytalanságot (szórás),
- kieső napokat,
- sebesség trendet,
- és a kiválasztott megbízhatósági szintet.
""")

col1, col2 = st.columns(2)

with col1:
    total_points = st.number_input("📦 Összes hátralévő story point", min_value=1, value=200)
    sprints_input = st.text_area("📈 Korábbi sprint sebességek", value="20,22,18,25,17,23,21", height=100)
    days_blocked = st.slider("🚧 Átlagos blokkolt napok aránya sprintekben (%)", 0, 50, 10)

with col2:
    n_simulations = st.slider("🔁 Szimulációk száma", 1000, 50000, 10000, step=1000)
    confidence = st.selectbox("✅ Megbízhatósági szint", [80, 90, 95])
    consider_trend = st.checkbox("📉 Figyelembe vesszük a sebesség trendjét?", value=True)

try:
    past_velocities = np.array(list(map(float, sprints_input.strip().split(","))))
    mean_velocity = np.mean(past_velocities)
    std_velocity = np.std(past_velocities)

    if consider_trend and len(past_velocities) >= 3:
        weights = np.linspace(0.5, 1.5, len(past_velocities))
        mean_velocity = np.average(past_velocities, weights=weights)
        st.info(f"📊 Trend figyelembevételével súlyozott átlag: {mean_velocity:.2f} SP/sprint")

    blocked_factor = 1 - (days_blocked / 100)

    if st.button("▶️ Szimuláció futtatása"):
        outcomes = []
        burnup_data = []

        for _ in range(n_simulations):
            remaining = total_points
            sprints = 0
            sprint_progress = [0]
            while remaining > 0:
                sampled_velocity = max(1, np.random.normal(mean_velocity, std_velocity)) * blocked_factor
                remaining -= sampled_velocity
                sprint_progress.append(total_points - max(0, remaining))
                sprints += 1
            outcomes.append(sprints)
            burnup_data.append(sprint_progress)

        # Eredmény statisztikák
        p_val = int(np.percentile(outcomes, confidence))
        st.success(f"✅ {confidence}%-os valószínűséggel {p_val} sprint alatt kész lesz a projekt.")

        # Histogram
        st.subheader("⏳ Sprint szükséglet eloszlása")
        fig1, ax1 = plt.subplots()
        ax1.hist(outcomes, bins=range(min(outcomes), max(outcomes)+1), edgecolor='black')
        ax1.set_xlabel("Szükséges sprintek")
        ax1.set_ylabel("Gyakoriság")
        ax1.set_title("Monte Carlo eloszlás")
        st.pyplot(fig1)

        # Burnup chart
        st.subheader("📈 Szimulált Burnup Chart (20 minta)")
        fig2, ax2 = plt.subplots()
        for i in range(min(20, len(burnup_data))):
            ax2.plot(burnup_data[i], alpha=0.4)
        ax2.axhline(y=total_points, color="red", linestyle="--", label="Cél")
        ax2.set_title("Burnup Chart szimuláció")
        ax2.set_xlabel("Sprint")
        ax2.set_ylabel("Elvégzett story point")
        ax2.legend()
        st.pyplot(fig2)

        # Letölthető riport
        df_outcomes = pd.DataFrame({"simulated_sprints": outcomes})
        csv = df_outcomes.to_csv(index=False)
        st.download_button("⬇️ Eredmények letöltése CSV-ben", csv, file_name="montecarlo_sprint_forecast.csv", mime='text/csv')

except Exception as e:
    st.warning(f"Hiba az adatfeldolgozásban: {e}")
