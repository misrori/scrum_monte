import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

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

        p_vals = {c: int(np.percentile(outcomes, c)) for c in [50, 80, 90, 95]}

        st.markdown("""
        ### 📦 Előrejelzés (valószínűségi becslések)
        """)
        cols = st.columns(4)
        for i, c in enumerate([50, 80, 90, 95]):
            with cols[i]:
                st.metric(label=f"{c}% valószínűségi határ", value=f"{p_vals[c]} sprint")

        # Histogram Plotly
        hist_fig = px.histogram(outcomes, nbins=max(outcomes)-min(outcomes)+1, labels={'value': 'Szükséges sprintek'},
                                title="Monte Carlo eloszlás", opacity=0.75)
        for c in [80, 90, 95]:
            hist_fig.add_vline(x=p_vals[c], line_dash="dot", annotation_text=f"{c}%: {p_vals[c]}",
                               annotation_position="top right")
        hist_fig.update_layout(height=400)
        st.plotly_chart(hist_fig, use_container_width=True)

        # Burnup Chart
        burnup_fig = go.Figure()
        for i in range(min(20, len(burnup_data))):
            burnup_fig.add_trace(go.Scatter(y=burnup_data[i], mode='lines', line=dict(width=1), opacity=0.4,
                                            showlegend=False))
        burnup_fig.add_hline(y=total_points, line_dash="dot", line_color="red", annotation_text="Cél", 
                             annotation_position="bottom right")
        burnup_fig.update_layout(title="Szimulált Burnup Chart (20 minta)", height=400,
                                 xaxis_title="Sprint", yaxis_title="Elvégzett story point")
        st.plotly_chart(burnup_fig, use_container_width=True)

        # Letölthető riport
        df_outcomes = pd.DataFrame({"simulated_sprints": outcomes})
        csv = df_outcomes.to_csv(index=False)
        st.download_button("⬇️ Eredmények letöltése CSV-ben", csv,
                           file_name="montecarlo_sprint_forecast.csv", mime='text/csv')

except Exception as e:
    st.warning(f"Hiba az adatfeldolgozásban: {e}")
