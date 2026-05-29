import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Pokémon Analytics Dashboard",
    page_icon="⚡",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #FAF3E0 0%, #F7EDE2 45%, #E8F3F1 100%);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFF8E7 0%, #E6F4F1 100%);
    border-right: 1px solid #DDD6C5;
}

.main-title {
    font-size: 46px;
    font-weight: 850;
    color: #2F2E41;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    color: #5F5B52;
    margin-bottom: 25px;
}

.hero-card {
    padding: 26px;
    border-radius: 22px;
    background: rgba(255, 255, 255, 0.72);
    border: 1px solid #E6DCCB;
    box-shadow: 0 8px 22px rgba(84, 72, 49, 0.08);
    margin-bottom: 25px;
}

.soft-card {
    padding: 18px;
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.68);
    border: 1px solid #E6DCCB;
    box-shadow: 0 6px 16px rgba(84, 72, 49, 0.06);
    margin-bottom: 18px;
}

.insight {
    padding: 15px 18px;
    border-left: 5px solid #C97C5D;
    background-color: #FFF8F0;
    border-radius: 10px;
    margin-top: 10px;
    margin-bottom: 25px;
    color: #3D3A35;
}

.prediction-box {
    padding: 22px;
    border-radius: 18px;
    background: #FFF8F0;
    border: 1px solid #E6DCCB;
    box-shadow: 0 6px 16px rgba(84, 72, 49, 0.08);
}

.pokemon-card {
    text-align: center;
    padding: 15px;
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.75);
    border: 1px solid #E6DCCB;
    box-shadow: 0 6px 16px rgba(84, 72, 49, 0.06);
    margin-bottom: 15px;
}

[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.72);
    padding: 18px;
    border-radius: 16px;
    border: 1px solid #E6DCCB;
    box-shadow: 0 5px 14px rgba(84, 72, 49, 0.06);
}

[data-testid="stMetricValue"] {
    color: #B85C38;
    font-weight: 800;
}

h1, h2, h3 {
    color: #2F2E41;
}

div.stButton > button {
    background-color: #B85C38;
    color: white;
    border-radius: 12px;
    padding: 0.65rem 1.2rem;
    border: none;
    font-weight: 600;
}

div.stButton > button:hover {
    background-color: #8F452A;
    color: white;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv("pokemon.csv")
    df["type2"] = df["type2"].fillna("None")
    df["percentage_male"] = df["percentage_male"].fillna(df["percentage_male"].median())
    df["height_m"] = df["height_m"].fillna(df["height_m"].median())
    df["weight_kg"] = df["weight_kg"].fillna(df["weight_kg"].median())
    return df


df = load_data()


def get_pokemon_image(pokedex_number):
    return f"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/{int(pokedex_number)}.png"


features = [
    "hp",
    "attack",
    "defense",
    "sp_attack",
    "sp_defense",
    "speed",
    "base_total"
]


@st.cache_resource
def train_model(data):
    X = data[features]
    y = data["is_legendary"]

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    return model


model = train_model(df)

page = st.sidebar.radio(
    "Navigation",
        [
         "Home",
         "Visualizations",
         "Prediction",
         "Pokémon Search",
         "Top Pokémon",
         "Model Insights"
        ]
    
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Dataset")
st.sidebar.write("Pokémon Dataset")
st.sidebar.write("801 Records | 41 Attributes")


if page == "Home":

    col_title, col_img = st.columns([4, 1])

    with col_title:
        st.markdown(
            '<div class="main-title">Pokémon Data Analysis Dashboard</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="subtitle">A professional EDA and machine learning dashboard for Pokémon statistics.</div>',
            unsafe_allow_html=True
        )

    with col_img:
        st.image(get_pokemon_image(25), width=130)

    st.markdown('<div class="hero-card">', unsafe_allow_html=True)
    st.write(
        """
        This dashboard analyzes Pokémon characteristics such as type, HP, attack, defense,
        speed, generation, and legendary status. It includes visual analysis, top Pokémon ranking,
        and a machine learning model that predicts whether a Pokémon is likely to be legendary.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Pokémon", df.shape[0])
    col2.metric("Total Attributes", df.shape[1])
    col3.metric("Legendary Pokémon", int(df["is_legendary"].sum()))
    col4.metric("Non-Legendary", int(df.shape[0] - df["is_legendary"].sum()))

    st.write("")
    st.subheader("Dataset Preview")
    st.dataframe(df.head(), width="stretch")

    st.subheader("Sample Pokémon Gallery")

    sample_pokemon = df[["pokedex_number", "name", "type1", "type2"]].head(5)
    gallery_cols = st.columns(5)

    for i, row in sample_pokemon.iterrows():
        with gallery_cols[i]:
            st.markdown('<div class="pokemon-card">', unsafe_allow_html=True)
            st.image(get_pokemon_image(row["pokedex_number"]), width=95)
            st.markdown(f"**{row['name'].title()}**")
            st.caption(f"{row['type1']} / {row['type2']}")
            st.markdown('</div>', unsafe_allow_html=True)


elif page == "Visualizations":

    st.title("Pokémon Visual Analysis")

    st.header("1. Pokémon Count by Primary Type")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(
        data=df,
        y="type1",
        order=df["type1"].value_counts().index,
        palette="crest",
        ax=ax
    )
    ax.set_title("Pokémon Count by Primary Type")
    ax.set_xlabel("Count")
    ax.set_ylabel("Primary Type")
    st.pyplot(fig)

    st.markdown(
        """
        <div class="insight">
        <b>Insight:</b> Water-type Pokémon are the most common primary type,
        followed by Normal and Grass types. Flying appears rarely as a primary type.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.header("2. Attack Distribution")

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(df["attack"], bins=20, kde=True, color="#B85C38", ax=ax)
    ax.set_title("Distribution of Attack Values")
    ax.set_xlabel("Attack")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    st.markdown(
        """
        <div class="insight">
        <b>Insight:</b> Most Pokémon have medium attack values, while extremely high
        attack values are less common.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.header("3. HP vs Attack")

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.scatterplot(
        data=df,
        x="hp",
        y="attack",
        hue="is_legendary",
        palette=["#6B8F71", "#B85C38"],
        ax=ax
    )
    ax.set_title("HP vs Attack")
    ax.set_xlabel("HP")
    ax.set_ylabel("Attack")
    st.pyplot(fig)

    st.markdown(
        """
        <div class="insight">
        <b>Insight:</b> Higher HP does not always mean higher attack.
        Legendary Pokémon are generally located toward stronger stat regions.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.header("4. Legendary vs Non-Legendary Pokémon")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(
        data=df,
        x="is_legendary",
        palette=["#6B8F71", "#B85C38"],
        ax=ax
    )
    ax.set_title("Legendary vs Non-Legendary Pokémon")
    ax.set_xlabel("Legendary Status")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.markdown(
        """
        <div class="insight">
        <b>Insight:</b> Legendary Pokémon represent only a small portion of the dataset,
        which means the dataset is imbalanced.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.header("5. Base Total by Generation")

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(
        data=df,
        x="generation",
        y="base_total",
        palette="Set2",
        ax=ax
    )
    ax.set_title("Base Total Stats by Generation")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Base Total")
    st.pyplot(fig)

    st.markdown(
        """
        <div class="insight">
        <b>Insight:</b> Base Total varies across generations, showing that each generation
        contains Pokémon with different strength ranges.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.header("6. Correlation Heatmap")

    stats_cols = [
        "hp",
        "attack",
        "defense",
        "sp_attack",
        "sp_defense",
        "speed",
        "base_total"
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[stats_cols].corr(), annot=True, cmap="BrBG", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

    st.markdown(
        """
        <div class="insight">
        <b>Insight:</b> Base Total has a strong relationship with Attack,
        Special Attack, Special Defense, Defense, HP, and Speed because it is calculated
        from these battle statistics.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.header("7. Average Base Total by Primary Type")

    type_power = df.groupby("type1")["base_total"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x=type_power.values,
        y=type_power.index,
        palette="mako",
        ax=ax
    )
    ax.set_title("Average Base Total by Primary Type")
    ax.set_xlabel("Average Base Total")
    ax.set_ylabel("Primary Type")
    st.pyplot(fig)

    st.markdown(
        """
        <div class="insight">
        <b>Insight:</b> Some Pokémon types have higher average total stats than others,
        suggesting differences in overall strength across types.
        </div>
        """,
        unsafe_allow_html=True
    )


elif page == "Prediction":

    st.title("Legendary Pokémon Predictor")

    st.write(
        "Enter Pokémon battle statistics below. The trained Random Forest model will predict whether the Pokémon is likely legendary."
    )

    col1, col2 = st.columns(2)

    with col1:
        hp = st.number_input("HP", 0, 300, 80)
        attack = st.number_input("Attack", 0, 300, 80)
        defense = st.number_input("Defense", 0, 300, 80)
        sp_attack = st.number_input("Special Attack", 0, 300, 80)

    with col2:
        sp_defense = st.number_input("Special Defense", 0, 300, 80)
        speed = st.number_input("Speed", 0, 300, 80)
        base_total = st.number_input("Base Total", 0, 1000, 480)

    if st.button("Predict Legendary Status"):

        input_values = [[
            hp,
            attack,
            defense,
            sp_attack,
            sp_defense,
            speed,
            base_total
        ]]

        prediction = model.predict(input_values)
        probability = model.predict_proba(input_values)[0]

        df_copy = df.copy()

        df_copy["similarity_score"] = (
            abs(df_copy["hp"] - hp) +
            abs(df_copy["attack"] - attack) +
            abs(df_copy["defense"] - defense) +
            abs(df_copy["sp_attack"] - sp_attack) +
            abs(df_copy["sp_defense"] - sp_defense) +
            abs(df_copy["speed"] - speed) +
            abs(df_copy["base_total"] - base_total)
        )

        similar = df_copy.sort_values("similarity_score").head(1).iloc[0]

        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)

        if prediction[0] == 1:
            st.success("Prediction: Likely Legendary Pokémon")
        else:
            st.info("Prediction: Likely Non-Legendary Pokémon")

        st.write(f"Non-Legendary Probability: {probability[0]:.2f}")
        st.write(f"Legendary Probability: {probability[1]:.2f}")

        st.markdown('</div>', unsafe_allow_html=True)

        st.subheader("Most Similar Pokémon in Dataset")

        img_col, info_col = st.columns([1, 3])

        with img_col:
            st.image(
                get_pokemon_image(similar["pokedex_number"]),
                width=150
            )

        with info_col:
            st.markdown(f"### {similar['name'].title()}")
            st.write(f"Type: {similar['type1']} / {similar['type2']}")
            st.write(f"HP: {similar['hp']}")
            st.write(f"Attack: {similar['attack']}")
            st.write(f"Speed: {similar['speed']}")
            st.write(f"Base Total: {similar['base_total']}")
            st.write(
                f"Legendary Status: {'Legendary' if similar['is_legendary'] == 1 else 'Non-Legendary'}"
            )


elif page == "Pokémon Search":

    st.title("Pokémon Search & Stat Radar")

    st.write(
        "Search for a Pokémon to view its image, type, battle statistics, and radar chart."
    )

    pokemon_names = sorted(df["name"].str.title().tolist())

    selected_name = st.selectbox(
        "Choose a Pokémon",
        pokemon_names
    )

    selected_pokemon = df[
        df["name"].str.title() == selected_name
    ].iloc[0]

    col1, col2 = st.columns([1, 3])

    with col1:
        st.image(
            get_pokemon_image(selected_pokemon["pokedex_number"]),
            width=180
        )

    with col2:
        st.markdown(f"## {selected_pokemon['name'].title()}")
        st.write(f"**Type:** {selected_pokemon['type1']} / {selected_pokemon['type2']}")
        st.write(f"**Generation:** {selected_pokemon['generation']}")
        st.write(f"**Legendary Status:** {'Legendary' if selected_pokemon['is_legendary'] == 1 else 'Non-Legendary'}")
        st.write(f"**Base Total:** {selected_pokemon['base_total']}")

    st.subheader("Battle Statistics")

    stat_cols = st.columns(6)

    stat_cols[0].metric("HP", int(selected_pokemon["hp"]))
    stat_cols[1].metric("Attack", int(selected_pokemon["attack"]))
    stat_cols[2].metric("Defense", int(selected_pokemon["defense"]))
    stat_cols[3].metric("Sp. Attack", int(selected_pokemon["sp_attack"]))
    stat_cols[4].metric("Sp. Defense", int(selected_pokemon["sp_defense"]))
    stat_cols[5].metric("Speed", int(selected_pokemon["speed"]))

    st.subheader("Radar Chart")

    radar_labels = [
        "HP",
        "Attack",
        "Defense",
        "Sp. Attack",
        "Sp. Defense",
        "Speed"
    ]

    radar_values = [
        selected_pokemon["hp"],
        selected_pokemon["attack"],
        selected_pokemon["defense"],
        selected_pokemon["sp_attack"],
        selected_pokemon["sp_defense"],
        selected_pokemon["speed"]
    ]

    angles = np.linspace(
        0,
        2 * np.pi,
        len(radar_labels),
        endpoint=False
    ).tolist()

    radar_values += radar_values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(
        figsize=(6, 6),
        subplot_kw=dict(polar=True)
    )

    ax.plot(
        angles,
        radar_values,
        linewidth=2
    )

    ax.fill(
        angles,
        radar_values,
        alpha=0.25
    )

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels)

    ax.set_title(
        f"{selected_pokemon['name'].title()} Battle Stat Radar",
        size=14,
        pad=20
    )

    ax.set_ylim(0, 200)

    st.pyplot(fig)

    st.markdown(
        """
        <div class="insight">
        <b>Insight:</b> The radar chart shows the balance of a Pokémon's battle statistics.
        A wider shape indicates stronger overall stats, while an uneven shape shows that the Pokémon
        is stronger in some areas than others.
        </div>
        """,
        unsafe_allow_html=True
    )

elif page == "Top Pokémon":

    st.title("Top Pokémon Analysis")
    st.header("Top 10 Strongest Pokémon by Base Total")

    top10 = df[
        [
            "pokedex_number",
            "name",
            "type1",
            "type2",
            "hp",
            "attack",
            "defense",
            "speed",
            "base_total",
            "is_legendary"
        ]
    ].sort_values(
        by="base_total",
        ascending=False
    ).head(10).reset_index(drop=True)

    cols = st.columns(5)

    for index, row in top10.iterrows():
        with cols[index % 5]:
            st.markdown('<div class="pokemon-card">', unsafe_allow_html=True)
            st.image(get_pokemon_image(row["pokedex_number"]), width=100)
            st.markdown(f"### {row['name'].title()}")
            st.write(f"{row['type1']} / {row['type2']}")
            st.write(f"Base Total: {row['base_total']}")
            st.caption("Legendary" if row["is_legendary"] == 1 else "Non-Legendary")
            st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("Top 10 Data Table")
    st.dataframe(top10, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=top10,
        x="base_total",
        y="name",
        palette="crest",
        ax=ax
    )
    ax.set_title("Top 10 Strongest Pokémon")
    ax.set_xlabel("Base Total")
    ax.set_ylabel("Pokémon Name")
    st.pyplot(fig)

    st.markdown(
        """
        <div class="insight">
        <b>Insight:</b> Pokémon with the highest Base Total generally have very strong
        overall battle statistics and many of them are legendary or special Pokémon.
        </div>
        """,
        unsafe_allow_html=True
    )


elif page == "Model Insights":

    st.title("Machine Learning Model Insights")

    st.header("Model Used: Random Forest Classifier")

    st.write(
        """
        The Random Forest model was trained to predict whether a Pokémon is legendary
        using HP, Attack, Defense, Special Attack, Special Defense, Speed, and Base Total.
        """
    )

    st.metric("Notebook Model Accuracy", "93.17%")

    importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values(
        by="Importance",
        ascending=False
    )

    st.subheader("Feature Importance")
    st.dataframe(importance, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=importance,
        x="Importance",
        y="Feature",
        palette="mako",
        ax=ax
    )
    ax.set_title("Feature Importance for Legendary Prediction")
    st.pyplot(fig)

    st.markdown(
        """
        <div class="insight">
        <b>Insight:</b> Base Total was the most influential feature for predicting legendary status,
        followed by HP and Special Attack. This shows that overall strength is a major factor
        in identifying legendary Pokémon.
        </div>
        """,
        unsafe_allow_html=True
    )