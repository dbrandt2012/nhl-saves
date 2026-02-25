"""NHL Saves Dashboard — entry point."""

import streamlit as st

st.set_page_config(page_title="NHL Saves Dashboard", layout="wide")

pg = st.navigation(
    [
        st.Page("pages/1_Team_Table.py", title="Team Table"),
        st.Page("pages/2_Goalie_Detail.py", title="Goalie Detail"),
    ]
)
pg.run()
