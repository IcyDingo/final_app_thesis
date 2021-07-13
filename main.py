import streamlit as st
from apps import algo_1, algo_2, intro
from multiApp import MultiApp
#Set Full Page Width
st.set_page_config(layout="wide")
app = MultiApp() # Getting MultiApp


#adding apllicaitons 
app.add_app("Intro", intro.app)
app.add_app("Portfolio Diversification Framework", algo_1.app)
#app.add_app("Check Portfolio Diversification", algo_2.app)

app.run()


