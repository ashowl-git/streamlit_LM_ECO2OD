# 분석전에 필요한 라이브러리들을 불러오기
# 테스트
# plotly라이브러리가 없다면 아래 설치
# conda install -c plotly plotly=4.12.0
# conda install -c conda-forge cufflinks-py
# conda install seaborn

import glob 
import os
import sys, subprocess
from subprocess import Popen, PIPE
import numpy as np
import pandas as pd

import streamlit as st
import sklearn
import seaborn as sns
# sns.set(font="D2Coding") 
# sns.set(font="Malgun Gothic") 
# from IPython.display import set_matplotlib_formats
# set_matplotlib_formats("retina")
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go 
import chart_studio.plotly as py
import cufflinks as cf
# # get_ipython().run_line_magic('matplotlib', 'inline')


# # Make Plotly work in your Jupyter Notebook
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
# init_notebook_mode(connected=True)
# # Use Plotly locally
cf.go_offline()


# 사이킷런 라이브러리 불러오기 _ 통계, 학습 테스트세트 분리, 선형회귀등
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import r2_score 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_log_error






# import streamlit as st

# def main_page():
#     st.markdown("# Main page 🎈")
#     st.sidebar.markdown("# Main page 🎈")

# def page2():
#     st.markdown("# Page 2 ❄️")
#     st.sidebar.markdown("# Page 2 ❄️")

# def page3():
#     st.markdown("# Page 3 🎉")
#     st.sidebar.markdown("# Page 3 🎉")

# page_names_to_funcs = {
#     "Main Page": main_page,
#     "Page 2": page2,
#     "Page 3": page3,
# }

# selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
# page_names_to_funcs[selected_page]()


# # hide the hamburger menu? hidden or visible
hide_menu_style = """
        <style>
        #MainMenu {visibility: visible;}
        footer {visibility: visible;}
        footer:after {content:'Copyright 2023. EAN TECHNOLOGY Corp. All rights reserved.';
        display:block;
        opsition:relatiive;
        color:orange; #tomato
        padding:5px;
        top:100px;}

        </style>
        """

st.set_page_config(layout="wide", page_title="국토안전관리원_온실가스감축계수")
st.markdown(hide_menu_style, unsafe_allow_html=True) # hide the hamburger menu?



# st.subheader('국토안전관리원 온실가스절감 예측')
# st.markdown('Streamlit is **_really_ cool**.')
st.markdown('국토안전관리원 **_온실가스절감량_ 예측**.')
st.markdown('요약')
st.markdown('01_ ECO2OD 아이템별, 아이템조합 시뮬레이션 ')
st.markdown('02_ 머신러닝 회귀모델 구축')
st.markdown('03_ 회귀모델로 소요량 예측 개선전(BASE), 개선후(ALT)')
st.markdown('04_ 사용처별 연료비율 정의')
st.markdown('05_ 에너지소요량과 정의된 연료비율에 따라 이산화탄소(CO2), 메탄(CH4), 아산화질소(N2O) 발생량 산정')
st.markdown('06_ 온실가스 배출량 산정 (개전전-개선후) ')



col1, col2, col3, col4 = st.columns(4)

with col1:
   st.header("국토안전관리원")
   st.image("image/국토안전관리원.png")

with col2:
   st.header("동아대학교")
   st.image("image/동아대학교.jpg")

with col3:
   st.header("서울대학교")
   st.image("image/서울대학교.jpeg")

with col4:
   st.header("(주)이에이엔테크놀로지")
   st.image("image/ean.png")

