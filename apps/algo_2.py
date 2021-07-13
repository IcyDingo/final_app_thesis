import streamlit as st
import streamlit as st
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px 
from itertools import combinations
from sklearn.decomposition import PCA
import random
import ast

def app():

    st.title("Algo 2")
###########################################################################################################################################################################################
    def calculate_pdi_weights( returns,return_mean_range): 

        n = len(returns.columns)
        eq = [1/n]*n
        w = []
        w.append(eq)
        for i in range(1,20000):
            weights = [random.random() for _ in range(n)]
            sum_weights = sum(weights)
            weights = [1*w/sum_weights for w in weights]
            w.append(list(np.round(weights,2)))
        weights_new = []
        for i in w:
            if i not in weights_new:
                weights_new.append(i)


        def meanRetAn(data):             
            Result = 1
            
            for i in data:
                Result *= (1+i)
                
            Result = Result**(1/float(len(data)/return_mean_range))-1
            
            return(Result)

        pca = PCA()
        PDI_dict = {}

        for y,num in tqdm(zip(weights_new, range(0,len(weights_new),1))):
            
            port_ret  = returns.mul(y,axis=1).sum(axis=1)

            ann_ret = meanRetAn(list(port_ret))
            an_cov = returns.cov()
            port_std = np.sqrt(np.dot(np.array(y).T, np.dot(an_cov, y)))*np.sqrt(return_mean_range)
            corr_matrix = np.array(returns.mul(y).cov())
            principalComponents = pca.fit(corr_matrix)
            PDI = 2*sum(principalComponents.explained_variance_ratio_*range(1,len(principalComponents.explained_variance_ratio_)+1,1))-1

            PDI_dict[num ] = {}
            PDI_dict[num ]["PDI_INDEX"] = PDI
            PDI_dict[num ]["# of Assets"] = len(y)
            PDI_dict[num ]["Sharpe Ratio"] = ann_ret/port_std
            PDI_dict[num ]["Annual Return"] = ann_ret
            PDI_dict[num ]["weights"] = y
            PDI_dict[num ]["Annual STD"] = port_std

        df = pd.DataFrame(PDI_dict).T
        df["PDI_INDEX"] = df["PDI_INDEX"].astype(float)
        df["Sharpe Ratio"] = df["Sharpe Ratio"].astype(float)
        df["Annual Return"] = df["Annual Return"].astype(float)
        df["Annual STD"] = df["Annual STD"].astype(float)

        return df

 ################################################################################################################################################################################
