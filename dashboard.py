import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px 
from itertools import combinations
from sklearn.decomposition import PCA
import random

################################################################################# PDI Function #################################################################################
@st.cache(show_spinner=False)
def calculate_pdi(num_assets, tickers, weekly_returns): 
    
    def meanRetAn(data):             
        Result = 1
        
        for i in data:
            Result *= (1+i)
            
        Result = Result**(1/float(len(data)/52))-1
        
        return(Result)

    pca = PCA()
    PDI_dict = {}
    samples = [["SPY"]]
    for number in range(2,num_assets, 1):
        for i in range(1,2000):
            #samples.extend([list(x) for x in combinations(selected_tickers, number_of_assets)])
            samples.append(random.sample(list(tickers),number))
    samples_mini = []
    for i in samples:
        if i not in samples_mini:
            samples_mini.append(i)


    
    for i,y in zip(samples_mini,range(1,len(samples_mini)+1)):
        prog = int(y/len(samples_mini)*100)
        progress_bar.progress(prog)
        status_text.text("{}% Complete".format(prog))
        n_assets = len(i)
        portfolio_weights_ew = np.repeat(1/n_assets, n_assets)
        port_weekly_return = weekly_returns[i].mul(portfolio_weights_ew,axis=1).sum(axis=1)
        ann_ret = meanRetAn(list(port_weekly_return))
        an_cov = weekly_returns[i].cov()
        port_std = np.sqrt(np.dot(portfolio_weights_ew.T, np.dot(an_cov, portfolio_weights_ew)))*np.sqrt(52)
        corr_matrix = np.array(weekly_returns[i].corr())
        principalComponents = pca.fit(corr_matrix)
        PDI = 2*sum(principalComponents.explained_variance_ratio_*range(1,len(principalComponents.explained_variance_ratio_)+1,1))-1
        PDI_dict[y] = {}
        PDI_dict[y]["PDI_INDEX"] = PDI
        PDI_dict[y]["# of Assets"] = len(i)
        PDI_dict[y]["Assets"] = i
        PDI_dict[y]["Sharpe Ratio"] = ann_ret/port_std
        PDI_dict[y]["Annual Return"] = ann_ret
        PDI_dict[y]["Annual STD"] = port_std
    

        


    PDI_DF = pd.DataFrame(PDI_dict).T
    PDI_DF["Assets"] = PDI_DF["Assets"].astype(str)
    PDI_DF["# of Assets"] = PDI_DF["# of Assets"].astype(str)
    PDI_DF["Sharpe Ratio"] = PDI_DF["Sharpe Ratio"].astype(float)
    PDI_DF["Annual STD"] = PDI_DF["Annual STD"].astype(float)
    PDI_DF["PDI_INDEX"] = PDI_DF["PDI_INDEX"].astype(float)
    PDI_DF["Annual Return"] = PDI_DF["Annual Return"].astype(float)
    SPY_DF = PDI_DF.iloc[0,:]
    return PDI_DF,SPY_DF
################################################################################################################################################################################

################################################################################# Trading Strategy #################################################################################
def pca_per(return_data, initial_port, interval, ret_range_mean):
    data = return_data.copy() # data containing weekly returns
    tickers = list(data.columns)
    data.index = pd.to_datetime(data.index) # Conveting the index which is date to datetime
    weeks_list = data.index # grabbing all index dates
    data.index = data.index.to_period(interval) # converting the index to quarterly sets
    periods = data.index.unique() # taking the unique quarters to loop

    
    print(periods)
    first_period = periods[0] # the first period of the time frame
    remaining_periods = periods[1:] # the remianing periods of the time framr
    first_periods = periods[:-1] # all periods minus the last

    def meanRetAn(data):             
        Result = 1
        
        for i in data:
            Result *= (1+i)
            
        Result = Result**(1/float(len(data)/ret_range_mean))-1
        
        return(Result)

    
    def port_ret(returns,port, period): # function for calculating returns
        n_assets = len(port)
        portfolio_weights_ew = np.repeat(1/n_assets, n_assets)
        port_return = returns.loc[period][port].mul(portfolio_weights_ew,axis=1).sum(axis=1)
        return  port_return

    performance_eq = []
    performance_eq_pdi = []
    sharpe_performance_eq_2 = []
    sharpe_performance_eq_2_pdi = []
    sharpe_performance_eq_3 = []
    sharpe_performance_eq_3_pdi = []
    sharpe_performance_eq_max = []
    sharpe_performance_eq_max_pdi =  []
    
    ############################################################ Calculate first period ####################################################################################
    #### Equal weight #####
    print(first_period)
    performance_eq.extend(port_ret(data, initial_port,first_period))
    sharpe_performance_eq_2.extend(port_ret(data, initial_port,first_period))
    sharpe_performance_eq_3.extend(port_ret(data, initial_port,first_period))
    sharpe_performance_eq_max.extend(port_ret(data, initial_port,first_period))
    tickers_weekly = list(data.columns)
    number_of_assets = [len(initial_port)]







    ######################################################## Calculation of portfolio perfomnce #############################################################################

    for init_time, next_time in zip(first_periods,remaining_periods):
        ############ Portfolio Creation ##############################
        samples = [["URTH"]]
        for number in [len(initial_port)]:
            for i in range(1,30000):
                #samples.extend([list(x) for x in combinations(selected_tickers, number_of_assets)])
                samples.append(random.sample(list(tickers),number))
        
        seen = set()
        samples_mini = [x for x in samples if frozenset(x) not in seen and not seen.add(frozenset(x))]

        print("Number of Portfolios: {}".format(len(samples_mini)))
        print("first time: {} - last time: {}".format(init_time, next_time))
        PDI_DF, SPY_DF = calculate_pdi(weekly_returns = data.loc[init_time].dropna(axis=1), portfolios = samples_mini, return_mean_range = ret_range_mean)


        ################################################## Taking the higest PDI ##########################################################################

        id = PDI_DF["PDI_INDEX"].idxmax()
        port_max_pdi = PDI_DF["Assets"][id]
        performance_eq_pdi.append(PDI_DF["PDI_INDEX"][id])
        
        port_max_ret_period = port_ret(data,port_max_pdi, next_time)
        performance_eq.extend(port_max_ret_period)

        ################################################## Taking the higest Sharpe Ration - PDI > 2##########################################################################
        id_sharpe = PDI_DF[PDI_DF["PDI_INDEX"]>2]["Sharpe Ratio"].idxmax()
        port_max_sharpe = PDI_DF["Assets"][id_sharpe]
        sharpe_performance_eq_2_pdi.append(PDI_DF["PDI_INDEX"][id_sharpe])
        
        port_max_ret_period_sharpe = port_ret(data,port_max_sharpe, next_time)
        sharpe_performance_eq_2.extend(port_max_ret_period_sharpe)
                
        ################################################## Taking the higest Sharpe Ration - PDI > 2##########################################################################
        try: 
            try:
                id_sharpe_3= PDI_DF[PDI_DF["PDI_INDEX"]>3]["Sharpe Ratio"].idxmax()
                port_max_sharpe_3 = PDI_DF["Assets"][id_sharpe_3]
                sharpe_performance_eq_3_pdi.append(PDI_DF["PDI_INDEX"][id_sharpe_3])
            except:
                id_sharpe_3= PDI_DF[PDI_DF["PDI_INDEX"]>2.5]["Sharpe Ratio"].idxmax()
                port_max_sharpe_3 = PDI_DF["Assets"][id_sharpe_3]
                sharpe_performance_eq_3_pdi.append(PDI_DF["PDI_INDEX"][id_sharpe_3])
        except:
            id_sharpe_3= PDI_DF["PDI_INDEX"].idxmax()
            port_max_sharpe_3 = PDI_DF["Assets"][id_sharpe_3]
            sharpe_performance_eq_3_pdi.append(PDI_DF["PDI_INDEX"][id_sharpe_3])


        
        port_max_ret_period_sharpe_3 = port_ret(data,port_max_sharpe_3, next_time)
        sharpe_performance_eq_3.extend(port_max_ret_period_sharpe_3)

        ################################################## Taking the higest Sharpe Ration - PDI > 2##########################################################################
        id_sharpe_max = PDI_DF["Sharpe Ratio"].idxmax()
        port_max_sharpe_max = PDI_DF["Assets"][id_sharpe_max]
        sharpe_performance_eq_max_pdi.append(PDI_DF["PDI_INDEX"][id_sharpe_max])
        port_max_ret_period_sharpe_max = port_ret(data,port_max_sharpe_max, next_time)
        sharpe_performance_eq_max.extend(port_max_ret_period_sharpe_max)




    dd = pd.DataFrame()
    dd["Time"] = weeks_list
    dd["per_pdi"] = performance_eq
    dd["per_sharpe_2"] = sharpe_performance_eq_2
    dd["per_sharpe_3"] = sharpe_performance_eq_3
    dd["per_sharpe_max"] = sharpe_performance_eq_max
    dd["URTH"] = list(data["URTH"])
    dd["per_pdi_cum"] = dd["per_pdi"].cumsum(axis=0)
    dd["per_sharpe_cum_2"] = dd["per_sharpe_2"].cumsum(axis=0)
    dd["per_sharpe_cum_3"] = dd["per_sharpe_3"].cumsum(axis=0)
    dd["per_sharpe_cum_max"] = dd["per_sharpe_max"].cumsum(axis=0)
    dd["URTH_cum"] = dd["URTH"].cumsum(axis=0)

    pdi = [performance_eq_pdi,sharpe_performance_eq_2_pdi,sharpe_performance_eq_3_pdi,sharpe_performance_eq_max_pdi]


    return dd










#Set Full Page Width
st.set_page_config(layout="wide")

#Title 
st.title("ETF Funnel")
#Load DataFrames
df = pd.read_csv("ETFs_info.csv", index_col="Ticker")
df["Inception"] = pd.to_datetime(df["Inception"])
df = df[df["Inception"] <= "2015-01-01"]

fundamental_df = pd.read_csv("fund_risk_cluster_reduced.csv",index_col="Ticker") #Fundamental Clustering Data
weekly_return = pd.read_csv("WeeklyReturns.csv",index_col="Date") #Weekly Return Data
fundamental_df = fundamental_df.loc[fundamental_df.index.intersection(weekly_return.columns)]
fundamental_df = fundamental_df.loc[fundamental_df.index.intersection(df.index)]
st.write(len(fundamental_df))

#Streamlit Code
col1, col2, col3 = st.beta_columns(3)
risk = col1.multiselect("Choose Risk Category", list(fundamental_df["Risk Cluster"].unique()))
if not risk:
    col1.error("Please choose an option")
funds = list(fundamental_df[(fundamental_df["Risk Cluster"].isin(risk))]["Fundamental Cluster"].unique())
funds_selected = col2.multiselect("Select Desired Fundamentals",funds) # Select Box of Fundamental Clusters
if not funds_selected:
    col2.error("Please choose an option")
category = list(fundamental_df[(fundamental_df["Risk Cluster"].isin(risk)) & (fundamental_df["Fundamental Cluster"].isin(funds_selected))]["Category"].unique())
category_selected = col3.multiselect("Select Desired Investment Categories",category) # Select Box of Fundamental Clusters
if not category_selected:
    col3.error("Please choose an option")

@st.cache(suppress_st_warning=True)
def make_selected_df(x_risk,x_funds, x_cat):
    selected_tickers = fundamental_df[(fundamental_df["Risk Cluster"].isin(x_risk)) & (fundamental_df["Fundamental Cluster"].isin(x_funds)) & (fundamental_df["Category"].isin(x_cat))  ].index # Tickers Selected

    selected_df = fundamental_df[(fundamental_df["Risk Cluster"].isin(x_risk)) & (fundamental_df["Fundamental Cluster"].isin(x_funds)) & (fundamental_df["Category"].isin(x_cat))  ]
    return selected_tickers, selected_df

selected_tickers, selected_df = make_selected_df(risk,funds_selected,category_selected)



if len(risk) > 0 and len(funds) > 0 and len(category_selected)>0:
    col4, col5 = st.beta_columns([1.5,1])
    col4.header("ETF's in Universe")
    col4.write("Number of ETf's in universe: {}".format(len(selected_df)))
    col4.dataframe(selected_df)
    col5.header("Category Distribution in Universe")
    

    fig = px.pie(selected_df, names = "Category")
    col5.plotly_chart(fig)


######################################################################## Portfolio Divercification ######################################################################## 
st.header("Diversification Calculation")
col1,col2 = st.beta_columns(2)
PDI_DF = pd.DataFrame()
PDI_dict = {}
number_of_assets = [0]
number_of_assets.extend([x for x in range(2,12)])
number_of_assets_selected = col1.selectbox("Please pick number of assets desired to invest in", number_of_assets)
col2.write("The diversification index desscribes how broad a investment is, withnin the selected universe. The large the index number, the more diversified the portfolio is.")

if number_of_assets_selected == 0:
    col1.error("Please choose a number of assets")
else:
    progress_bar = st.progress(0)
    status_text = st.empty()
    PDI_DF,SPY_DF = calculate_pdi(number_of_assets_selected, selected_tickers, weekly_return)
    progress_bar.empty()

    changing_pdi_df = PDI_DF.copy()
    changing_pdi_df["# of Assets"] = changing_pdi_df["# of Assets"]
    min = float(PDI_DF["PDI_INDEX"].min())
    max = float(PDI_DF["PDI_INDEX"].max())

    if min < max and min!= max:
        div_choice = st.slider("Diversification Slider", min_value=min, max_value=max)
        changing_pdi_df = PDI_DF[PDI_DF["PDI_INDEX"].astype(float) >= div_choice]

    col1, col2 = st.beta_columns([2,1])
    col1.subheader("Different Portfolio combinations")
    col1.dataframe(changing_pdi_df)
    st.write(len(changing_pdi_df))
    col2.subheader("Performance of World Index")
    col2.write("Name of ETF: {}".format(SPY_DF["Assets"]))
    col2.write("Sharpe Ratio: {}".format(SPY_DF["Sharpe Ratio"].round(3)))
    col2.write("Annual Mean Return: {}".format(SPY_DF["Annual Return"].round(3)))
    col2.write("Annual Standard Deviation: {}".format(SPY_DF["Annual STD"].round(3)))

    fig = px.scatter(changing_pdi_df, x ="PDI_INDEX" , y = "Sharpe Ratio", hover_data=["Assets",changing_pdi_df.index, "Annual STD"], color = "# of Assets")
    # fig.update_layout(
    #             title="Portfolio Diversificaton",
    #             xaxis_title="Diversification",
    #             yaxis_title="Sharpe Ratio",
    #             legend_title="Volatility")
    fig.add_hline(y=SPY_DF["Sharpe Ratio"], line_color= "orange", annotation_text=SPY_DF["Assets"], line_dash="dot",annotation_position="bottom right")
    st.plotly_chart(fig,use_container_width=True)
    
################################################################################################################################################################



# funds_extend = [list(x) for x in changing_pdi_df["Assets"]]


# dfff = pd.DataFrame()
# for i in funds_extend:
#     #tickers = list(fundamental_df[fundamental_df["Fundamental Cluster"].isin(i)].index)
#     n_assets = len(i)
#     portfolio_weights_ew = np.repeat(1/n_assets, n_assets)
#     port_weekly_return = weekly_return[i].mul(portfolio_weights_ew,axis=1).sum(axis=1)
#     dfff[str(i)] = port_weekly_return
# cumsum = dfff.cumsum(axis=0)
# cumsum["SPY"]= weekly_return["SPY"].cumsum(axis=0)

# fig = px.line(cumsum, x = cumsum.index, y = cumsum.columns)
# fig.update_layout(
#     title="Cluster Perforance",
#     xaxis_title="Time",
#     yaxis_title="Cumulative Performance",
#     legend_title="Clusters",
#     legend = dict(orientation = "v", y=-0.1, x=0 ,xanchor = 'left',
#     yanchor ='top'))
# st.plotly_chart(fig,use_container_width=True)



