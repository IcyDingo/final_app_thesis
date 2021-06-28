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


    
    st.title("1/N Strategy")
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
        samples = [["URTH"]]
        for number in [num_assets]:
            for i in range(1,30000):
                #samples.extend([list(x) for x in combinations(selected_tickers, number_of_assets)])
                samples.append(random.sample(list(tickers),number))
        seen = set()
        samples_mini = [x for x in samples if frozenset(x) not in seen and not seen.add(frozenset(x))]


        
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
    ################################################################################# PDI Function - weights#################################################################################
    @st.cache(show_spinner=False)
    def calculate_pdi_weights(returns,return_mean_range): 

        n = len(returns.columns)
        w = [[(100/n)/100]*n]
        for i in range(1,30000):
            weights = [random.random() for _ in range(n)]
            sum_weights = sum(weights)
            weights = [1*w/sum_weights for w in weights]
            w.append(list(np.round(weights,2)))
        seen = set()
        samples_weigth = [x for x in w if frozenset(x) not in seen and not seen.add(frozenset(x))]


        def meanRetAn(data):             
            Result = 1
            
            for i in data:
                Result *= (1+i)
                
            Result = Result**(1/float(len(data)/return_mean_range))-1
            
            return(Result)

        pca = PCA()
        PDI_dict = {}

        for y,num in zip(samples_weigth, range(0,len(samples_weigth),1)):
            
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

    ################################################################################# Trading Strategy - weights #################################################################################
    @st.cache(show_spinner=False)
    def pca_per_weights(return_data, portfolio, interval, ret_range_mean):
            data = return_data.copy() # data containing weekly returns
            tickers = list(data.columns)
            data.index = pd.to_datetime(data.index) # Conveting the index which is date to datetime
            weeks_list = data.index # grabbing all index dates
            data.index = data.index.to_period(interval) # converting the index to quarterly sets
            periods = data.index.unique() # taking the unique quarters to loop

            
            #print(periods)
            first_period = periods[0] # the first period of the time frame
            remaining_periods = periods[1:] # the remianing periods of the time framr
            first_periods = periods[:-1] # all periods minus the last

            def meanRetAn(data):             
                Result = 1
                
                for i in data:
                    Result *= (1+i)
                    
                Result = Result**(1/float(len(data)/ret_range_mean))-1
                
                return(Result)

            
            def port_ret(returns, period, weights): # function for calculating returns
                portfolio_weights_ew = weights
                port_return = returns.loc[period].mul(portfolio_weights_ew,axis=1).sum(axis=1)
                return  port_return

            pdi_performance_w = []
            sharpe_performance_w= []
            equal_performance = []

            
            ############################################################ Calculate first period ####################################################################################
            #### Weighted #####
            print(first_period)
            n_assets = len(portfolio) # equal weigt initialisation
            ini_weights = np.repeat(1/n_assets, n_assets) # equal weigt initialisation

            pdi_performance_w.extend(port_ret(data,first_period,ini_weights))

            sharpe_performance_w.extend(port_ret(data,first_period,ini_weights))

            equal_performance.extend(port_ret(data,first_period,ini_weights))

            tickers_weekly = list(data.columns)
            #number_of_assets = [len(initial_port)]







            ######################################################## Calculation of portfolio perfomnce #############################################################################

            for init_time, next_time, per in zip(first_periods,remaining_periods, range(1,len(periods)+1)):
                prog = int(per/len(periods)*100)
                progress_bar.progress(prog)
                status_text.text("{}% Complete".format(prog))
                ############ Portfolio Creation ##############################
                print("first time: {} - last time: {}".format(init_time, next_time))
                PDI_DF = calculate_pdi_weights(returns = data.loc[init_time].dropna(axis=1), return_mean_range = ret_range_mean)


                ################################################## Taking the higest PDI ##########################################################################

                id = PDI_DF["PDI_INDEX"].idxmax()
                port_max_pdi_weights = PDI_DF["weights"][id]
                
                port_max_ret_period = port_ret(data, next_time,port_max_pdi_weights)
                pdi_performance_w.extend(port_max_ret_period)

                ################################################## Taking the higest Sharpe Ration - PDI ##########################################################################
                id_sharpe = PDI_DF["Sharpe Ratio"].idxmax()
                port_max_sharpe_weights = PDI_DF["weights"][id_sharpe]

                
                port_max_ret_period_sharpe = port_ret(data, next_time,port_max_sharpe_weights)
                sharpe_performance_w.extend(port_max_ret_period_sharpe)
                
                ################################################################## Equal Weigths Portoflio ##########################################################################

                equal_weights = PDI_DF.iloc[0]["weights"]

                
                port_max_ret_period_equal = port_ret(data, next_time,equal_weights)
                equal_performance.extend(port_max_ret_period_equal)




            dd = pd.DataFrame()
            dd["Time"] = weeks_list
            dd["Equal Weights"] = equal_performance
            dd["Max PDI Weights"] = pdi_performance_w
            dd["Max Sharpe Ratio Weights"] = sharpe_performance_w
            dd["Equal Weights Cummulative"] = dd["Equal Weights"].cumsum(axis=0)
            dd["Max PDI Weights Cummulative"] = dd["Max PDI Weights"].cumsum(axis=0) # cummulative returns max pdi
            dd["Max Sharpe Ratio Weights Cummulative"] = dd["Max Sharpe Ratio Weights"].cumsum(axis=0) #cummulative return sharpe ratio




            return dd
    ################################################################################################################################################################################


    #Title 
    st.title("Investment Universe Definition")
    #Load DataFrames
    cluster_df = pd.read_csv("all_clusters.csv").rename(columns={"Unnamed: 0": "symbol","kmean_env_cluster": "Enviromental Cluster Impact","Kmean_Social_cluster":"Social Impact Cluster","Kmean_Gov_cluster": "Gouvernace Impact Cluster"}).set_index("symbol")
    week_return_df = pd.read_csv("weeklyReturns.csv", index_col="Date")

    #Streamlit Code
    col1, col2, col3, col4 = st.beta_columns(4)
    category = col1.multiselect("Choose Category", list(cluster_df["Category"].unique()))
    social = col2.multiselect("Choose Social Investment Desire",list(cluster_df[cluster_df["Category"].isin(category)]["Social Impact Cluster"].unique()))
    enviro = col3.multiselect("Choose Enviromental Desire", list(cluster_df[cluster_df["Category"].isin(category) & (cluster_df["Social Impact Cluster"].isin(social))]["Enviromental Cluster Impact"].unique()))
    gov = col4.multiselect("Choose Governance Desire", list(cluster_df[cluster_df["Category"].isin(category) & (cluster_df["Social Impact Cluster"].isin(social)) &(cluster_df["Enviromental Cluster Impact"].isin(enviro))]["Gouvernace Impact Cluster"].unique()))

    @st.cache(suppress_st_warning=True)
    def make_selected_df(category,social,enviro,gov):

        selected_tickers = cluster_df[(cluster_df["Category"].isin(category))&(cluster_df["Social Impact Cluster"].isin(social)) & (cluster_df["Enviromental Cluster Impact"].isin(enviro)) & (cluster_df["Gouvernace Impact Cluster"].isin(gov))  ].index # Tickers Selected
        selected_df = cluster_df[(cluster_df["Category"].isin(category))&(cluster_df["Social Impact Cluster"].isin(social)) & (cluster_df["Enviromental Cluster Impact"].isin(enviro)) & (cluster_df["Gouvernace Impact Cluster"].isin(gov))  ] # Tickers df
        return selected_tickers, selected_df

    
    sel_tick, sel_df = make_selected_df(category,social,enviro,gov)



    col4, col5 = st.beta_columns([1.5,1])
    col4.header("ETF's in Universe")
    col4.write("Number of ETf's in universe: {}".format(len(sel_df)))
    col4.dataframe(sel_df)
    col5.header("Category Distribution in Universe")
    

    fig = px.pie(sel_df, names = "Category")
    col5.plotly_chart(fig)

    ####################################################################### Portfolio Divercification ######################################################################## 
    st.header("Portfolio Generation")
    col1,col2 = st.beta_columns(2)
    PDI_DF = pd.DataFrame()
    PDI_dict = {}

    number_of_assets_selected = col1.selectbox("Pick number of assets desired to invest in",[0,3,4,5,6,7,8,9,10] )
    col2.write("The diversification index desscribes how broad a investment is, withnin the selected universe. The large the index number, the more diversified the portfolio is.")

    if number_of_assets_selected == 0:
        col1.error("Please choose a number of assets")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()
        PDI_DF,world_DF = calculate_pdi(number_of_assets_selected, sel_tick, week_return_df)
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
        col2.write("Name of ETF: {}".format(world_DF["Assets"]))
        col2.write("Sharpe Ratio: {}".format(world_DF["Sharpe Ratio"].round(3)))
        col2.write("Annual Mean Return: {}".format(world_DF["Annual Return"].round(3)))
        col2.write("Annual Standard Deviation: {}".format(world_DF["Annual STD"].round(3)))

        best_pdi = PDI_DF["PDI_INDEX"].idxmax()
        pdi_asset = ast.literal_eval(PDI_DF["Assets"][best_pdi])
        best_sharpe = PDI_DF["Sharpe Ratio"].idxmax()
        sharpe_asset = ast.literal_eval(PDI_DF["Assets"][best_sharpe])
        col1,col2 = st.beta_columns(2)
        col1.subheader("Most Diversified Portfolio Performance")
        col1.write("Portfolio combination: {}".format(PDI_DF["Assets"][best_pdi]))
        col1.write("Diversification of diviserified portfolio: {}".format(PDI_DF["PDI_INDEX"][best_pdi].round(3)))
        col1.write("Sharpe Ratio: {}".format(PDI_DF["Sharpe Ratio"][best_pdi].round(3)))
        col1.write("Annual Average Return: {}".format(PDI_DF["Annual Return"][best_pdi].round(3)))
        col1.write("Standard Deviation of Return: {}".format(PDI_DF["Annual STD"][best_pdi].round(3)))
        col1.write("Categories in portfolio: {}".format(list(cluster_df.loc[pdi_asset]["Category"])))

        col2.subheader("Highest Sharpe Ratio Portfolio Performance")
        col2.write("Portfolio combination: {}".format(PDI_DF["Assets"][best_sharpe]))
        col2.write("Diversification of diviserified portfolio: {}".format(PDI_DF["PDI_INDEX"][best_sharpe].round(3)))
        col2.write("Sharpe Ratio: {}".format(PDI_DF["Sharpe Ratio"][best_sharpe].round(3)))
        col2.write("Annual Average Return: {}".format(PDI_DF["Annual Return"][best_sharpe].round(3)))
        col2.write("Standard Deviation of Return: {}".format(PDI_DF["Annual STD"][best_sharpe].round(3)))
        col2.write("Categories in portfolio: {}".format(list(cluster_df.loc[sharpe_asset]["Category"])))

        fig_1 = px.scatter(changing_pdi_df, x ="PDI_INDEX" , y = "Sharpe Ratio", hover_data=["Assets",changing_pdi_df.index, "Annual STD","Annual Return"], color = "Annual STD")
        # fig.update_layout(
        #             title="Portfolio Diversificaton",
        #             xaxis_title="Diversification",
        #             yaxis_title="Sharpe Ratio",
        #             legend_title="Volatility")
        fig_1.add_hline(y= world_DF["Sharpe Ratio"], line_color= "orange", annotation_text= world_DF["Assets"], line_dash="dot",annotation_position="bottom right")
        st.plotly_chart(fig_1,use_container_width=True)
        # fig_2 = px.scatter(changing_pdi_df, y ="Annual STD" , x = "PDI_INDEX", hover_data=["Assets",changing_pdi_df.index,"Sharpe Ratio","PDI_INDEX"], color = "Sharpe Ratio")
        # # fig.update_layout(
        # #             title="Portfolio Diversificaton",
        # #             xaxis_title="Diversification",
        # #             yaxis_title="Sharpe Ratio",
        # #             legend_title="Volatility")
        # fig_2.add_hline(y= world_DF["Annual STD"], line_color= "orange", annotation_text= "MSCI World URTH Return", line_dash="dot",annotation_position="bottom right")
        # #fig_2.add_vline(x= world_DF["Annual STD"], line_color= "orange", annotation_text= "MSCI World URTH STD", line_dash="dot",annotation_position="bottom right")
        # st.plotly_chart(fig_2,use_container_width=True)
        
    ################################################################################################################################################################

        col1, col2 = st.beta_columns(2)
        col1.write("Pick Desired Portfolio")
        a =  col1.text_input("Index Number")
        dingo = PDI_DF.iloc[int(a)-1].T
        col2.dataframe(dingo)
        port_pick = ast.literal_eval(PDI_DF.iloc[int(a)-1]["Assets"])
        return_df = week_return_df[port_pick]

        run = st.button("Run Strategy")

    ########################################################### Strategy Trade #####################################################################################
        if run:
            progress_bar = st.progress(0)
            status_text = st.empty()
            performance_w = pca_per_weights(return_data = return_df, portfolio = port_pick , interval = "Q", ret_range_mean = 12)
            progress_bar.empty()

            fig_performance = px.line(performance_w, x="Time", y=["Max PDI Weights Cummulative","Max Sharpe Ratio Weights Cummulative","Equal Weights Cummulative"])
            fig_performance.update_layout(
                    title="Performance of Strategy",
                    xaxis_title="Time",
                    yaxis_title="Cummulative Performance",
                    legend_title="Strategies")
            st.plotly_chart(fig_performance,use_container_width=True)


