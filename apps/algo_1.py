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
    @st.cache(show_spinner=False)
    def calculate_pdi_crypto(num_assets, num_crytpos, etf_tickers,crypto_tickers, weekly_returns): 
        
        def meanRetAn(data):             
            Result = 1
            
            for i in data:
                Result *= (1+i)
                
            Result = Result**(1/float(len(data)/52))-1
            
            return(Result)

        pca = PCA()
        PDI_dict = {}
        samples = []
        for i in range(1,20000):
            t = random.sample(etf_tickers,num_assets-num_crytpos)
            crytpo = random.sample(crypto_tickers,num_crytpos)
            t.extend(crytpo)
            samples.append(t)
        
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

        return PDI_DF



############################################################## Trading Strategy #################################################################################
    ################################################################################# PDI Function #################################################################################
    @st.cache(show_spinner=False)
    #Finding the initial portoflio from desired investemnt universe 

    def calculate_pdi(num_assets, tickers, weekly_returns): 
            
            def meanRetAn(data):             
                Result = 1
                
                for i in data:
                    Result *= (1+i)
                    
                Result = Result**(1/float(len(data)/52))-1
                
                return(Result)

            pca = PCA()
            PDI_dict = {}
            samples = []
            for number in [num_assets]:
                for i in range(1,20000):
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

            return PDI_DF



    ############################################################## Trading Strategy #################################################################################
    # Trading algorithm that uses the portfolio chosen, and allocated weights accordingly
    def calculate_pdi_weights( returns,return_mean_range): 

        n = len(returns.columns)
        eq = [1/n]*n
        w = []
        w.append(eq)
        for i in range(1,10000):
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

        for y,num in zip(weights_new, range(0,len(weights_new),1)):
            
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



    ############################################################## Trading Strategy #################################################################################
    # Trading algorithm that finds new portfolios each quarter
    def pca_per_weights_rolling(return_data, portfolio, interval, ret_range_mean,pdi_max_train):
            data = return_data.copy() # data containing weekly returns
            tickers = list(data.columns)
            data.index = pd.to_datetime(data.index) # Conveting the index which is date to datetime
            weeks_list = data[data.index.year > 2015].index # grabbing all index dates
            data.index = data.index.to_period(interval) # converting the index to quarterly sets
            periods = data.index.unique() # taking the unique quarters to loop

            
            list_range = [] # saving rolling periods
            list_period = periods[4:] # periods of return
            for i in range(1,21): 
                list_range.append(periods[i:4+i])

            #print(periods)
            first_period = list_period[0] # the first period of the time frame
            remaining_periods = list_period[1:] # the remianing periods for returns calculations
            pdi_rolling_periods = list_range[:-1] # all periods minus the last

            ########################################  Function for pdi ########## ########## ########## ########## ########### #########  
            def pdi_period(returns, period, weights):
                pca = PCA()
                corr_matrix = np.array(returns.loc[period].mul(weights).cov())
                principalComponents = pca.fit(corr_matrix)
                return 2*sum(principalComponents.explained_variance_ratio_*range(1,len(principalComponents.explained_variance_ratio_)+1,1))-1
            ########## ########## ########## ##########  Mean Annual Return Function ########## ########## ########## ########## ########## 
            def meanRetAn(data):             
                Result = 1
                
                for i in data:
                    Result *= (1+i)
                    
                Result = Result**(1/float(len(data)/ret_range_mean))-1
                
                return(Result)

            ########## ########## ########## ##########  Portfolio Return ########## ########## ########## ########## ########## ########## 
            def port_ret(returns, period, weights): # function for calculating returns
                portfolio_weights_ew = weights
                port_return = returns.loc[period].mul(portfolio_weights_ew,axis=1).sum(axis=1)
                return  port_return

            pdi_performance_w = [] #saving performance of max pdi
            sharpe_performance_w= [] #saving performance of optimal portfolio of max portfolio
            equal_performance = [] #saving equal weight portflio performance 
            sharpe_2_performance_w =[]


            weights_pdi_performance_w = [] #weights for max pdi allocation over time
            weights_sharpe_performance_w= [] #weights for max sharpe ratio allocation over time
            weights_equal_performance = [] # weights for equal portfolio over time "same all periods"
            weights_sharpe_2_performance_w= []
            periods_weights = [] # saving periods for the weights allocation

            pdi_weights_pdi_performance_w = [] #pdi for max pdi allocation over time
            pdi_weights_sharpe_performance_w= [] #pdi for max sharpe ratio allocation over time
            pdi_weights_equal_performance = [] #pdi for equal portfolio over time "same all periods"
            pdi_weights_sharpe_2_performance_w= []



            
            assets = [] # store asstes for all periods
            assets.append(portfolio) # appending portfolio
            ############################################################ Calculate first period ################################################################################
            print(first_period)
            periods_weights.append(first_period)
            first_period_df = calculate_pdi_weights( returns = data[data.index.year == 2015],return_mean_range = 52)

            ################################################## Taking the higest PDI ###########################################################################################

            id = first_period_df["PDI_INDEX"].idxmax()
            port_max_pdi_weights = first_period_df["weights"][id] # getting weights for period
            port_max_pdi_weights_pdi = first_period_df["PDI_INDEX"][id] # getting weights for period
            
            weights_pdi_performance_w.append(port_max_pdi_weights) # saving weights for period
            port_max_ret_period = port_ret(data, first_period,port_max_pdi_weights)  # calculating return for periods 
            pdi_performance_w.extend(port_max_ret_period) # saving return 
            pdi_weights_pdi_performance_w.append(port_max_pdi_weights_pdi) # calculating pdi for first period

            ################################################## Taking the higest Sharpe Ration - PDI ##########################################################################
            id_sharpe = first_period_df["Sharpe Ratio"].idxmax()
            port_max_sharpe_weights_1 = first_period_df["weights"][id_sharpe] # getting weights for period
            port_max_sharpe_weights_pdi = first_period_df["PDI_INDEX"][id_sharpe] # getting weights for period

            weights_sharpe_performance_w.append(port_max_sharpe_weights_1)   # saving weights for periods
            port_max_ret_period_sharpe = port_ret(data, first_period,port_max_sharpe_weights_1)  # calculating return for periods 
            sharpe_performance_w.extend(port_max_ret_period_sharpe) # saving return 
            pdi_weights_sharpe_performance_w.append(port_max_sharpe_weights_pdi) # calculating pdi for first period


            ################################################## Taking the higest Sharpe Ration - PDI ##########################################################################
            if len(first_period_df[first_period_df["PDI_INDEX"] > 2]) == 0:
                mini_df = first_period_df.copy()
            else:
                mini_df = first_period_df[first_period_df["PDI_INDEX"] > 2].copy()

            id_sharpe_2 = mini_df["Sharpe Ratio"].idxmax()
            port_max_sharpe_2_weights = mini_df["weights"][id_sharpe_2] # getting weights for period
            port_max_sharpe_2_weights_pdi = mini_df["PDI_INDEX"][id_sharpe_2] # getting weights for period

            weights_sharpe_2_performance_w.append(port_max_sharpe_2_weights)   # saving weights for periods
            port_max_ret_period_sharpe_2 = port_ret(data, first_period,port_max_sharpe_2_weights)  # calculating return for periods 
            sharpe_2_performance_w.extend(port_max_ret_period_sharpe_2) # saving return 
            pdi_weights_sharpe_2_performance_w.append(port_max_sharpe_2_weights_pdi) # calculating pdi for first period

            ################################################################## Equal Weigths Portoflio ##########################################################################

            equal_weights = first_period_df.iloc[0]["weights"] # getting weights for period
            

            weights_equal_performance.append(equal_weights)  # saving weights for periods
            port_max_ret_period_equal = port_ret(data, first_period,equal_weights) # calculating return for periods 
            equal_performance.extend(port_max_ret_period_equal) # saving return 
            pdi_weights_equal_performance.append(pdi_max_train) # calculating pdi for first period


            ######################################################## Calculation of portfolio perfomnce #############################################################################

            for init_time, next_time ,y in zip(pdi_rolling_periods, remaining_periods,range(1,len(remaining_periods)+1)):

                prog = int(y/len(pdi_rolling_periods)*100)
                progress_bar.progress(prog)
                status_text.text("{}% Complete".format(prog))
                ############ Portfolio Creation ##############################
                print("Rolling range for calculatio: {} - period of return: {}".format(init_time, next_time))
                PDI_DF = calculate_pdi_weights(returns = data.loc[init_time].dropna(axis=1), return_mean_range = ret_range_mean)
                periods_weights.append(next_time) # saving first period
                assets.append(portfolio) # appending portfolio

                ################################################## Taking the higest PDI ##########################################################################

                id = PDI_DF["PDI_INDEX"].idxmax()
                port_max_pdi_weights = PDI_DF["weights"][id] # getting weights for period
                port_max_pdi_weights_pdi = PDI_DF["PDI_INDEX"][id]
                
                weights_pdi_performance_w.append(port_max_pdi_weights) # saving weights for period
                port_max_ret_period = port_ret(data, next_time,port_max_pdi_weights)  # calculating return for periods 
                pdi_performance_w.extend(port_max_ret_period) # saving return 
                pdi_weights_pdi_performance_w.append(port_max_pdi_weights_pdi) # calculating pdi for first period

                ################################################## Taking the higest Sharpe Ration - PDI ##########################################################################
                id_sharpe = PDI_DF["Sharpe Ratio"].idxmax()
                port_max_sharpe_weights = PDI_DF["weights"][id_sharpe] # getting weights for period
                port_max_sharpe_weights_pdi = PDI_DF["PDI_INDEX"][id_sharpe] # getting weights for period

                weights_sharpe_performance_w.append(port_max_sharpe_weights)   # saving weights for periods
                port_max_ret_period_sharpe = port_ret(data, next_time,port_max_sharpe_weights)  # calculating return for periods 
                sharpe_performance_w.extend(port_max_ret_period_sharpe) # saving return 
                pdi_weights_sharpe_performance_w.append(port_max_sharpe_weights_pdi) # calculating pdi for first period

                ################################################## Taking the higest Sharpe Ration - PDI above 2 ##########################################################################
                if len(PDI_DF[PDI_DF["PDI_INDEX"] > 2]) == 0:
                    mini_df = PDI_DF.copy()
                else:
                    mini_df = PDI_DF[PDI_DF["PDI_INDEX"] > 2].copy()

                id_sharpe_2 = mini_df["Sharpe Ratio"].idxmax()
                port_max_sharpe_weights_2 = mini_df["weights"][id_sharpe_2] # getting weights for period
                port_max_sharpe_weights_pdi_2 = mini_df["PDI_INDEX"][id_sharpe_2] # getting weights for period

                weights_sharpe_2_performance_w.append(port_max_sharpe_weights_2)   # saving weights for periods
                port_max_ret_period_sharpe_2 = port_ret(data, next_time,port_max_sharpe_weights_2)  # calculating return for periods 
                sharpe_2_performance_w.extend(port_max_ret_period_sharpe_2) # saving return 
                pdi_weights_sharpe_2_performance_w.append(port_max_sharpe_weights_pdi_2) # calculating pdi for first period

                ################################################################## Equal Weigths Portoflio ##########################################################################

                equal_weights = PDI_DF.iloc[0]["weights"] # getting weights for period
                equal_weights_pdi = PDI_DF.iloc[0]["PDI_INDEX"] # getting weights for period
                

                weights_equal_performance.append(equal_weights)  # saving weights for periods
                port_max_ret_period_equal = port_ret(data, next_time,equal_weights) # calculating return for periods 
                equal_performance.extend(port_max_ret_period_equal) # saving return 
                pdi_weights_equal_performance.append(equal_weights_pdi) # calculating pdi for first period






            performance_frame = pd.DataFrame()
            performance_frame["Time"] = weeks_list
            performance_frame["Equal Weights"] = equal_performance
            performance_frame["Max PDI Weights"] = pdi_performance_w
            performance_frame["Max Sharpe Ratio Weights"] = sharpe_performance_w
            performance_frame["Equal Weights Cumulative"] = performance_frame["Equal Weights"].cumsum(axis=0)
            performance_frame["Max PDI Weights Cumulative"] = performance_frame["Max PDI Weights"].cumsum(axis=0) # cummulative returns max pdi
            performance_frame["Max Sharpe Ratio Weights Cumulative"] = performance_frame["Max Sharpe Ratio Weights"].cumsum(axis=0) #cummulative return sharpe ratio

            weights_frame = pd.DataFrame()
            weights_frame["Period"] = periods_weights
            weights_frame["Weights Max PDI"] = weights_pdi_performance_w
            weights_frame["Weights Max sharpe"] = weights_sharpe_performance_w
            weights_frame["Weights Equal"] = weights_equal_performance
            weights_frame["Weights Max PDI - PDI"] = pdi_weights_pdi_performance_w
            weights_frame["Weights Max sharpe - PDI"] = pdi_weights_sharpe_performance_w
            weights_frame["Weights Equal - PDI"] = pdi_weights_equal_performance
            weights_frame["Assets"] = assets







            return performance_frame, weights_frame


#####################################################################################################################################################################
    #Title 
    st.title("Investment Universe Definition")
    #Load DataFrames
    cluster_df = pd.read_csv("all_clusters.csv").rename(columns={"Unnamed: 0": "symbol","kmean_env_cluster": "Enviromental Cluster Impact","Kmean_Social_cluster":"Social Impact Cluster","Kmean_Gov_cluster": "Gouvernace Impact Cluster"}).set_index("symbol")
    week_return_df = pd.read_csv("weeklyReturns.csv", index_col="Date")
    # Add Crypto 
    returns_crypto = pd.read_csv("crypto_weekly.csv" , index_col="Date")
    tick_crypto = list(returns_crypto.columns)
    returns_crypto.index = pd.to_datetime(returns_crypto.index)
    #returns_merge = pd.merge(returns_crypto,returns_weekly,left_index=True,right_index=True)
    
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

    returns_weekly = pd.read_csv("weeklyReturns.csv", index_col="Date") # loading returns dataframe
    returns_weekly.index = pd.to_datetime(returns_weekly.index) # converting returns dataframe to datetime
    returns_weekly_1 = returns_weekly[sel_tick]
    #Defining training data
    train_return = returns_weekly_1[returns_weekly_1.index.year <= 2015] # training on data from 2015
    urth_2015 = returns_weekly[returns_weekly.index.year <= 2015]
    urth_2016 = returns_weekly[returns_weekly.index.year <= 2015]

    col4, col5 = st.beta_columns([1.5,1])
    col4.header("ETF's in Universe")
    col4.write("Number of ETf's in universe: {}".format(len(sel_df)))
    col4.dataframe(sel_df)
    col5.header("Category Distribution in Universe")
    

    fig = px.pie(sel_df, names = "Category")
    col5.plotly_chart(fig)
    ####################################################################### World Df ######################################################################## 
    
    
    def meanRetAn(data):             
        Result = 1
        
        for i in data.index:
            Result *= (1+data.loc[i])
            
        Result = Result**(1/float(len(data.index)/52))-1
        
        return(Result)


    world_ret = meanRetAn(urth_2015["URTH"])
    world_std = urth_2015["URTH"].std(axis=0)*np.sqrt(52)
    world_sharpe = world_ret/ world_std


    world_cum = pd.DataFrame()
    world_cum["MSCI World Cumulative"] = urth_2016["URTH"].cumsum(axis=0)



    ####################################################################### Portfolio Divercification ######################################################################## 
    st.header("Portfolio Generation")
    
    add_crypto = st.checkbox("Add Cryptocurrencies")
    
    no_crypt = st.checkbox("No Cryptocurrencies")
    col1,col2 = st.beta_columns(2)
    PDI_DF = pd.DataFrame()
    PDI_dict = {}



    if add_crypto:
        returns_crypto = pd.read_csv("crypto_weekly.csv" , index_col="Date")
        tick_crypto = list(returns_crypto.columns)
        returns_crypto.index = pd.to_datetime(returns_crypto.index)
        number_of_assets_selected = col1.selectbox("Pick number of assets desired to invest in",[0,1,2,3,4,5,6,7,8,9,10] )
        number_of_cryptos_selected = col2.selectbox("Pick number of Cryptocurrencies",[0,1,2,3,4,5,6,7,8,9] )
        #col2.write("The diversification index desscribes how broad a investment is, withnin the selected universe. The large the index number, the more diversified the portfolio is.")
        returns_merge = pd.merge(returns_crypto,returns_weekly,left_index=True,right_index=True)
        train_return = returns_merge[returns_merge.index.year <= 2015] # training on data from 2015
        if number_of_assets_selected == 0:
            col1.error("Please choose a number of assets")
        if number_of_cryptos_selected == 0:
            col1.error("Please choose a number of assets")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            PDI_DF = calculate_pdi_crypto(num_assets=int(number_of_assets_selected), num_crytpos=int(number_of_cryptos_selected), etf_tickers=list(sel_tick),crypto_tickers=tick_crypto, weekly_returns=train_return)
            progress_bar.empty()

            changing_pdi_df = PDI_DF.copy()
            changing_pdi_df["# of Assets"] = changing_pdi_df["# of Assets"]
            min = float(PDI_DF["PDI_INDEX"].min())
            max = float(PDI_DF["PDI_INDEX"].max())

            if min < max and min!= max:
                div_choice = st.slider("Diversification Slider", min_value=min, max_value=max)
                changing_pdi_df = PDI_DF[PDI_DF["PDI_INDEX"].astype(float) >= div_choice]

            
            st.subheader("Different Portfolio combinations")
            st.dataframe(changing_pdi_df)
            st.write(len(changing_pdi_df))
         

            best_pdi = PDI_DF["PDI_INDEX"].idxmax()
            pdi_asset = ast.literal_eval(PDI_DF["Assets"][best_pdi])
            best_sharpe = PDI_DF["Sharpe Ratio"].idxmax()
            sharpe_asset = ast.literal_eval(PDI_DF["Assets"][best_sharpe])
            col1,col2,col3 = st.beta_columns(3)
            col1.subheader("Most Diversified Portfolio Performance")
            col1.write("Portfolio combination: {}".format(PDI_DF["Assets"][best_pdi]))
            col1.write("Diversification of diviserified portfolio: {}".format(PDI_DF["PDI_INDEX"][best_pdi].round(3)))
            col1.write("Sharpe Ratio: {}".format(PDI_DF["Sharpe Ratio"][best_pdi].round(3)))
            col1.write("Annual Average Return: {}".format(PDI_DF["Annual Return"][best_pdi].round(3)))
            col1.write("Standard Deviation of Return: {}".format(PDI_DF["Annual STD"][best_pdi].round(3)))
            #col1.write("Categories in portfolio: {}".format(list(cluster_df.loc[pdi_asset]["Category"])))

            col2.subheader("Highest Sharpe Ratio Portfolio Performance")
            col2.write("Portfolio combination: {}".format(PDI_DF["Assets"][best_sharpe]))
            col2.write("Diversification of diviserified portfolio: {}".format(PDI_DF["PDI_INDEX"][best_sharpe].round(3)))
            col2.write("Sharpe Ratio: {}".format(PDI_DF["Sharpe Ratio"][best_sharpe].round(3)))
            col2.write("Annual Average Return: {}".format(PDI_DF["Annual Return"][best_sharpe].round(3)))
            col2.write("Standard Deviation of Return: {}".format(PDI_DF["Annual STD"][best_sharpe].round(3)))
            #col2.write("Categories in portfolio: {}".format(list(cluster_df.loc[sharpe_asset]["Category"])))

            col3.subheader("Performance of World Index")
            col3.write("Name of ETF: {}".format("URTH"))
            col3.write("Sharpe Ratio: {}".format(world_sharpe.round(3)))
            col3.write("Annual Mean Return: {}".format(world_ret.round(3)))
            col3.write("Annual Standard Deviation: {}".format(world_std.round(3)))

            fig_1 = px.scatter(changing_pdi_df, x ="PDI_INDEX" , y = "Sharpe Ratio", hover_data=["Assets",changing_pdi_df.index, "Annual STD","Annual Return"], color = "Annual STD")
            fig_1.update_layout(
                            #title="Portfolio Diversificaton",
                            xaxis_title="Diversification",
                            yaxis_title="Risk Adjusted Return",
                            legend_title="Volatility")
            fig_1.add_hline(y= world_sharpe, line_color= "orange", annotation_text= "URTH", line_dash="dot",annotation_position="bottom right")
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
            if a:
                dingo = PDI_DF.iloc[int(a)-1].T
                col2.dataframe(dingo)
                port_pick = ast.literal_eval(PDI_DF.iloc[int(a)-1]["Assets"])
                return_df = returns_merge[port_pick]
                id_index_pdi = PDI_DF.iloc[int(a)-1]["PDI_INDEX"]



                run = st.checkbox("Run Optimisation Backtest")


            ########################################################### Strategy Trade #####################################################################################
                if run:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    performance_w, weight_w = pca_per_weights_rolling(return_data = return_df, portfolio = port_pick , interval = "Q", ret_range_mean = 12,pdi_max_train=id_index_pdi)
                    progress_bar.empty()
                    performance_w["MSCI World"] = world_cum["MSCI World Cumulative"]
                    fig_performance = px.line(performance_w, x="Time", y=["Max PDI Weights Cummulative","Max Sharpe Ratio Weights Cummulative","Equal Weights Cummulative"])
                    fig_performance.update_layout(
                            title="Performance of Strategy",
                            xaxis_title="Time",
                            yaxis_title="Cummulative Performance",
                            legend_title="Strategies")
                    st.plotly_chart(fig_performance,use_container_width=True)
        
    
    if no_crypt:
        number_of_assets_selected = st.selectbox("Pick number of assets desired to invest in",[0,3,4,5,6,7,8,9,10] )
        #col2.write("The diversification index desscribes how broad a investment is, withnin the selected universe. The large the index number, the more diversified the portfolio is.")

        if number_of_assets_selected == 0:
            col1.error("Please choose a number of assets")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            PDI_DF = calculate_pdi(number_of_assets_selected, sel_tick, train_return)
            progress_bar.empty()

            changing_pdi_df = PDI_DF.copy()
            changing_pdi_df["# of Assets"] = changing_pdi_df["# of Assets"]
            min = float(PDI_DF["PDI_INDEX"].min())
            max = float(PDI_DF["PDI_INDEX"].max())

            if min < max and min!= max:
                div_choice = st.slider("Diversification Slider", min_value=min, max_value=max)
                changing_pdi_df = PDI_DF[PDI_DF["PDI_INDEX"].astype(float) >= div_choice]


            st.subheader("Different Portfolio combinations")
            st.dataframe(changing_pdi_df)
            st.write(len(changing_pdi_df))
  

            best_pdi = PDI_DF["PDI_INDEX"].idxmax()
            pdi_asset = ast.literal_eval(PDI_DF["Assets"][best_pdi])
            best_sharpe = PDI_DF["Sharpe Ratio"].idxmax()
            sharpe_asset = ast.literal_eval(PDI_DF["Assets"][best_sharpe])
            col1,col2 ,col3= st.beta_columns(3)
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

            col3.subheader("Performance of World Index")
            col3.write("Name of ETF: {}".format("URTH"))
            col3.write("Sharpe Ratio: {}".format(world_sharpe.round(3)))
            col3.write("Annual Mean Return: {}".format(world_ret.round(3)))
            col3.write("Annual Standard Deviation: {}".format(world_std.round(3)))

            fig_1 = px.scatter(changing_pdi_df, x ="PDI_INDEX" , y = "Sharpe Ratio", hover_data=["Assets",changing_pdi_df.index, "Annual STD","Annual Return"], color = "Annual STD")
            fig.update_layout(
                            #title="Portfolio Diversificaton",
                            xaxis_title="Diversification",
                            yaxis_title="Risk Adjusted Return",
                            legend_title="Volatility")
            fig_1.add_hline(y= world_sharpe, line_color= "orange", annotation_text= "URTH", line_dash="dot",annotation_position="bottom right")
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
            if a:
                dingo = PDI_DF.iloc[int(a)-1].T
                col2.dataframe(dingo)
                port_pick = ast.literal_eval(PDI_DF.iloc[int(a)-1]["Assets"])
                return_df = week_return_df[port_pick]
                id_index_pdi = PDI_DF.iloc[int(a)-1]["PDI_INDEX"]



                run = st.checkbox("Run Optimisation Backtest")


            ########################################################### Strategy Trade #####################################################################################
                if run:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    performance_w, weight_w = pca_per_weights_rolling(return_data = return_df, portfolio = port_pick , interval = "Q", ret_range_mean = 12,pdi_max_train=id_index_pdi)
                    progress_bar.empty()
                    performance_w["MSCI World"] = world_cum["MSCI World Cumulative"]
                    fig_performance = px.line(performance_w, x="Time", y=["Max PDI Weights Cumulative","Max Sharpe Ratio Weights Cumulative","Equal Weights Cumulative"])
                    fig_performance.update_layout(
                            title="Performance of Strategy",
                            xaxis_title="Time",
                            yaxis_title="Cummulative Performance",
                            legend_title="Strategies")
                    st.plotly_chart(fig_performance,use_container_width=True)

