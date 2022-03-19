import pandas as pd
import numpy as np
import seaborn as sns
import seaborn as sns
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
import warnings

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.0f' % x)

df_2010_2011 = pd.read_excel(r"C:\Users\hp\PycharmProjects\pythonProject2\online_retail_II.xlsx", sheet_name = "Year 2010-2011")
df = df_2010_2011.copy()
df.head()



def create_rfm(dataframe):
    #Preparing Data
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]

    #Calculating RFM Metrics
    today_date = dt.datetime(2011, 12, 11)
    rfm = dataframe.groupby('Customer ID').agg({'InvoiceDate': lambda date: (today_date - date.max()).days,
                                                'Invoice': lambda num: num.nunique(),
                                                "TotalPrice": lambda price: price.sum()})
    rfm.columns = ['recency', 'frequency', "monetary"]
    rfm = rfm[(rfm['monetary'] > 0)]

    #Calculating RFM Scores
    rfm["recency_score"]=pd.qcut(rfm["recency"],5,labels=[5,4,3,2,1])
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1,2,3,4,5])
    #NOTE:Monetary segment is not processed because it is not used in definition.

    #Naming segments
    rfm["rfm_segment"]=rfm["recency_score"].astype(str)+rfm["frequency_score"].astype(str)
    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }
    rfm["rfm_segment"]=rfm["rfm_segment"].replace(seg_map,regex=True)
    rfm=rfm[["recency","frequency","monetary","rfm_segment"]]
    return rfm



#We give our data to the function that we created above.The function extracts the classes.
rfm=create_rfm(df)
rfm

rfm = rfm.loc[:,"recency":"monetary"]
rfm.head()

# K-Means
sc = MinMaxScaler((0,1))
df = sc.fit_transform(rfm)
kmeans = KMeans(n_clusters = 10)
k_fit = kmeans.fit(df)
k_fit.labels_

# The optimum number of clusters will be determined by iterative operations
kmeans = KMeans()
k_fit = kmeans.fit(df)
ssd = []

K = range(1,30)

for k in K:
    kmeans = KMeans(n_clusters = k).fit(df)
    ssd.append(kmeans.inertia_)

plt.plot(K, ssd, "bx-")
plt.xlabel("Distance Residuals according to chosen k")
plt.title("Elbow method for optimum k ")
plt.show()

#second way
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df)
elbow.show()

#Final Kmeans model with optimum number of clusters.
kmeans = KMeans(n_clusters = 6).fit(df)
kumeler = kmeans.labels_

pd.DataFrame({"Customer ID": rfm.index, "Kumeler": kumeler})


#Final rfm dataframe after segmentation with K-means algorithm.
rfm["cluster_no"] = kumeler
rfm["cluster_no"] = rfm["cluster_no"] + 1
rfm.head()