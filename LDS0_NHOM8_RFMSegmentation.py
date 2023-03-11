import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from pyhelpers.store import load_pickle
from streamlit_yellowbrick import st_yellowbrick
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px
import squarify
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from scipy import stats
from pyclustertend import hopkins, ivat
from sklearn.decomposition import PCA
import pickle
import io
from yellowbrick.classifier import ConfusionMatrix, ClassificationReport, ROCAUC, roc_auc

#--------------
#Function
#--------------

#--------------
#LOAD VÀ CHUẨN BỊ DỮ LIỆU
#--------------
#LOAD dữ liệu raw
df = pd.read_fwf('CDNOW_master.txt',header = None, names = ['Customer_id','Dates','Net_quantity','Total_sales'])
df['Dates'] = pd.to_datetime(df['Dates'], format='%Y%m%d')

##Chuẩn bị df df_recency
df_recency = df.groupby(by='Customer_id',
                        as_index=False)['Dates'].max()
df_recency.columns = ['Customer_id', 'LastPurchaseDate']
recent_date = df_recency['LastPurchaseDate'].max()
df_recency['Recency'] = df_recency['LastPurchaseDate'].apply(
    lambda x: (recent_date - x).days)

##Chuẩn bị df frequency_df
frequency_df = df.groupby(by=['Customer_id'], as_index=False)['Dates'].count()
frequency_df.columns = ['Customer_id', 'Frequency']

##Chuẩn bị df frequency_df
monetary_df = df.groupby(by='Customer_id', as_index=False)['Total_sales'].sum()
monetary_df.columns = ['Customer_id', 'Monetary']

##Chuẩn bị dataframe RFM
rf_df = df_recency.merge(frequency_df, on='Customer_id')
rfm_df = rf_df.merge(monetary_df, on='Customer_id').drop(
    columns='LastPurchaseDate')

##Create labels for Recency, Frequency, Monetary
r_labels = range(4, 0, -1)  
f_labels = range(1, 5)
m_labels = range(1, 5)
##Assign these labels to 4 equal percentile groups
r_groups = pd.qcut(rfm_df['Recency'].rank(method='first'), q=4, labels=r_labels)     
f_groups = pd.qcut(rfm_df['Frequency'].rank(method='first'), q=4, labels=f_labels)
m_groups = pd.qcut(rfm_df['Monetary'].rank(method='first'), q=4, labels=m_labels)
# Create new columns R, F, M
RFM = rfm_df.assign(R = r_groups.values, F = f_groups.values, M = m_groups.values)
rfm_df_2 = RFM.copy()

#Concat RFM quartile values to create RFM Segment
def join_rfm(x): return str(int(x['R'])) + str(int(x['F'])) + str(int(x['M']))
rfm_df_2['RFM_Segment'] = rfm_df_2.apply(join_rfm, axis=1)
rfm_df_3 = rfm_df_2.copy()
##Count num of unique segments
rfm_count_unique = rfm_df_2.groupby('RFM_Segment')['RFM_Segment'].nunique()
rfm_count_unique.sum()

#Calculate RFM score and level and Manual Segmentation
rfm_df_3['RFM_Score'] = rfm_df_3[['R','F','M']].sum(axis=1)

def rfm_level(df):
    if (df['RFM_Score'] == 12) :
        return 'STARS'  
    elif (df['R'] == 4 and df['F'] ==1 and df['M'] == 1):
        return 'NEW'
    else:     
        if df['M'] == 4:
            return 'BIG SPENDER'
        elif df['F'] == 4:
            return 'LOYAL'
        elif df['R'] == 4:
            return 'ACTIVE'
        elif df['R'] == 1:
            return 'LOST'
        elif df['M'] == 1:
            return 'LIGHT'
        return 'REGULARS'

rfm_df_3['RFM_level'] = rfm_df_3.apply(rfm_level, axis=1)

## Calculate mean values for each segments
rfm_agg = rfm_df_3.groupby('RFM_level').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': ['mean', 'count']}).round(0)

rfm_agg.columns = rfm_agg.columns.droplevel()
rfm_agg.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
rfm_agg['Percent'] = round((rfm_agg['Count']/rfm_agg.Count.sum())*100, 2)

# Reset the index
rfm_agg = rfm_agg.reset_index()

#Create our plot and resize it.
# fig = plt.gcf()
# ax = fig.add_subplot()
# fig.set_size_inches(8, 10)
# colors_dict = {'ACTIVE':'aliceblue','BIG SPENDER':'royalblue', 'LIGHT':'cyan',
#                'LOST':'red', 'LOYAL':'purple', 'POTENTIAL':'green', 'STARS':'gold'}
# squarify.plot(sizes=rfm_agg['Count'],
#               text_kwargs={'fontsize':14,'weight':'bold', 'fontname':"sans serif"},
#               color=colors_dict.values(),
#               label=['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*rfm_agg.iloc[i])
#                       for i in range(0, len(rfm_agg))], alpha=0.5 )
# plt.title("Customers Segments",fontsize=26,fontweight="bold")
# plt.axis('off')
# plt.savefig('RFM Segments.png')
# plt.show()

#Prepare for new df which remove outliers
new_df = RFM[['Customer_id','Recency','Frequency','Monetary']].copy()

# remove outliers
z_scores = stats.zscore(new_df)
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
new_df = new_df[filtered_entries]

#Scaling data
def preprocess(df):
    """Preprocess data for model clustering"""
    df_log = np.log1p(df)
    scaler = StandardScaler()
    scaler.fit(df_log)
    norm = scaler.transform(df_log)
    return norm

df_scale = preprocess(new_df)

# hopkins(new_df, new_df.shape[0]) 
#feature selection with PCA
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(df_scale)
features = range(pca.n_components_)
PCA_components = pd.DataFrame(principalComponents)

#kmean
ks = range(1,10)
inertias=[]
for k in ks :
    # Create a KMeans clusters
    kmean = KMeans(n_clusters=k,random_state=1)
    kmean.fit(PCA_components.iloc[:,:1])
    inertias.append(kmean.inertia_)

# clustering
kmean1 = KMeans(n_clusters= 3, random_state=42)
kmean1.fit(PCA_components.iloc[:,:1])
cluster_labels = kmean1.labels_
rfm_rfm_k = new_df.assign(K_Cluster = cluster_labels)

km_pca = rfm_rfm_k.groupby('K_Cluster').agg({'Recency': 'mean','Frequency': 'mean',
                                         'Monetary': ['mean', 'count'],}).round(0)

rfm_rfm_k2 = rfm_rfm_k.copy()
rfm_rfm_k2["Segment"]=rfm_rfm_k2['K_Cluster'].map(lambda x:"STAR" if x ==1
                                              else "HIDDEN GEM" if x == 2 
                                              else "ROCK")
#save result and model
rfm_rfm_k.to_csv("result_KMeans.csv")
import dill
with open("KMeans.pkl", 'wb') as file:  
    dill.dump(kmean1, file)

#Calculate average values for each RFM_Level, and return a size of each segment 
rfm_agg2 = rfm_rfm_k2.groupby('Segment').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': ['mean', 'count']}).round(1)
rfm_agg2.columns = rfm_agg2.columns.droplevel()
rfm_agg2.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
rfm_agg2['Percent'] = round((rfm_agg2['Count']/rfm_agg2.Count.sum())*100, 2)
# Reset the index
rfm_agg2 = rfm_agg2.reset_index()
silhouette_score(PCA_components.iloc[:,:1], kmean1.labels_, metric='euclidean')

#Create tree plot
#fig_2 = plt.gcf()
#ax2 = fig_2.add_subplot()
#fig_2.set_size_inches(8, 10)
#colors_dict_2 = {'GOLD':'gold','HIDDEN GEM':'cyan','Promising':'red'}
#squarify.plot(sizes=rfm_agg2['Count'],
#            text_kwargs={'fontsize':12,'weight':'bold', 'fontname':"sans serif"},
#            color=colors_dict_2.values(),
#            label=['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*rfm_agg2.iloc[i])
#                    for i in range(0, len(rfm_agg2))], alpha=0.5 )
#plt.title("Customers Segments",fontsize=26,fontweight="bold")
#plt.axis('off')
#plt.savefig('Kmean_Segments.png')
#plt.show()

#--------------
#LOAD MODEL ĐÃ BUILD 
#--------------
with open('KMeans.pkl', 'rb') as model_file:
    phanloai_khachhang = pickle.load(model_file)

#--------------
#FUNCTION Chạy phần 2
#--------------
def run_all(rfm_df):
    # 1. Show data
    st.subheader('1. Thông tin bộ dữ liệu')
    st.write('5 dòng dữ liệu đầu')
    st.dataframe(rfm_df.head(5))
    st.write('5 dòng dữ liệu cuối')
    st.dataframe(rfm_df.tail(5))
    st.write('Kích thước dữ liệu')
    st.code('Số dòng: '+str(rfm_df.shape[0]) + ' và số cột: '+ str(rfm_df.shape[1]))
    n_null = rfm_df.isnull().any().sum()
    st.code('Số dòng bị NaN: '+ str(n_null))
   
    st.subheader('Thống kê dữ liệu')
    st.dataframe(rfm_df.describe())    

    # 2. Trực quan hóa dữ liệu
    st.subheader('2. EDA')
    fig_plot = plt.figure(figsize=(8,10))
    plt.subplot(3, 1, 1)
    sns.distplot(rfm_df['Recency'])# Plot distribution of R
    plt.subplot(3, 1, 2)
    sns.distplot(rfm_df['Frequency'])# Plot distribution of F
    plt.subplot(3, 1, 3)
    sns.distplot(rfm_df['Monetary']) # Plot distribution of M
    st.pyplot(fig_plot)

    st.write('''
    #### Dựa vào biểu đồ phân phối:
    * Phân phối của Recency bị lệch trái
    * Phân phối của Frequency và Monetary lệch phải
    ''')
    # 3. Sử dụng quartiles để tính RFM 
    st.subheader('3. Sử dụng quartiles để tính RFM')
    st.write('Create labels for Recency, Frequency, Monetary')
    st.code('''r_labels = range(4, 0, -1)
    
f_labels = range(1, 5)
    
m_labels = range(1, 5)  
    ''')
    st.write('Assign these labels to 4 equal percentile groups')
    st.code('''r_groups = pd.qcut(rfm_df['Recency'].rank(method='first'), q=4, labels=r_labels) 
        
f_groups = pd.qcut(rfm_df['Frequency'].rank(method='first'), q=4, labels=f_labels)
        
m_groups = pd.qcut(rfm_df['Monetary'].rank(method='first'), q=4, labels=m_labels)
    ''') 
    st.write('Create new columns R, F, M')
    st.code('''RFM = rfm_df.assign(R = r_groups.values, F = f_groups.values, M = m_groups.values)
    ''')
    st.write('5 dòng dữ liệu đầu RFM sau khi được tạo')
    st.dataframe(RFM.head(5))

    st.write('Concat RFM quartile values to create RFM Segment')
    st.dataframe(rfm_df_2.head())
    st.code('Số lượng unique segment là' + ' ' + str(rfm_count_unique.sum()))
    st.write('')
    # 4. Calculate RFM score and level and Manual Segmentation. 
    st.subheader('4. Tính RFM score, level và Manual Segmentation')
    st.write('Dữ liệu sau khi tính toán')
    st.dataframe(rfm_df_3.head())
    st.write('Mean & tỷ trọng của các segments sau khi làm Manual Segmentation')
    st.dataframe(rfm_agg)
    st.write('Treemap')
    st.image("RFM_Segments_manual.png", caption='Manual Segmentation')
    st.write('Scatter Plot (RFM)')
    pig_dot = px.scatter(rfm_agg, x="RecencyMean", y="MonetaryMean", size="FrequencyMean", color="RFM_level",
           hover_name="RFM_level", size_max=50)
    st.plotly_chart(pig_dot)
    st.write('''
    Nhận xét:
    Nhìn 2 biểu đồ ta có thể thấy kết quả phân cụm của phương pháp vẫn chưa được tốt vì các nhóm còn trùm nhau nhiều.
    Vì vậy ta nên dùng phương pháp khác để phân cụm lại bộ dữ liệu khách hàng sẵn có
    ''')
    # 5. Data Pre-Processing
    st.subheader('5. Data Pre-Processing')
    #Outliers
    st.write('Biểu đồ boxplot')
    fig_box = plt.figure(figsize=(8, 10))
    plt.subplot(3, 1, 1)
    sns.boxplot(x = rfm_df['Recency'])
    plt.subplot(3, 1, 2)
    sns.boxplot(x = rfm_df['Frequency'])
    plt.subplot(3, 1, 3)
    sns.boxplot(x = rfm_df['Monetary'])
    st.pyplot(fig_box)
    st.write('Loại outliers')
    st.code('''# remove outliers
    z_scores = stats.zscore(new_df)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    new_df = new_df[filtered_entries]
    ''')
    st.write('df trước khi loại outliers')
    st.dataframe(RFM[['Recency','Frequency','Monetary']].describe())
    st.write('df sau khi loại outliers')
    st.dataframe(new_df[['Recency','Frequency','Monetary']].describe())
    # plot the distribution of RFM values after remove outliers
    st.write('Biểu đồ distplot sau khi loại bỏ outliers')
    f,ax = plt.subplots(figsize=(10, 12))
    plt.subplot(3, 1, 1)
    sns.distplot(new_df.Recency, label = 'Recency')
    plt.subplot(3, 1, 2)
    sns.distplot(new_df.Frequency, label = 'Frequency')
    plt.subplot(3, 1, 3)
    sns.distplot(new_df.Monetary, label = 'Monetary Value')
    st.pyplot(f)
    st.write('''
    Nhận xét:
    Nhìn biểu đồ ta có thể thấy biểu đồ phân phối không thay đổi mấy sau khi loại bỏ các outliers
    ''')
    st.code('hopkins test' + ' ' + str(hopkins(new_df, new_df.shape[0])) )
    st.write('''
    Nhận xét:
    hopkins test cho số kết quả tiệm cận 0 =>>> có thể phân cụm tốt
    ''')
    st.subheader('Scale data & PCA')
    st.dataframe(df_scale)
    f, ax = plt.subplots(figsize=(8, 5))
    plt.plot(ks, inertias, '-o')
    plt.xlabel('Number of clusters, k')
    plt.ylabel('Inertia')
    plt.xticks(ks)
    plt.style.use('ggplot')
    plt.title('Đâu là k tốt nhất cho thuật toán KMeans ?')
    st.plotly_chart(f)
    st.write('=>> Theo K-Means Elbow Method ta chọn K = 3')
    st.write('')



#---------------------------------------------
# GUI
#---------------------------------------------
st.set_page_config(
    page_title="Capstone Project", page_icon="📈", initial_sidebar_state="expanded"
)

st.title("Trung Tâm Tin Học")
st.write("## Capstone Project - Đồ án tốt nghiệp Data Science")

st.sidebar.header('Capstone Project')
st.sidebar.subheader("Customer Segmetation sử dụng RFM")

menu=['GIỚI THIỆU','TIỀN XỬ LÝ DỮ LIỆU','KẾT QUẢ PHÂN CỤM']
choice=st.sidebar.selectbox('Menu',menu)
st.sidebar.markdown(
        "<hr />",
        unsafe_allow_html=True
    )
if choice == 'GIỚI THIỆU':
    st.markdown("<h1 style='text-align: center; color: white;'>Customer Segmentation</h1>", unsafe_allow_html=True) 
    st.write('''
     ''')
    st.subheader("Customer Segmentation là gì?")
    st.image("RFM_Model_1.png")
    st.image("RFM_Model_2.png")
    st.write('''
        Phân khúc/nhóm/cụm khách hàng (market segmentation – còn được gọi là phân khúc thị trường) là quá trình
    nhóm các khách hàng lại với nhau dựa trên các đặc điểm chung. Nó phân chia và nhóm khách hàng thành các nhóm 
    nhỏ theo đặc điểm địa lý, nhân khẩu học, tâm lý học, hành vi (geographic, demographic, psychographic, behavioral)
    và các đặc điểm khác.''')
    st.write('''
    Các nhà tiếp thị sử dụng kỹ thuật này để nhắm mục tiêu khách hàng thông qua việc cá nhân hóa, khi họ muốn
    tung ra các chiến dịch quảng cáo, truyền thông, thiết kế một ưu đãi hoặc khuyến mãi mới, và cũng để bán hàng.
    ''')
    st.subheader("Customer Segmetation sử dụng RFM?")
    st.image("RFM_hinh1.png")
    st.write('''
    Phân tích RFM (Recency, Frequency, Monetary) là phương pháp tiếp cận dựa trên hành vi của khách
    hàng thành để nhóm thành các phân khúc. RFM phân nhóm khách hàng trên cơ sở các giao dịch mua hàng
    trước đó của họ, nhằm mục đích phục vụ khách hàng tốt hơn.
    ''')
    st.write('''
    RFM giúp người quản lý xác định được khách hàng tiềm năng để kinh doanh có lợi nhuận hơn. Ngoài
    ra, nó giúp các nhà quản lý thực hiện các chiến dịch quảng cáo hiệu quả cho dịch vụ được cá nhân hóa.
    ''')
    st.write('''
    RFM nghiên cứu hành vi của khách hàng và phân nhóm dựa trên ba yếu tố đo lường:
    - Recency (R): đo lường số ngày kể từ lần mua hàng cuối cùng (lần truy cập gần đây nhất) đến ngày giả định chung
    để tính toán (ví dụ: ngày hiện tại, hoặc ngày max trong danh sách giao dịch).
    - Frequency (F): đo lường số lượng giao dịch (tổng số lần mua hàng) được thực hiện trong thời gian nghiên cứu.
    - Monetary Value (M): đo lường số tiền mà mỗi khách hàng đã chi tiêu trong thời gian nghiên cứu.
    ''')

    st.subheader('Business Objective/Problem')
    st.write('''
    Công ty X chủ yếu phát hành và bán sản phẩm là CD tới khách hàng.
    Công ty X mong muốn có thể bán được nhiều sản phẩm hơn cũng như giới
    thiệu sản phẩm đến đúng đối tượng, chăm sóc và làm hài lòng khách hàng.
    Data set CDNOW_master.txt chứa toàn bộ lịch sử mua hàng từ quý đầu tiên 
    năm 1997 cho đến hết quý thứ hai năm 1998 (cuối tháng 06/1998) 
    của 23.570 khách hàng đã thực hiện.
    ''')
elif choice =='TIỀN XỬ LÝ DỮ LIỆU':
    run_all(rfm_df)
    
elif choice=='KẾT QUẢ PHÂN CỤM':
    st.write('Trung bình giá trị RFM và số lượng khách hàng ở mỗi nhóm được phân cụm')
    st.dataframe(km_pca)
    st.code('silhouette score' + ' ' + str(silhouette_score(PCA_components.iloc[:,:1], kmean1.labels_, metric='euclidean')))
    st.write('''
    Nhận xét:
    silhouette score cho giá trị gần bằng 1, có thể thấy kết quả khá tốt.
    ''')
    st.subheader('Trực quan hóa kết quả được phân cụm')
    fig = px.scatter_3d(rfm_rfm_k, x="Recency", y="Monetary", z="Frequency",
                    color = 'K_Cluster', width=800, height=400)
    st.plotly_chart(fig)
    st.write('''
    Dựa vào kết quả ta có thể đặt tên cho các cụm khách hàng như sau:
    
    0. GOLD
    
    1. HIDDEN GEM
    
    2. STAR
    ''')
    st.write('')
    st.write('')
    st.write('tính giá trị trung bình của các RFM_Level và tỷ trọng của các Segments')
    st.dataframe(rfm_agg2)
    st.image('Kmean_Segments.png', caption = 'Unsupervised Segments')
