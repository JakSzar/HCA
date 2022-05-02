import streamlit as st
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage
import plotly.figure_factory as ff


st.sidebar.header('Please read informations below, after that feel free to close this panel')
st.sidebar.write('Methods are for calculating the distance between the newly formed cluster u and each v.')
st.sidebar.write('Methods ‘centroid’, ‘median’, and ‘ward’ are correctly defined only if Euclidean pairwise'
                 ' metric is used, otherwise the produced result will be incorrect.')
st.sidebar.write('Read more: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html')


st.header('Hierarchical Clustering')
st.subheader("File selection")

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    dataset = pd.read_csv(uploaded_file)

    st.dataframe(dataset)
    st.subheader("Select columns for HCA")
    options = st.multiselect(
         'What are the columns of your dataset you want to use?',
         list(dataset.columns.values), default=list(dataset.columns.values))

    st.dataframe(dataset[options])
    st.subheader("Enter arguments for dendrogram or leave default")

    style = str(st.selectbox('Template', ('plotly', 'plotly_white', 'plotly_dark', 'ggplot2', 'seaborn',
                                          'simple_white', 'none')))

    side_1, side_2, side_3 = st.columns(3)

    with side_1:
        linkage_method = st.selectbox(
            'Methods',
            ('ward', 'complete', 'average', 'single', 'weighted', 'centroid', 'median'))

    with side_2:
        metric = st.selectbox(
            'Metric',
            ('euclidean', 'braycurtis', 'canberra', 'chebyshev', 'cityblock', "correlation", 'cosine', 'dice',
             'hamming', 'hamming', 'jensenshannon', 'kulsinski', 'mahalanobis', 'matching',
             'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
             'sqeuclidean', 'yule'))
    with side_3:
        orientation = st.selectbox('Orientation:', ('bottom', 'left', 'top', 'right'))

    side_1, side_2 = st.columns(2)
    with side_1:
        dendrogram_width = st.number_input('width in px', value=1200)
    with side_2:
        dendrogram_height = st.number_input('height in px', value=1200)

    side_1, side_2, side_3 = st.columns(3)
    with side_1:
        title = st.text_input('Title', value='Dendrogram')
        title_size = st.number_input('Title font size', value=25)
        color_threshold = st.number_input('Color threshold. Leave 0 for default')
        x_label_size = st.number_input('X labels font size', value=10)

    with side_2:
        x_label = st.text_input('X title')
        x_title_size = st.number_input('X Title font size', value=20)
        x_label_rotation = st.number_input(' X labels rotation angle', value=0)
        y_label_size = st.number_input('Y labels font size', value=10)
    with side_3:
        y_label = st.text_input('Y title')
        y_title_size = st.number_input('Y title font size', value=20)
        y_label_rotation = st.number_input(' Y label rotation angle', value=0)

    dendrogram_button = st.button('Generate dendrogram')

    data = dataset[options]
    X = data.values
    if dendrogram_button:
        if color_threshold != 0:
            fig = ff.create_dendrogram(X, orientation=orientation,
                                       linkagefun=lambda x: linkage(x, method=linkage_method, metric=metric),
                                       color_threshold=color_threshold)
        else:
            fig = ff.create_dendrogram(X, orientation=orientation,
                                       linkagefun=lambda x: linkage(x, method=linkage_method, metric=metric))
        fig.update_layout(height=dendrogram_height, width=dendrogram_width,
                          title={'text': title, 'xanchor': 'center', 'x': 0.5},
                          template=style, title_font_size=title_size)
        fig.update_xaxes(title_text=x_label, title_font_size=x_title_size,
                         tickangle=x_label_rotation, tickfont=dict(family='default', size=x_label_size))
        fig.update_yaxes(title_text=y_label, title_font_size=y_title_size,
                         tickangle=y_label_rotation, tickfont=dict(family='default', size=y_label_size))
        fig.show()
        # st.plotly_chart(fig, use_container_width=False)

    st.subheader("Clusterization")
    num_of_clusters = int(st.number_input("Number of clusters you wanna use:", min_value=1, step=1))
    metric = st.selectbox('Metric', options=["euclidean", "l1", "l2", "manhattan", "cosine", "precomputed"])
    linkage = st.selectbox('Linkage method', options=['ward', 'complete', 'average', 'single'])
    hc = AgglomerativeClustering(n_clusters=num_of_clusters, affinity=metric, linkage=linkage)
    y_hc = hc.fit_predict(X)
    dataset['Clusters'] = y_hc
    st.dataframe(dataset)


    @st.cache
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')


    csv = convert_df(dataset)
    st.download_button(
       "Press to Download",
       csv,
       "file.csv",
       "text/csv",
       key='download-csv'
    )

    st.subheader('Below you can find csv files separated by clusters')

    for i in range(0, num_of_clusters):
        csv = convert_df(dataset[dataset['Clusters'] == i])
        st.download_button(f'Cluster {i} file',
                           csv,
                           f'cluster_{i}.csv',
                           "text/csv",
                           key='download-csv')

