import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.cluster import KMeans

def main():
    st.title('Cities Analysis Dashboard: Live Where You want!✈️')

    st.sidebar.title('Choose your options here')

    @st.cache(persist=True, allow_output_mutation=True)
    def get_move_df():
        return pd.read_csv('movehub.csv')
    
    @st.cache(persist=True)
    def get_price_df():
        return pd.read_csv('price.csv')
    
    def get_multiplier(rating, neg):
        if rating == 'Low':
            return 1*neg
        elif rating == 'Medium':
            return 2*neg
        else:
            return 3*neg
    
    movehub_df = get_move_df()
    price_df = get_price_df()
    
    if st.sidebar.checkbox('See all cities', False):
        st.write('### Map of all cities:')
        fig = px.scatter_mapbox(movehub_df, lat='lat', lon='lng', hover_name='City', color='Movehub Rating')
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig)
    
    st.sidebar.subheader('Search criteria')

    my_search = st.sidebar.radio('Locations', ('Worldwide', 'USA'))

    if my_search == 'Worldwide':
        st.sidebar.markdown('Choose continents')

        conts = st.sidebar.multiselect('Continents', ('America', 'Africa', 'Asia', 'Europe', 'Pacific', 'Australia', 'Atlantic', 'Any'))

        st.sidebar.markdown('Rate importance')
        lmh = ('Low', 'Medium', 'High')
        move_hub_ind = get_multiplier(st.sidebar.radio('Move Hub Rating', lmh), 1)
        purchase_ind = get_multiplier(st.sidebar.radio('Purchase Rating', lmh), 1)
        health_ind = get_multiplier(st.sidebar.radio('Health Care Rating', lmh), 1)
        polln_ind = get_multiplier(st.sidebar.radio('Pollution Rating', lmh), -1)
        crime_ind = get_multiplier(st.sidebar.radio('Crime Rating', lmh), -1)
        qol_ind = get_multiplier(st.sidebar.radio('Quality of Life Rating', lmh), 1)

        def get_scores(row):
            return (move_hub_ind*row['Movehub Rating'] +
                    purchase_ind*row['Purchase Power'] +
                    health_ind*row['Health Care'] +
                    polln_ind*row['Pollution'] +
                    crime_ind*row['Crime Rating'])
        
        def filter_cont(row):
            return row['continent'] in conts

        if 'Any' in conts:
            new_mvdf = movehub_df.copy()
            new_mvdf['scores'] = new_mvdf.apply(get_scores, axis=1)
            new_mvdf = new_mvdf.set_index('scores')
            new_mvdf = new_mvdf.sort_index(ascending=False).reset_index()
            st.write('## Your top 5 recommended cities: ')
            new_mvdf = new_mvdf.iloc[:5]
            for i, row in new_mvdf.iterrows():
                st.write(row['City'], int(row['scores']))
            
            if st.button('Show map', False):
                fig = px.scatter_mapbox(new_mvdf, lat='lat', lon='lng', hover_name='City', color='scores')
                fig.update_layout(mapbox_style="open-street-map")
                fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
                st.plotly_chart(fig)
            
            if st.button('Show ratings chart', False):
                fig = px.bar(new_mvdf, x='City', y='scores')
                st.plotly_chart(fig)
        elif len(conts) != 0:
            new_mvdf = movehub_df.copy()
            new_mvdf = new_mvdf[new_mvdf.apply(filter_cont, axis=1)]
            new_mvdf['scores'] = new_mvdf.apply(get_scores, axis=1)
            new_mvdf = new_mvdf.set_index('scores')
            new_mvdf = new_mvdf.sort_index(ascending=False).reset_index()
            st.write('## Your top 5 recommended cities: ')
            new_mvdf = new_mvdf.iloc[:5]
            for i, row in new_mvdf.iterrows():
                st.write(row['City'], int(row['scores']))
            
            if st.button('Show map', key='map1'):
                fig = px.scatter_mapbox(new_mvdf, lat='lat', lon='lng', hover_name='City', color='scores')
                fig.update_layout(mapbox_style="open-street-map")
                fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
                st.plotly_chart(fig)
            
            if st.button('Show ratings chart', False):
                fig = px.bar(new_mvdf, x='City', y='scores')
                st.plotly_chart(fig)
    else:
        merged = movehub_df.merge(price_df, on='City', how='left')
        merged = merged[merged['continent'] == 'America'].dropna()
        merged = merged.groupby('City').agg(np.average)
        merged = merged.reset_index()

        st.sidebar.markdown('Rate importance')
        lmh = ('Low', 'Medium', 'High')
        move_hub_ind = get_multiplier(st.sidebar.radio('Move Hub Rating', lmh), 1)
        purchase_ind = get_multiplier(st.sidebar.radio('Purchase Rating', lmh), 1)
        health_ind = get_multiplier(st.sidebar.radio('Health Care Rating', lmh), 1)
        polln_ind = get_multiplier(st.sidebar.radio('Pollution Rating', lmh), -1)
        crime_ind = get_multiplier(st.sidebar.radio('Crime Rating', lmh), -1)
        qol_ind = get_multiplier(st.sidebar.radio('Quality of Life Rating', lmh), 1)

        price_ind = get_multiplier(st.sidebar.radio('Price of Housing', lmh), -1)

        def get_scores(row):
            return (move_hub_ind*row['Movehub Rating'] +
                    purchase_ind*row['Purchase Power'] +
                    health_ind*row['Health Care'] +
                    polln_ind*row['Pollution'] +
                    crime_ind*row['Crime Rating'] +
                    price_ind*row['January 2017'])
        
        new_mvdf = merged.copy()
        new_mvdf['scores'] = merged.apply(get_scores, axis=1)
        new_mvdf = new_mvdf.set_index('scores')
        new_mvdf = new_mvdf.sort_index(ascending=False).reset_index()

        km = KMeans(n_clusters=8, random_state=42)
        km.fit(new_mvdf.drop(['City', 'City Code'], axis=1))
        new_mvdf['predicted'] = km.labels_

        st.write('## Your top 5 recommended cities: ')
        wr_new_mvdf = new_mvdf.iloc[:5]

        num_classes = []
        for i, row in wr_new_mvdf.iterrows():
            st.write(row['City'], int(row['scores']))
            num_classes.append(row['predicted'])

        st.write('## Other cities you may like: ')

        max_class = np.bincount(num_classes).argmax()
        hello = new_mvdf[new_mvdf['predicted'] == max_class]

        for i, row in hello.iterrows():
            st.write(row['City'])
        
        if st.button('Show map', key='map2'):
            fig = px.scatter_mapbox(hello, lat='lat', lon='lng', hover_name='City', color='scores')
            fig.update_layout(mapbox_style="open-street-map")
            fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig)

if __name__ == '__main__':
    main()