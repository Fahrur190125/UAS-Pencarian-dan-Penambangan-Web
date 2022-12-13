import streamlit as st
import pandas as pd
import time

def wordCount(text) :
    from collections import Counter
    count = Counter()
    for i in text :
        for word in i.split() :
            count[word] += 1
    return count

def app():
    st.title('KLASIFIKASI KOMENTAR YOUTUBE')
    st.header('TF-IDF')
    st.markdown('TF-IDF (Term Frequency–Inverse Document Frequency) adalah ukuran statistik yang menggambarkan pentingnya suatu istilah terhadap sebuah dokumen dalam sebuah kumpulan atau korpus. Ukuran ini sering dipakai sebagai faktor pembobot dalam pencarian temu balik informasi, penambangan teks, dan pemodelan pengguna. Nilai tf–idf bertambah sebanding dengan jumlah kemunculan istilah dalam dan bergantung pada jumlah dokumen dalam korpus yang memiliki istilah tersebut. TF-IDF dapat dilakukan setelah dataset dilakukan preprocessing, dan hasil dari TF-IDF dataset komentar youtube sabagai berikut.')
    column_data = pd.read_csv('data/meta/column_data.csv')
    column = column_data['column'][0]
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.preprocessing import normalize
    import numpy as np
    data = pd.read_csv('data/main_data.csv')
    col1, col2 = st.columns(2)
    with col1 :
        st.subheader('Data Text')
        st.markdown('Data text ini didapatkan setelah dataset yang diperoleh dari proses crawling, yang sudah diberikan label/nilai dan telah dilakukan tahapan preprocessing, yang mana data text inilah yang selanjutnya akan diproses untuk dilakukan TF-IDF')
    with col2 :
        teks = np.array(data[column])
        counter = wordCount(teks)
        num_word = len(counter)
        max_features = num_word

        st.subheader('Data TF-IDF')
        st.markdown('Data ini merupakan data yang selanjutnya akan digunakan pada tahapan modelling')
        feature = st.number_input('Berapa banyak fitur atau kata yang ingin anda masukkan',min_value=0,max_value=max_features,value=max_features,key='feature')
        # calc TF vector
        cvect = CountVectorizer(max_features=feature)
        TF_vector = cvect.fit_transform(data[column])

        # # normalize TF vector
        normalized_TF_vector = normalize(TF_vector, norm='l1', axis=1)

        # # calc IDF
        tfidf = TfidfVectorizer(max_features=feature)

        tfs = tfidf.fit_transform(data[column])
        IDF_vector = tfidf.idf_
        # # hitung TF x IDF sehingga dihasilkan TFIDF matrix / vector
        tfidf_mat = normalized_TF_vector.multiply(IDF_vector).toarray()
        df_tfidf = pd.DataFrame(tfidf_mat,columns=tfidf.get_feature_names())
    df1, df2 = st.columns(2)
    with df1 :
        st.markdown('Dataset setelah tahapan preprocessing')
        st.write(data)
    with df2 :
        st.markdown('Dataset hasil dari proses TF-IDF')
        st.write(df_tfidf)        
        pd.DataFrame(df_tfidf).to_csv('data/tf_idf.csv',index=False)
        with st.spinner('tunggu sebentar ...'):
            time.sleep(2)
            st.success('data tf-idf disimpan')

    tokenizer = tfidf.build_tokenizer() 
    st.subheader('Cari kata pada dokumen')
    feature_select = st.selectbox('Pilih fitur atau kata :',options=tfidf.get_feature_names(),key='feature_list')
    doc_list = st.number_input('Pilih dokumen ke berapa (dari index ke-0):',min_value=0,max_value=99,key='doc_list')
    count_token = tokenizer(data['Komentar'][doc_list]).count(feature_select)
    len_doc = len(tokenizer(data['Komentar'][doc_list]))
    st.write("Di dokumen ", doc_list ," " , feature_select ," ada ", count_token ," kata " , "dari ", len_doc, " kata")