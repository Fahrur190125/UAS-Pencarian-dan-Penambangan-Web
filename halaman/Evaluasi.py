import streamlit as st
import pandas as pd
def app():
    st.title('KLASIFIKASI KOMENTAR YOUTUBE')
    st.header('Evaluasi')
    st.markdown('Setelah melakukan eksperimen/percobaan dengan beberapa metode, maka akan ditemukan evaluasi terhadap modelling yang paling terbaik untuk data komentar youtube yang didapat dari link https://youtu.be/aeF8c348Urc. Percobaan/Eksperimen yang dilakukan untuk mendapatkan metode terbaik untuk data komentar tersebut untuk mendapatkan nilai akurasi yang terbaik dengan melakukan eksperimen memainkan jumlah data testing dan data training yang dipakai. Eksperimen dilakukan dengan menggunakan 5 eksperimen yaitu mulai dari 10% data testing dan 90% data training, hingga 50% data testing dan 50% data training,. Hasil dari setiap eksperimen tersebut sebagai berikut.')

    st.subheader('Eksperimen/Percobaan dengan 10% data testing dan 90% data training')
    with st.expander("Tampilkan"):
        col1, col2 = st.columns(2)
        with col1:
            eks = pd.read_excel('data/evaluasi/10.xls')
            st.write(eks)
        with col2:
            st.markdown('Dari tabel dapat dilihat bahwa eksperimen dengan menggunakan 10% data testing dan 90% data training menyatakan bahwa metode stacking yang memiliki nilai akurasi yang paling tinggi dengan perolehan secara keseluruhan dari 10x percobaan menghasilkan 81% disusul dengan naive bayes 80%.')

    st.subheader('Eksperimen/Percobaan dengan 20% data testing dan 80% data training')
    with st.expander("Tampilkan"):
        col1, col2 = st.columns(2)
        with col1:
            eks = pd.read_excel('data/evaluasi/20.xls')
            st.write(eks)
        with col2:
            st.markdown('Dari tabel dapat dilihat bahwa eksperimen dengan menggunakan 20% data testing dan 80% data training menyatakan bahwa metode naive bayes multinomial yang memiliki nilai akurasi yang paling tinggi dengan perolehan secara keseluruhan dari 10x percobaan menghasilkan 75%.')


    st.subheader('Eksperimen/Percobaan dengan 30% data testing dan 70% data training')
    with st.expander("Tampilkan"):
        col1, col2 = st.columns(2)
        with col1:
            eks = pd.read_excel('data/evaluasi/30.xls')
            st.write(eks)
        with col2:
            st.markdown('Dari tabel dapat dilihat bahwa eksperimen dengan menggunakan 30% data testing dan 70% data training menyatakan bahwa metode naive bayes multinomial yang memiliki nilai akurasi yang paling tinggi dengan perolehan secara keseluruhan dari 10x percobaan menghasilkan 81%.')


    st.subheader('Eksperimen/Percobaan dengan 40% data testing dan 60% data training')
    with st.expander("Tampilkan"):
        col1, col2 = st.columns(2)
        with col1:
            eks = pd.read_excel('data/evaluasi/40.xls')
            st.write(eks)
        with col2:
            st.markdown('Dari tabel dapat dilihat bahwa eksperimen dengan menggunakan 40% data testing dan 60% data training menyatakan bahwa metode stacking dan random forest yang memiliki nilai akurasi yang paling tinggi dengan perolehan secara keseluruhan dari 10x percobaan menghasilkan 74%.')

    st.subheader('Eksperimen/Percobaan dengan 50% data testing dan 50% data training')
    with st.expander("Tampilkan"):
        col1, col2 = st.columns(2)
        with col1:
            eks = pd.read_excel('data/evaluasi/50.xls')
            st.write(eks)
        with col2:
            st.markdown('Dari tabel dapat dilihat bahwa eksperimen dengan menggunakan 50% data testing dan 50% data training menyatakan bahwa metode stacking yang memiliki nilai akurasi yang paling tinggi dengan perolehan secara keseluruhan dari 10x percobaan menghasilkan 88%.')

    st.subheader('Kesimpulan')
    st.markdown('Berdasar dari 5 eksperimen yang dilakukan maka dapat disimpulkan bahwa metode stacking dengan menggunakan 50% data testing dan data training menghasilkan nilai akurasi yang cukup tinggi yaitu sebesar 88%. Sehingga metode/algoritma ini cocok untuk digunakan sebagai pengklasifikasian komentar youtube yang diambil dari link https://youtu.be/aeF8c348Urc.')