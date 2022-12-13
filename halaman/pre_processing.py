import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import time
import timeit
import swifter
import re
import string

#remove char spesial
def remove_special(text):
    # remove tab, new line, ans back slice
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\'," ").replace('\\f'," ").replace('\\r'," ")
    # remove non ASCII (emoticon, chinese word, .etc)
    text = text.encode('ascii', 'replace').decode('ascii')
    # remove mention, link, hashtag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
    # remove incomplete URL
    return text.replace("http://", " ").replace("https://", " ")

#remove number
def remove_number(text):
    return  re.sub(r"\d+", "", text)

#remove punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans("","",string.punctuation))


#remove whitespace leading & trailing
def remove_whitespace_LT(text):
    return text.strip()

#remove multiple whitespace into single whitespace
def remove_whitespace_multiple(text):
    return re.sub('\s+',' ',text)

# remove single char
def remove_singl_char(text):
    return re.sub(r"\b[a-zA-Z]\b", " ", text)

def word_tokenize_wrapper(text):
    return word_tokenize(text)

nltk.download('stopwords')

list_stopwords = stopwords.words('indonesian')

# append additional stopword
list_stopwords.extend(["yg", "dg", "rt", "dgn", "ny", "d", 'klo', 
                       'kalo', 'amp', 'biar', 'bikin', 'bilang', 
                       'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 
                       'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 
                       'jd', 'jgn', 'sdh', 'aja', 'n', 't', 
                       'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                       '&amp', 'yah'])

# convert list to dictionary
list_stopwords = set(list_stopwords)

#Menghapus Stopword dari list token
def stopwords_removal(words):
    return [word for word in words if word not in list_stopwords]

def text_preprocessing(data,step,column='column',list_stem_stop_leng=['eng','ind'],index_detect_leng=0):
    global start
    global stop
    col1, col2 = st.columns(2)
    with col1 :
        st.subheader('Data sebelum di proses')
        st.write(data)
        if step == "Stopwords Removal/Filtering":
            language = st.selectbox("Pilih bahasa yang anda gunakan :",
            list_stem_stop_leng,index=index_detect_leng)
        if step == 'Stemming' :
            language = st.selectbox("Pilih bahasa yang anda gunakan :",
            list_stem_stop_leng,index=index_detect_leng)
    with col2 :
        with st.spinner('tunggu sebentar ...'):
            time.sleep(2)
            if step == 'Case Folding':
                start = timeit.default_timer()
                data[column] = data[column].str.lower()
                stop = timeit.default_timer()
            if step == 'Stopwords Removal/Filtering' :
                import nltk
                start = timeit.default_timer()
                data[column] = data[column].apply(word_tokenize_wrapper)
                data[column] = data[column].apply(stopwords_removal)
                data[column] =data[column].apply(lambda x: [token for token in x if token not in stopwords.words(language)])
                stop = timeit.default_timer()
                data[column] = data[column].str.join(" ")
            if step == 'Stemming' :
                if language == 'indonesian' :
                    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
                    factory = StemmerFactory()
                    stemmer = factory.create_stemmer()
                    data[column] = data[column].str.split()
                    start = timeit.default_timer()
                    data[column] = data[column].swifter.apply(lambda x: [stemmer.stem(y) for y in x])
                    data[column] = data[column].str.join(" ")
                    stop = timeit.default_timer()
                else :
                    from nltk.stem.snowball import SnowballStemmer
                    stemmer = SnowballStemmer(language)
                    data[column] = data[column].str.split()
                    start = timeit.default_timer()
                    data[column] = data[column].apply(lambda x: [stemmer.stem(y) for y in x])
                    stop = timeit.default_timer()
                    data[column] = data[column].str.join(" ")
            data.to_csv('data/data_branch.csv',index=False)
            data_branch = pd.read_csv('data/data_branch.csv')
        
            st.subheader('Data setelah di proses')
            st.write(data_branch)
            st.write('lama proses : ', stop-start,' detik')

            if st.button('simpan data'):
                data_branch.to_csv('data/main_data.csv',index=False)
                st.success('Berhasil disimpan')
def app():
    #Your statements here
    data = pd.read_csv('data/main_data.csv')
    column_data = pd.read_csv('data/meta/column_data.csv')
    column = column_data['column'][0]
    from googletrans import Translator,LANGUAGES

    translator = Translator()
    list_stem_stop_leng = ['english', 'indonesian']
    index_detect_leng =list_stem_stop_leng.index(LANGUAGES[translator.detect(data[column][0]).lang])
    st.title('KLASIFIKASI KOMENTAR YOUTUBE')
    st.header('Preprocessing Text')
    st.markdown('Setelah proses crawling, selanjutnya lakukan prepocessing text, yaitu sebuah proses mesin yang digunakan untuk menyeleksi data teks agar lebih terstruktur dengan melalui beberapa tahapan-tahapan yang meliputi tahapan Case Folding, Tokenizing, Stopwords Removal/Filtering dan Stemming.')
    st.sidebar.markdown("lakukan preprosesing data")
    steps = st.sidebar.radio('Langkah langkah pre-prosessing : ',('Case Folding','Tokenizing','Stopwords Removal/Filtering','Stemming'))
    st.subheader(f'Preprosesing - {steps}')
    global start
    global stop
    if steps == 'Case Folding':
        st.markdown('Setelah berhasil mengambil dataset, selanjutnya proses prepocessing, tahapan Case Folding yaitu tahapan pertama untuk melakukan prepocessing text dengan mengubah text menjadi huruf kecil semua.')
        text_preprocessing(data,'Case Folding',column)
    if steps == 'Tokenizing':
        st.markdown('Tokenizing atau disebut juga tahap Lexical Analysis adalah proses pemotongan teks menjadi bagian-bagian yang lebih kecil, yang disebut token. Pada proses ini juga dilakukan penghilangan angka, tanda baca dan karakter lain yang dianggap tidak memiliki pengaruh terhadap pemrosesan teks.')
        options = st.multiselect('Pilih proses',
            ['hapus karakter spesial','hapus angka', 'hapus tanda baca', 'hapus single karakter','hapus spasi'])
        start = timeit.default_timer()
        for option in options:
            if option == 'hapus karakter spesial':
                st.markdown('**Menghilangkan/Menghapus karakter Spesial**')
                st.markdown('Proses penghapusan karakter spesial dalam dataset yang tidak penting.')
                data[column] = data[column].apply(remove_special)
            if option == 'hapus angka':
                st.markdown('**Menghilangkan/Menghapus Angka**')
                st.markdown('Proses penghapusan angka dalam dataset yang tidak penting.')
                data[column] = data[column].apply(remove_number)
            if option == "hapus tanda baca":
                st.markdown('**Menghilangkan/Menghapus Tanda Baca**')
                st.markdown('Proses menghilangkan tanda baca dalam text yang tidak dibutuhkan/diperlukan')
                data[column] = data[column].apply(remove_punctuation)
            if option == 'hapus single karakter':
                st.markdown('**Menghilangkan/Menghapus Single Karakter**')
                st.markdown('Proses penghapusan single karakter yang tidak berpengaruh')
                data[column] = data[column].apply(remove_singl_char)
            if option == 'hapus spasi':
                st.markdown('**Menghilangkan Spasi**')
                st.markdown('Proses menghilangkan spasi  yang tidah dibutuhkan')
                data[column] = data[column].apply(remove_whitespace_multiple)
                data[column] = data[column].apply(remove_whitespace_LT)
        stop = timeit.default_timer()
    if steps == "Stopwords Removal/Filtering":
        st.markdown('Lanjutan dari tahapan tokenizing adalah tahapan filtering yang digunakan untuk mengambil kata-kata yang penting dari hasil token tadi. Kata umum yang biasanya muncul dan tidak memiliki makna disebut dengan stopword. Penghilangan stopword ini dapat mengurangi ukuran index dan waktu pemrosesan. Selain itu, juga dapat mengurangi level noise.')
        text_preprocessing(data,'Stopwords Removal/Filtering',column,list_stem_stop_leng,index_detect_leng)
    if steps == 'Stemming' :
        st.markdown('Tahap stemming adalah tahapan yang juga diperlukan untuk memperkecil jumlah indeks yang berbeda dari satu data sehingga sebuah kata yang memiliki suffix maupun prefix akan kembali ke bentuk dasarnya. Selain itu juga untuk melakukan pengelompokan kata-kata lain yang memiliki kata dasar dan arti yang serupa namun memiliki bentuk yang berbeda karena mendapatkan imbuhan yang berbeda pula.')
        text_preprocessing(data, 'Stemming',column,list_stem_stop_leng,index_detect_leng)
