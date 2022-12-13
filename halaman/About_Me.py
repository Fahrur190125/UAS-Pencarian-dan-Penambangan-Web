import streamlit as st
import pandas as pd
def app():
	st.title("KLASIFIKASI KOMENTAR YOUTUBE")
	st.header('About Me')
	st.markdown('Halaman ini merupakan halaman yang menerangkan informasi seputar penulis dan isi dari halaman Website yang dimuat. Informasi tersebut sebagai berikut.')
	st.subheader('Biodata Penulis')
	with st.expander("Tampilkan"):
		col1, col2 = st.columns(2)
		with col1:
			bio = pd.read_excel('data/bio.xlsx')
			st.write(bio)
		with col2:
			st.image("https://raw.githubusercontent.com/Fahrur190125/img/master/foto_diri.jpeg")
	st.subheader('Informasi Penulis')
	with st.expander("Tampilkan"):
		st.markdown('Penulis merupakan Seorang Mahasiswa Aktif Angkatan ‘19 di Universitas Trunojoyo Madura Fakultas Teknik Program Studi Teknik Informatika. Dalam penulisan ini penulis dibimbing oleh Bapak Mula’ab, S.Si., M.Kom., selaku dosen pengampu mata kuliah Pencarian dan Penambangan Web.')
	st.subheader('Informasi Website')
	with st.expander("Tampilkan"):
		st.markdown('Website ini merupakan Website yang dibuat guna memenuhi UAS mata kuliah Pencarian dan Penambangan Web yang diampu oleh  Bapak Mula’ab, S.Si., M.Kom., selaku dosen pengampu mata kuliah Pencarian dan Penambangan Web. Pada Website ini menjelaskan tentang beberapa metode/algoritma yang digunakan untuk mengklasifikasi data dan pada Website ini juga dijelaskan proses pengklasifikasi data website yang diambil dari komentar youtube mulai dari proses crawling, pemberian label, preprocessing, TF-IDF beserta modelling. Dengan adanya Website ini selain untuk memenuhi tugas UAS, penulis juga berharap dapat membantu para pembaca dalam memahami proses klasifikasi')
