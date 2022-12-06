from cgitb import text
import streamlit as st
import pandas as pd
import time

def app() :
    from sklearn.model_selection import train_test_split
    data = pd.read_csv('data/main_data.csv')
    column_data = pd.read_csv('data/meta/column_data.csv')
    label = column_data['label'][0]
    y = data[[label]].values
    tf_idf = pd.read_csv('data/tf_idf.csv')
    ## pembagian data test dengan data secara otomatis

    st.subheader('Klasifikasi')

    classfication_list = st.multiselect('Pilih jenis klasifikasi',
            ['naive bayes','tree','knn'])

    many_tries = st.number_input('Ingin berapa kali anda mencoba',min_value=1,max_value=20,value=10)
    train_size = (st.number_input('Data Training Sebanyak',min_value=0,max_value=100,value=80,step=1,key='train_size'))/100
    test_size = (st.number_input('Data Training Sebanyak',min_value=0,max_value=int(100-(train_size*100)),value=20,step=1,key='test_size'))/100
    data_suffle = st.checkbox('Acak Data',value=True)
    if classfication_list.count('knn') !=0 :
        count_neigh = st.number_input('pilih banyak jumlah neighbors',min_value=1,max_value=len(list(tf_idf.columns)),value=3,key='btn_neigh')

    df_accuracy = pd.DataFrame(columns=classfication_list)
    with st.spinner('Wait for it...'):
        for i in classfication_list :
            from sklearn.metrics import classification_report, confusion_matrix
            if i == "naive bayes":
                df = pd.DataFrame(columns=['naive bayes'])
                for j in range(many_tries):
                    from sklearn.naive_bayes import MultinomialNB
                    text_train, text_test, y_train, y_test = train_test_split(tf_idf, y, test_size = test_size,train_size= train_size,shuffle=data_suffle)
                    modelnb = MultinomialNB()
                    nbtrain = modelnb.fit(text_train, y_train)
                    y_pred = nbtrain.predict(text_test)
                    
                    # st.write(confusion_matrix(y_test,y_pred))
                    accuracy = classification_report(y_test,y_pred,output_dict=True)['accuracy']
                    df = df.append({'naive bayes' : accuracy},ignore_index=True)
                df_accuracy['naive bayes'] = df["naive bayes"]
            if i == 'tree' :
                df = pd.DataFrame(columns=['tree'])
                for j in range(many_tries):
                    from sklearn import tree
                    text_train, text_test, y_train, y_test = train_test_split(tf_idf, y, test_size = test_size,train_size= train_size,shuffle=data_suffle)
                    from sklearn.tree import DecisionTreeClassifier
                    from sklearn import tree 
                    import matplotlib.pyplot as plt
                    # Create Decision Tree classifer object
                    clf = DecisionTreeClassifier()
                    # Train Decision Tree Classifer
                    clf = clf.fit(text_train,y_train)
                    #Predict the response for test dataset
                    y_pred = clf.predict(text_test)
                    fig = plt.figure(figsize=(25,20))
                    _ = tree.plot_tree(clf,feature_names=list(clf.feature_names_in_),class_names=list(clf.classes_),
                    filled=True)
                    fig.savefig(f"data/pictures/classification/tree/train-{j}.png")
                    accuracy = classification_report(y_test,y_pred,output_dict=True)['accuracy']
                    df = df.append({'tree' : accuracy},ignore_index=True)
                    # text_representation = tree.export_text(clf)
                    # st.write(text_representation)
                    # st.write(tree.plot_tree(clf))
                df_accuracy["tree"] = df["tree"]
            if i == 'knn' :
                df = pd.DataFrame(columns=['knn'])
                for j in range(many_tries):
                    text_train, text_test, y_train, y_test = train_test_split(tf_idf, y, test_size = test_size,train_size= train_size,shuffle=data_suffle)
                    from sklearn.neighbors import KNeighborsClassifier
                    neigh = KNeighborsClassifier(n_neighbors=count_neigh)
                    neightrain = neigh.fit(text_train, y_train)
                    y_pred = neightrain.predict(text_test)
                    accuracy = classification_report(y_test,y_pred,output_dict=True)['accuracy']
                    df = df.append({'knn' : accuracy},ignore_index=True)
                df_accuracy['knn'] = df["knn"]

    with st.expander("Lihat Hasil"):
        st.line_chart(df_accuracy)
        st.caption('data akurasi tiap percobaan')
        st.write(df_accuracy)
        if classfication_list.count('tree') !=0:
            st.write('GAMBAR POHON KEPUTUSAN')
            for i in range(many_tries):
                st.caption(f'train ke {i+1}')
                st.image(f"data/pictures/classification/tree/train-{i}.png")
    # st.write(text_train)
    # st.write(text_test)
