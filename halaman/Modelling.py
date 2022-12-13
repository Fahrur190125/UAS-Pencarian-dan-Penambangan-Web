from cgitb import text
import streamlit as st
import pandas as pd
import time

def app() :
    st.title('KLASIFIKASI KOMENTAR YOUTUBE')
    from sklearn.model_selection import train_test_split
    data = pd.read_csv('data/main_data.csv')
    column_data = pd.read_csv('data/meta/column_data.csv')
    label = column_data['label'][0]
    y = data[[label]].values
    tf_idf = pd.read_csv('data/tf_idf.csv')
    st.header('Modelling')
    st.markdown('Selanjutnya setelah didapat data TF-IDF, lakukan tahapan modelling dengan menggunakan metode atau algoritma yang paling optimal guna mencari nilai akurasi yang terbaik. Untuk menghasilkan akurasi yang terbaik, maka lakukan eksperimen atau percobaan dengan beberapa metode atau algoritma. Pada pembahasan kali ini kita akan menggunakan beberapa metode yang telah dijelaskan, untuk melakukan eksperimen tersebut agar menghasilkan nilai akurasi yang akurat dengan model yang tepat.')
    ## pembagian data test dengan data secara otomatis
 
    st.subheader('Klasifikasi')
    st.markdown('Untuk memperoleh hasil akurasi yang baik, harus menggunakan metode/algoritma yang tepat untuk dataset. Untuk itu lakukan pengklasifikasian dengan beberapa algoritma/metode berikut untuk menemukan metode/algoritma yang tepat, guna memperoleh nilai akurasi yang terbaik.')
    classfication_list = st.multiselect('Pilih jenis klasifikasi',
            ['naive bayes','knn', 'svm','bagging','stacking','random forest','tree'])

    many_tries = st.number_input('Berapa kali anda ingin mencoba? (Nilai K-Fold)',min_value=1,max_value=20)
    train_size = (st.number_input('Berapa % Dataset yang digunakan sebagai data training',min_value=0,step=1,key='train_size'))/100
    test_size = (st.number_input('Berapa % Dataset yang digunakan sebagai data testing',min_value=0,max_value=int(100-(train_size*100)),step=1,key='test_size'))/100
    data_suffle = st.checkbox('Data Acak',value=True)
    if classfication_list.count('knn') !=0 :
        count_neigh = st.number_input('pilih banyak jumlah neighbors',min_value=3,max_value=len(list(tf_idf.columns)),key='btn_neigh')
    if classfication_list.count('bagging') !=0 :
        count_bagg = st.number_input('tentukan banyak estimator bagging classification',min_value=1,key='btn_bagg')
    if classfication_list.count('random forest') !=0 :
        count_rf = st.number_input('tentukan banyak estimator random forest classification',min_value=1,key='btn_rf')
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

            if i == "svm":
                df = pd.DataFrame(columns=['svm'])
                for j in range(many_tries):
                    text_train, text_test, y_train, y_test = train_test_split(tf_idf, y, test_size = test_size,train_size= train_size,shuffle=data_suffle)
                    from sklearn import svm
                    svm = svm.SVC()
                    svmcs = svm.fit(text_train, y_train)
                    y_pred = svmcs.predict(text_test)
                    
                    # st.write(confusion_matrix(y_test,y_pred))
                    accuracy = classification_report(y_test,y_pred,output_dict=True)['accuracy']
                    df = df.append({'svm' : accuracy},ignore_index=True)
                df_accuracy['svm'] = df["svm"]

            if i == "bagging":
                df = pd.DataFrame(columns=['bagging'])
                for j in range(many_tries):
                    text_train, text_test, y_train, y_test = train_test_split(tf_idf, y, test_size = test_size,train_size= train_size,shuffle=data_suffle)
                    from sklearn import model_selection
                    from sklearn.ensemble import BaggingClassifier
                    from sklearn.tree import DecisionTreeClassifier
                    seed = 20
                    #kfold = model_selection.KFold(n_splits = 3,random_state = seed)
                    # initialize the base classifier
                    base_cls = DecisionTreeClassifier()
                    num_trees = count_bagg
                    model = BaggingClassifier(base_estimator = base_cls,
                        n_estimators = num_trees,
                        random_state = seed)
                    results = model_selection.cross_val_score(model, text_train, y_train)
                        #Nilai base classifier dan hasil nilai classifier disimpan dan akan ditampilkan di grafik
                    score_bagging = results.mean()
                    accuracy = score_bagging
                    df = df.append({'bagging' : accuracy},ignore_index=True)
                df_accuracy['bagging'] = df["bagging"]

            if i == "stacking":
                df = pd.DataFrame(columns=['stacking'])
                for j in range(many_tries):
                    text_train, text_test, y_train, y_test = train_test_split(tf_idf, y, test_size = test_size,train_size= train_size,shuffle=data_suffle)
                    from sklearn.ensemble import RandomForestClassifier
                    from sklearn.svm import LinearSVC
                    from sklearn.naive_bayes import MultinomialNB
                    from sklearn.neighbors import KNeighborsClassifier
                    from sklearn import svm
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.pipeline import make_pipeline
                    from sklearn.ensemble import StackingClassifier
                    estimators = [
                    ('rf', svm.SVC(),'rf1', RandomForestClassifier(n_estimators=20, random_state=42)),
                    ('svr', make_pipeline(StandardScaler(), LinearSVC(random_state=42)))]
                    clf = StackingClassifier(
                        estimators=estimators, final_estimator=LogisticRegression()
                        )
                    st_score = clf.fit(text_train, y_train).score(text_test, y_test)
                    accuracy = st_score
                    df = df.append({'stacking' : accuracy},ignore_index=True)
                df_accuracy['stacking'] = df["stacking"]

            if i == "random forest":
                df = pd.DataFrame(columns=['random forest'])
                for j in range(many_tries):
                    text_train, text_test, y_train, y_test = train_test_split(tf_idf, y, test_size = test_size,train_size= train_size,shuffle=data_suffle)
                    from sklearn.ensemble import RandomForestClassifier
                    #Create a Gaussian Classifier
                    clf=RandomForestClassifier(n_estimators=count_rf)
                    rfc=clf.fit(text_train, y_train)
                    y_pred = rfc.predict(text_test)

                    # st.write(confusion_matrix(y_test,y_pred))
                    accuracy = classification_report(y_test,y_pred,output_dict=True)['accuracy']
                    df = df.append({'random forest' : accuracy},ignore_index=True)
                df_accuracy['random forest'] = df["random forest"]

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
    with st.expander("Lihat Hasil"):
        st.line_chart(df_accuracy)
        st.caption('data akurasi tiap percobaan')
        st.write(df_accuracy)
        df_accuracy.to_csv('data/akurasi.csv',index=False)
        if classfication_list.count('tree') !=0:
            st.write('GAMBAR POHON KEPUTUSAN')
            for i in range(many_tries):
                st.caption(f'train ke {i+1}')
                st.image(f"data/pictures/classification/tree/train-{i}.png")
    # st.write(text_train)
    # st.write(text_test)
