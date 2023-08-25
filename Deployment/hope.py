from flask import Flask,render_template,request
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import pandas as pd
import numpy as np

df=pickle.load(open('pop.pkl','rb'))
m=pickle.load(open('title.pkl','rb'))



app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html',
                           book_name = list(df['Book_title'].values),
                           desc=list(df['Description'].values),
                           votes=list(df['Reviews'].values),
                           rating=list(df['Rating'].values)
                           )

@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')

@app.route('/recommend_books',methods=['post'])
def recommend():
    user_input=request.form.get('user_input')
    # Locating target element by its specific value
    feature_locate = 'Book_title'
    feature_show = 'Book_title'
    index_of_element = df[df[feature_locate] == user_input].index.values[0]

    # Finding its value to show
    show_value_of_element = df.iloc[index_of_element][feature_show]

    # Dropping target element from df
    df_without = df.drop(index_of_element).reset_index().drop(['index'], axis=1)

    # Dropping target element from vectors array
    title_vectors = list(m)
    target = title_vectors.pop(index_of_element).reshape(1, -1)
    title_vectors = np.array(title_vectors)

    # Finding cosine similarity between vectors
    most_similar_sklearn = cosine_similarity(target, title_vectors)[0]

    # Sorting coefs in desc order
    idx = (-most_similar_sklearn).argsort()

    # Finding features of similar objects by index
    all_values = df_without[[feature_show]]
    for index in idx:
        simular = all_values.values[idx]

    recommendations_df = pd.DataFrame({feature_show: show_value_of_element,
                                       "rec_1": simular[0][0],
                                       "rec_2": simular[1][0],
                                       "rec_3": simular[2][0],
                                       "rec_4": simular[3][0],
                                       "rec_5": simular[4][0]}, index=[0])





    #print(recommendations_df.values.T.tolist()[0])

    #return (recommendations_df.values.T.tolist())

    return render_template('recommend.html',data=recommendations_df.values.T.tolist())


if __name__=='__main__':
    app.run(debug=True)
