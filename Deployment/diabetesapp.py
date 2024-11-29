from fastai import *
from fastbook import *
import streamlit as st
import csv

link_for_train = 'https://www.kaggle.com/code/rayhaank/diabetes-predictor'
#Title
st.title(':violet[ Interactive Report on how different parameters impact diabetes]')
st.subheader("By: Rayhaan Khan")

gluc = 'https://drive.google.com/file/d/1ur8gpMcK5C09tV6GVFthQXFm8pCiwGs6/view?usp=sharing'

learn = load_learner('Deployment/model.pkl')

if st.button("press to run", type="primary"):
    with open('Deployment/diabetes.csv', 'a') as file:
        writer = csv.writer(file)
        writer.writerow([pregnancies, glucose, bloodp, SkinThickness, insulin, bmi, DiabetesPedigreeFunction, Age])
        learn = load_learner('Deployment/model.pkl')
        test = pd.read_csv('Deployment/diabetes.csv')
        dl = learn.dls.test_dl(test)
        preds = learn.get_preds(dl=dl)
        preds = [x.item() for x in preds[0]]
        test['Outcome'] = preds
        st.warning('ignore first outcome. Use the second one.')
        st.info(preds)



