from fastai import *
import streamlit as st
from fastbook import *
import csv

link_for_train = 'https://www.kaggle.com/code/rayhaank/diabetes-predictor'
#Title
st.title(':violet[ Interactive Report on how different parameters impact diabetes]')
st.subheader("By: Rayhaan Khan")

gluc = 'https://drive.google.com/file/d/1ur8gpMcK5C09tV6GVFthQXFm8pCiwGs6/view?usp=sharing'

learn = load_learner('model.pkl')

if st.button("press to run", type="primary"):
    with open('diabetes.csv', 'a') as file:
        writer = csv.writer(file)
        writer.writerow([pregnancies, glucose, bloodp, SkinThickness, insulin, bmi, DiabetesPedigreeFunction, Age])
        learn = load_learner('model.pkl')
        test = pd.read_csv('diabetes.csv')
        dl = learn.dls.test_dl(test)
        preds = learn.get_preds(dl=dl)
        preds = [x.item() for x in preds[0]]
        test['Outcome'] = preds
        st.warning('ignore first outcome. Use the second one.')
        st.info(preds)


#st.title("Acknowledgements")
#st.markdown("I would like to say a special thank you to my uncle, Dr. Shadab Khan for providing me with this wonderful opportunity, to learn and expand my knowledge of AI, Deep Learning and ML and always providing valuable feedback for my assignments and projects.")

