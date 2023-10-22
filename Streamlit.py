import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler




#Fetching the model
#model = pickle.load(open('trained_model.sav','rb'))
# Loading the model
loaded_model = joblib.load('trained_model.pkl')


def predict(movement_reactions,mentality_composure,passing,potential,dribbling,
                                power_shot_power,physic,mentality_vision,attacking_short_passing,skill_long_passing):
    data_df = pd.DataFrame({"movement_reactions":[float(movement_reactions)],"mentality_composure":[float(mentality_composure)],
                            "passing":[float(passing)],"potential":[float(potential)],"dribbling":[float(dribbling)],
                            "power_shot_power":[float(power_shot_power)],"physic":[float(physic)],"mentality_vision":[float(mentality_vision)],
                            "attacking_short_passing":[float(attacking_short_passing)],"skill_long_passing":[float(skill_long_passing)]})
    
    #Pre-processing the data
    #Scaling the features 
    model_scaler = StandardScaler()
    scaled_features=model_scaler.fit_transform(data_df)
    new_df = pd.DataFrame(scaled_features, columns=data_df.columns)


    prediction = loaded_model.predict(new_df)
    return prediction



def main():
    #st.title("Arrest count prediction")
    st.write("This message is displayed in streamlit")
    html_temp = """
<div style = "background-color:tomato;padding:10px">
     <h2 style="color:blue;text-align:center;">
      Streamlit Fifa Player Prediction ML App </h2>
      </div>
      """
    

    st.markdown(html_temp,unsafe_allow_html=True)

    #Taking in the features for the prediction.
    
    player_name = st.text_input("Player name", "Type here")
    movement_reactions = st.text_input("movement_reactions","Type here")
    mentality_composure = st.text_input("mentality_composure","Type here")
    passing = st.text_input("passing","Type here")
    potential = st.text_input("potential","Type here")
    dribbling = st.text_input("dribbling","Type here")
    power_shot_power = st.text_input("power_shot_power","Type here")
    physic = st.text_input("physic","Type here")
    mentality_vision = st.text_input("mentality_vision","Type here")
    attacking_short_passing = st.text_input("attacking_short_passing","Type here")
    skill_long_passing = st.text_input("skill_long_passing","Type here")

    if st.button("Predict"):
        output = predict(movement_reactions,mentality_composure,passing,potential,dribbling,
                                power_shot_power,physic,mentality_vision,attacking_short_passing,skill_long_passing)
        
        st.success(f"Player: {player_name} ; Overall Rating: {output[0]}")


if __name__=='__main__':
        main()
