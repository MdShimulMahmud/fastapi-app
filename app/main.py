import json
import pickle

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class model_input(BaseModel):
    total_used_of_lights: int
    total_used_of_ac: int
    total_used_of_fans: int
    total_used_of_refrigerator: int
    total_used_of_motor: int
    total_used_of_tv: int
    total_used_of_washing_machine: int
    total_used_of_oven: int
    total_used_of_desktop: int
    total_used_of_pressure_cooker: int
    month: int


# loading the saved model
model = pickle.load(open('model/piperf.sav', 'rb'))


@app.post('/predict')
def diabetes_pred(input_parameters: model_input):

    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)

    lights = input_dictionary['total_used_of_lights']
    ac = input_dictionary['total_used_of_ac']
    fans = input_dictionary['total_used_of_fans']
    refrigerator = input_dictionary['total_used_of_refrigerator']
    motor = input_dictionary['total_used_of_motor']
    tv = input_dictionary['total_used_of_tv']
    washing = input_dictionary['total_used_of_washing_machine']
    oven = input_dictionary['total_used_of_oven']
    desktop = input_dictionary['total_used_of_desktop']
    cooker = input_dictionary['total_used_of_pressure_cooker']
    month = input_dictionary['month']

    input_list = [lights, ac, fans, refrigerator, motor,
                  tv, washing, oven, desktop, cooker, month]

    prediction = model.predict([input_list])

    return {round(prediction[0], 3)}
