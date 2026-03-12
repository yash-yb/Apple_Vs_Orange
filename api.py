from os import mkdir, remove

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import testing
import CNN_gen1

app = FastAPI()
temp_folder="__test"
try:
    mkdir(temp_folder)   
except FileExistsError:
    pass


# CORS configuration: allow all origins during development.
# In production, replace ["*"] with a list of allowed origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/CNN_gen1_test")
async def test(train_file:UploadFile = File(...)):
    data=await train_file.read()    
    with open(f"{temp_folder}/{train_file.filename}", "wb") as f:
        f.write(data)
    data_model = testing.classify(f"{temp_folder}/{train_file.filename}")
    data_model["File_addr"]=f"{temp_folder}/{train_file.filename}"
    return data_model


@app.post("/CNN_gen1_correct")
async def delete_file(file_addr:str):
    remove(file_addr)
    return {"message" : "File Removed Successfully"}


@app.post("/CNN_gen1_incorrect")
async def retrain_model(file_addr, label):
    #CNN_gen1.py retrain feedback
    #CNN_gen1.continue_training(file_addr,label)
    remove(file_addr)
    return {"message": "Successfully Model Trained"}

