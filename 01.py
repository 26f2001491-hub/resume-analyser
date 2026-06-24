from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

users = []

class AboutUser(BaseModel):
    User: str
    College: str
    cgpa: int
    skills: str


@app.post("/User")
def create_user(user: AboutUser):
    users.append(user)
    return {"message": "User Added"}


@app.get("/User")
def get_user():
    return users