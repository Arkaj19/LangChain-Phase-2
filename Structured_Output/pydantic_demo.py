from pydantic import BaseModel, EmailStr,Field
from typing import Optional

class Student(BaseModel):
    name: str = Field(..., description="Enter your name here")
    age: Optional[int] = Field(gt=20, lt=60)
    cgpa: float = Field( ..., gt=0, lt=10, description="Enter your CGPA", examples=5.7)
    email: EmailStr


student = Student( name='Arka',age='23', cgpa=8.9, email='arka@gamil.com')
print(student)