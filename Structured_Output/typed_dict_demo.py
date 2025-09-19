from typing import TypedDict

class Person(TypedDict):
    name: str
    age : int

person1: Person = { 'name':'Arka', 'age':23}
print(person1)