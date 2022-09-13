print("Массив или Список(list): numbers = [1, 2, 3, 4, 5]")
numbers = [1, 2, 3, 4, 5]
print("Также для создания списка можно использовать конструктор list():")
numbers1 = []
numbers2 = list()

print("Сравнение list(element) and value=elements")
numbers4 = numbers
print("Type vnumbers4 = numbers: ", type(numbers4))
numbers5 = list(numbers)
print("Type numbers5 = list(numbers): ", type(numbers5))

print("Перебор элекментов списка for")
companies = ["Microsoft", "Google", "Oracle", "Apple"]
for item in companies:
    print(item)

print("Перебор элекментов списка while")
companies = ["Microsoft", "Google", "Oracle", "Apple"]
i = 0
while i < len(companies):
    print(companies[i])
    i += 1

print("Сортировка")
users = ["Tom", "bob", "alice", "Sam", "Bill"]

users.sort(key=str.lower)
print(users)

print("Разное содержимое списков")
users = [
    ["Tom", 29],
    "alice",
    ["Bob", 27]
]

for user in users:
    for item in user:
        print(item, end=" | ")


print()
objects2 = {1: "Tom", "2": True, 1: 100.6}

for key in objects2:
        print(key,objects2[key], end=" | ")
