print("Приветствие")
try:
    x = int(input("Enter X: "))
except:
    print("Error")
finally:
    print("Complied")

print("Сложение двух чисел")

x = True
while x:
    try:
        a = int(input("Enter a: "))
        b = int(input("Enter b: "))
        print("Result",a+b)
        x = False
    except:
        print("Try Again")
    finally:
        print("Complied")




