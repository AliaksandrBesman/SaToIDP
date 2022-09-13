import numpy as np

a1 = np.array(range(1,10))
print("Cound Axis:", a1.ndim)
print(a1)
print("Shape", a1.shape)

list1 = list(range(5))
print("Start list",list1)
list1.reverse()
print("Reversed list",list1)
a2 = np.array([[list1],[list(range(5))]])
print("Array Axis2:\n",a2)
print("Shape", a2.shape)
# Axis 3
a3 = np.array([[[1,2,3],[3,2,1]],[[1,2,3],[3,2,1]]])
print(a3)
print("Shape", a3.shape)

ones_a2 = np.ones((5, 6))
print("Массив единиц:\n ")
print(ones_a2)

ones_a3 = np.ones((5, 6, 2))
print("Массив единиц:\n ")
print(ones_a3)

zero_a3 = np.zeros((5, 6, 2))
print("Массив единиц:\n ")
print(zero_a3)

identity_a = np.identity(4)
print("Единичная матрица:\n ")
print(identity_a)

empty_a2 = np.empty((3, 3))
print("Пустая матрица")
print(empty_a2)

arange_a1 = np.arange(1,6,2)
print("Arange: ")
print(arange_a1)

A = np.arange(8)
print(A)
A1= A.reshape((4,2))
print(A1)
A2 = A.reshape((2, -1))
print(A2)

AT = A2.T
print(AT)

A = np.arange(3)

print("Копирование")
print(np.tile(A,(2,2)))
print(np.tile(A,(3,1)))


# Операции с матрицами
D = np.arange(9).reshape((3,3))
print("D\n",D)
E = np.arange(2, 11).reshape((3,3))
print("E\n",E)

print("D+E\n",D+E)
print("D*E\n",D*E)
print("D+10\n",D+10)
print("D*10\n",D*10)
print("D**10\n",D**10)
print("dot(D,E)\n",np.dot(D,E))
print("sum(axis=1",D.sum(axis=1))
print("sum(axis=0",D.sum(axis=0))

# Срезы
A = np.arange(10)
print("A:\n",A)
print("A[3:7]:\n",A[3:7])
print("A[1:8:2]:\n",A[1:8:2])

A = A.reshape((5,-1))
print(A)

print("A[2:4]:\n",A[2:4])

# Условия
A = np.arange(10)
print(A[A % 2 == 0]) # выведет все четные элементы массива A
print(A[np.logical_and(A != 5, A != 0)]) # выведет все элементы массива A, которые не равны нулю и пяти
