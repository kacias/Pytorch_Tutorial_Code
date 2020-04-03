'''
class CustomNumbers:
    def __init__(self):
        self._numbers = [n for n in range(1, 11)]


a = CustomNumbers()


print(a._numbers[2:5])


'''

#img
class myClass:
    def __init__(self):

        self.numbers = [1,2,3,4,6,7]

    def __getitem__(self, idx):
        return self.numbers

#csv
class myCSV:
    def __init__(self):
        pass

    def __getitem__(self, item):
        return 0



a  = myClass()
b  = myCSV()


print(a.numbers[1])


for i in zip(a, b):
    print(i)


'''
for idx, i in enumerate(a):
    print(a[idx].numbers)

'''





'''
a = [1,2,3]

#a가 iterable 객체이어야 함
for i in a:
    print(i)

'''
