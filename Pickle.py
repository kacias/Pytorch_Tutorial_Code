import pickle
import gzip


#===========================
#피클 저장

list = ['a', 'b', 'c']
with open('list_pickle.pickle', 'wb') as f:
    pickle.dump(list, f)

#피클 로드
with open('list_pickle.pickle', 'rb') as f:
    data = pickle.load(f)


print(data)



#=============================
#gzip 압축

data2 = {
    'a': [1, 2.0, 3, 4+6j],
    'b': ("character string", b"byte string"),
    'c': {None, True, False}
}

# save and compress.
with gzip.open('testPickleFile.pickle', 'wb') as f:
    pickle.dump(data2, f)

# load and uncompress.
with gzip.open('testPickleFile.pickle','rb') as f:
    data = pickle.load(f)

print(data)


