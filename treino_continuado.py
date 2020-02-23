from random import sample,shuffle
import datetime as dt
import matplotlib.pyplot as plt
import pickle
import numpy as np

lstm = pickle.load( open('poesia sacra\\poesia_sacra.lstm','rb') )

data_i = dt.datetime.now()
print('Treino continuado iniciado em ',data_i)

secoes = 400
for secao in range(secoes):
    shuffle(lstm.T)
    for i, exemplo in enumerate(lstm.T):
        x,d = exemplo
        C = lstm._C_(x,lstm.h_a)
        h = lstm._h_(x,lstm.h_a)

        lstm.delta_parametros_treinaveis(C,h,d,x)

        lstm.C_a = np.copy(C)
        lstm.h_a = np.copy(h)

        if i % lstm.lote == 0:
            lstm.atualiza_parametros_treinaveis()

    lstm.E.append( lstm._E_(h,d) )

data_f = dt.datetime.now()
print('Treino continuado terminado em ',data_f)

lstm.secoes += secoes

t = [i for i in range(len(lstm.E))]
fig, sp = plt.subplots()
sp.plot(t,lstm.E,'g-',label='Pedagógica')
plt.xlabel('Seções de treino.')
plt.ylabel('E baseada na norma euclidiana.')

#plt.title(r.nome+'_'+str(len(r.secoes)+1)+'\n'+'Épocas: '+str(epoca + 1)+'. '+'Lotes por época: '+str(tamanho-(e+s+r.lote+1))+'.')
plt.savefig('poesia sacra\\lstm.png')

f = open('poesia sacra\\' + lstm.nome + '.lstm' , 'wb' )
pickle.dump(lstm,f)
f.close()
