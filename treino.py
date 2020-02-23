import LSTM
from random import sample,shuffle
import datetime as dt
import matplotlib.pyplot as plt
import pickle
import numpy as np

def dados_de_treino(tamanho,max_x,max_d,min_x,min_d):
    """Preparação do conjunto de treino."""
    """
    tamanho: quantidade de elementos no conjunto de treino;
    max_x: número máximo de caracteres no vetor de entrada;
    max_d: número máximo de caracteres no vetor resposta desejada;
    min_x: número mínimo de caracteres no vetor de entrada;
    min_d: número mínimo de caracteres no vetor resposta desejada."""

    texto = open('Livros.txt',encoding='Latin1').read()

    #Retira do texto todos os caracteres especiais não imprimíveis.
    #Estes caracteres foram identificados em análise prévia.
    #Fica só o caractere de nova linha.
    for i,j in enumerate(['\x97','\x93','\x94','\x91','\x92','\x85']):
        texto = ' '.join(texto.split(j))

    #Descobre o alfabeto do qual o texto é composto.
    alfabeto = []
    for i,caracter in enumerate(texto):
        if caracter not in alfabeto:
            alfabeto.append(caracter)
    print('O alfabeto utilizado é {}\n com um número total de símbolos igual a {}.'.format(alfabeto,len(alfabeto)))

    pickle.dump(alfabeto,open('poesia sacra\\alfabeto','wb'))
    print("""O alfabeto foi salvo, em ordem, no arquivo 'poesia sacra\\alfabeto'.""")

    #Cria um mapeamento unívoco dos 88 diferentes caracteres
    #no texto para 88 racionais no conjunto [-44/45,43/45].
    mapa = {}
    for i,caracter in enumerate(alfabeto):
        mapa[caracter] = (i-44)/100
    print('O mapeamento do alfabeto em racionais do conjunto [-44/100,43/100] é\n{}'.format(mapa))

    pickle.dump(mapa,open('poesia sacra\\mapa','wb'))
    print("""O dicionário de mapeamento foi salvo no arquivo 'poesia sacra\\mapa'.""")
    
    frases = texto.split('. ')

    #Criação preliminar do conjunto de treino levando-se em conta os comprimentos máximos.
    x_d = []
    for i,j in enumerate(frases):
        if i>18000:
            break
        k=0
        x = ''
        d = ''
        while True: #Cria x
            if len(x)+len(frases[i+k])+2<=max_x: #+2 conta os 2 caracteres em '. '.
                x += frases[i+k] + '. '
                k += 1
            else: break
        while True: #Cria d
            if len(d)+len(frases[i+k])+2<=max_d: #+2 conta os 2 caracteres em '. '.
                d += frases[i+k] + '. '
                k += 1
            else: break
        x_d.append([x,d])

    #Remove aqueles exemplos de treino que tenham x ou d com tamanho inferior a min_x ou min_d, respectivamente.
    for i,j in enumerate(x_d):
        if len(j[0])<min_x or len(j[1])<min_d:
            del x_d[i]
        #if len(j[0])>max_x or len(j[1])>max_d:
        #    del j
    print('Foi possível produzir um conjunto de treino com {} exemplos adequados aos tamanhos máximo e mínimo definidos.'.format(len(x_d)))

    #Cria uma cópia inteira de x_d em x_d_teste. Os exemplos de treino serão retirados de x_d_teste.
    x_d_teste = [exemplo for i,exemplo in enumerate(x_d)]
    
    x_d = sample(x_d,tamanho)
    print("""O conjunto de treino original foi reduzido para {} elementos como especificado em 'tamanho'.""".format(len(x_d)))

    #Retira de x_d_teste os exemplos de treino.
    x_d_teste = [exemplo for i,exemplo in enumerate(x_d_teste) if exemplo not in x_d]
    print('O conjunto de exemplos de teste tem {} elementos.'.format(len(x_d_teste)))
    
    #Salva os exemplos de teste.
    pickle.dump(x_d_teste,open('poesia sacra\\x_d_teste','wb'))
    print("""Os exemplos de teste foram salvos em 'poesia sacra\\x_d_teste'.""")

    #Executa o mapeamento dos elementos dos exemplos de treino contidos em x_d para algums racionais contidos no conjunto [-44/45,43/45].
    for i,exemplo in enumerate(x_d):
        x_d[i][0] = list( x_d[i][0] )
        x_d[i][1] = list( x_d[i][1] )
        for k,elemento in enumerate(exemplo[0]):
            x_d[i][0][k] = mapa[ x_d[i][0][k] ]
        for k,elemento in enumerate(exemplo[1]):
            x_d[i][1][k] = mapa[ x_d[i][1][k] ]

    #Deita x e d em cama de 1's.
    for i,j in enumerate(x_d):
        #Cama de 1's para x.
        s = 300 - len(j[0])
        s_metade = int(s/2) #Metade inteira, na verdade.
        if s_metade==s/2: #Caso verdadeiro quando s for par.
            esquerda = direita = s_metade
        else:
            esquerda = s_metade
            direita = s_metade + 1
        x_d[i][0] = esquerda*[1] + x_d[i][0] + direita*[1]
        #Cama de 1's para d.
        s = 300 - len(j[1])
        s_metade = int(s/2) #Metade inteira, na verdade.
        if s_metade==s/2: #Caso verdadeiro quando s for par.
            esquerda = direita = s_metade
        else:
            esquerda = s_metade
            direita = s_metade + 1
        x_d[i][1] = esquerda*[1] + x_d[i][1] + direita*[1]

    return x_d

lstm = LSTM.LSTM()

lstm.T = dados_de_treino(tamanho=15000,max_x=300,max_d=300,min_x=100,min_d=100)
print('lstm.T ',len(lstm.T))

lstm.nome = 'poesia_sacra'

lstm.cria_LSTM(bus=300,dim_x=300)

lstm.eta_i, lstm.eta_f, lstm.eta_o, lstm.eta_m = 1e-7,1e-7,1e-7,1e-7

lstm.lote = 25

data_i = dt.datetime.now()
print('Treino iniciado em ',data_i)

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
print('Treino terminado em ',data_f)

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
