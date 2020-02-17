import numpy as np
class LSTM():
    def __init__(self):
        #bus é o barramento. bus é o número de neurônios em cada portão e na memória - é igual para todos.
        self.bus = 0
        #dim_x é a dimensionalidade do vetor de netrada x.
        self.dim_x = 0
        #Taxas de aprendizagem.
        self.eta_i, self.eta_f, self.eta_o, self.eta_m = 0,0,0,0
        #Parâmetro para treino em lote.
        self.lote = 1
        self.W_i, self.U_i, self.b_i = 0,0,0
        self.W_f, self.U_f, self.b_f = 0,0,0
        self.W_o, self.U_o, self.b_o = 0,0,0
        self.W_m, self.U_m, self.b_m = 0,0,0
        #Célula no tempo anterior.
        self.C_a = 0
        #Saída da rede no tempo anterior.
        self.h_a = 0
        self.E = []
        #Conjunto de exemplos de treino.
        self.T = []
        self.nome = ''
        #Número de seções de treino. Em cada seção, todo o conjunto de treino é apresentado.
        self.secoes = 0
    
    def cria_LSTM(self, bus,dim_x):
        self.bus = bus
        self.dim_x = dim_x

        self.W_i = np.zeros(shape=(self.dim_x,self.bus))
        self.U_i = np.zeros(shape=(self.bus,self.bus))
        self.b_i = np.array(self.bus*[0.0])

        self.W_f = np.zeros(shape=(self.dim_x,self.bus))
        self.U_f = np.zeros(shape=(self.bus,self.bus))
        self.b_f = np.array(self.bus*[0.0])

        self.W_o = np.zeros(shape=(self.dim_x,self.bus))
        self.U_o = np.zeros(shape=(self.bus,self.bus))
        self.b_o = np.array(self.bus*[0.0])

        self.W_m = np.zeros(shape=(self.dim_x,self.bus))
        self.U_m = np.zeros(shape=(self.bus,self.bus))
        self.b_m = np.array(self.bus*[0.0])

        self.C_a = np.array(self.bus*[0.0])
        #A primeiríssima saída da rede nao é calculada pela rede.
        self.h_a = np.random.rand(self.bus)

        #É preciso guardar os valores atuais abaixo para o uso de dE_dv de cada elemento.
        self.i = 0
        self.f = 0
        self.m = 0
        self.o = 0
        self.C = 0
        self.h = 0

        #É preciso armazenar as variações dos parâmetros treináveis durante um lote.
        self.delta_W_i = np.zeros(shape=(self.dim_x,self.bus))
        self.delta_U_i = np.zeros(shape=(self.bus,self.bus))
        self.delta_b_i = np.array(self.bus*[0.0])

        self.delta_W_f = np.zeros(shape=(self.dim_x,self.bus))
        self.delta_U_f = np.zeros(shape=(self.bus,self.bus))
        self.delta_b_f = np.array(self.bus*[0.0])

        self.delta_W_o = np.zeros(shape=(self.dim_x,self.bus))
        self.delta_U_o = np.zeros(shape=(self.bus,self.bus))
        self.delta_b_o = np.array(self.bus*[0.0])

        self.delta_W_m = np.zeros(shape=(self.dim_x,self.bus))
        self.delta_U_m = np.zeros(shape=(self.bus,self.bus))
        self.delta_b_m = np.array(self.bus*[0.0])

    def sigmoid(self,z):
        return 1 / (1 + np.exp( -z ) )

    #Elementos da LSTM.
    def _i_(self,x,h_a):
        self.i = self.sigmoid( np.matmul(x,self.W_i) + np.matmul(h_a,self.U_i) + self.b_i)
        return self.i

    def _f_(self,x,h_a):
        self.f = self.sigmoid( np.matmul(x,self.W_f) + np.matmul(h_a,self.U_f) + self.b_f)
        return self.f

    def _o_(self,x,h_a):
        self.o = self.sigmoid( np.matmul(x,self.W_o) + np.matmul(h_a,self.U_o) + self.b_o)
        return self.o

    def _m_(self,x,h_a):
        self.m = np.tanh( np.matmul(x,self.W_m) + np.matmul(h_a,self.U_m) + self.b_m)
        return self.m

    def _C_(self,x,h_a):
        return ( self.C_a * self._i_(x,h_a) ) + ( self._f_(x,h_a) * self._m_(x,h_a) )

    def _h_(self,x,h_a):
        return np.tanh( self.C_a ) * self._o_(x,h_a)


    #Gradientes locais de cada elemento da LSTM.
    def dE_dv_i(self,C,h,d):
        return (h-d)*(1-C**2)*self.o*self.C_a*self.i*(1-self.i)

    def dE_dv_f(self,C,h,d):
        return (h-d)*(1-C**2)*self.o*self.m*self.f*(1-self.f)

    def dE_dv_m(self,C,h,d):
        return (h-d)*(1-C**2)*self.o*self.f*(1-(self.m)**2)

    def dE_dv_o(self,C,h,d):
        return (h-d)*np.tanh(C)*self.o*(1-self.o)

    
    def delta_parametros_treinaveis(self,C,h,d,x):
        self.delta_W_i += self.eta_i * np.outer( x , self.dE_dv_i(C,h,d) )
        self.delta_U_i += self.eta_i * np.outer( h , self.dE_dv_i(C,h,d) )
        self.delta_b_i += self.eta_i * self.dE_dv_i(C,h,d)

        self.delta_W_f += self.eta_f * np.outer( x , self.dE_dv_f(C,h,d) )
        self.delta_U_f += self.eta_f * np.outer( h , self.dE_dv_f(C,h,d) )
        self.delta_b_f += self.eta_f * self.dE_dv_f(C,h,d)

        self.delta_W_m += self.eta_m * np.outer( x , self.dE_dv_m(C,h,d) )
        self.delta_U_m += self.eta_m * np.outer( h , self.dE_dv_m(C,h,d) )
        self.delta_b_m += self.eta_m * self.dE_dv_m(C,h,d)

        self.delta_W_o += self.eta_o * np.outer( x , self.dE_dv_o(C,h,d) )
        self.delta_U_o += self.eta_o * np.outer( h , self.dE_dv_o(C,h,d) )
        self.delta_b_o += self.eta_o * self.dE_dv_o(C,h,d)


    def atualiza_parametros_treinaveis(self):
        self.W_i -= self.delta_W_i
        self.U_i -= self.delta_U_i
        self.b_i -= self.delta_b_i

        self.W_f -= self.delta_W_f
        self.U_f -= self.delta_U_f
        self.b_f -= self.delta_b_f

        self.W_m -= self.delta_W_m
        self.U_m -= self.delta_U_m
        self.b_m -= self.delta_b_m

        self.W_o -= self.delta_W_o
        self.U_o -= self.delta_U_o
        self.b_o -= self.delta_b_o

        #Anula todas as variações após cada atualização.
        self.delta_W_i = np.zeros(shape=(self.dim_x,self.bus))
        self.delta_U_i = np.zeros(shape=(self.bus,self.bus))
        self.delta_b_i = np.array(self.bus*[0.0])

        self.delta_W_f = np.zeros(shape=(self.dim_x,self.bus))
        self.delta_U_f = np.zeros(shape=(self.bus,self.bus))
        self.delta_b_f = np.array(self.bus*[0.0])

        self.delta_W_o = np.zeros(shape=(self.dim_x,self.bus))
        self.delta_U_o = np.zeros(shape=(self.bus,self.bus))
        self.delta_b_o = np.array(self.bus*[0.0])

        self.delta_W_m = np.zeros(shape=(self.dim_x,self.bus))
        self.delta_U_m = np.zeros(shape=(self.bus,self.bus))
        self.delta_b_m = np.array(self.bus*[0.0])

    def _E_(self,h,d):
        return 0.5 * np.sum( (h - d)**2 )
