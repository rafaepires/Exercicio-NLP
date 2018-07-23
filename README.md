max_sentences = 60
max_trigrams = 80
train_word2vec = 0
save_train_data = 0
train_tf = 0



Utilizei 60 músicas de cada classe para treinamento, sendo que de cada uma das músicas foi extraido no máximo 80 trigramas para formar a entrada de uma rede neural Multi Layer Perceptron com 3 hidden layers de 1.000 unidades cada uma, função de ativação ReLU, otimizador Adam e Learning rate = 1e-3


Infelizmente não fiz a parte da web, sendo assim para ser possível testar o código fiz a divisão em 4 etapas:


1 train_word2vec: quando habilitado (1) gera a matriz de embedding utilizado os arquivos .CSV.

2 save_train_data: utiliza o embedding gerado para gerar a entrada e labels da MLP.

3 train_tf: treina a rede neural utilizando TensorFlow

4 Caso todas as 3 variaveis estejam (0) desabilitadas é feito o predict da música que deve ser inserida no arquivo "fileTeste.csv" observando que a música deve estar entre "".