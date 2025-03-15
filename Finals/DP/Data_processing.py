import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

# 1. Carregar os dados de treinamento
# Agora carregando o arquivo "treinamento.csv"
train_df = pd.read_csv('treino.csv')
print("Amostra dos dados de treinamento:")
print(train_df.head())

# 2. Preparação dos dados
# Remove a coluna 'id' (pois é apenas identificador) e separe a variável target
X = train_df.drop(['id', 'target'], axis=1)
y = train_df['target']

# Dividir os dados em treino e validação (80% treino, 20% validação)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)

# Normalizar as features para melhorar o desempenho do modelo
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# 3. Definir e treinar a rede neural (MLPClassifier)
# Utilizamos uma arquitetura com duas camadas ocultas com 64 e 32 neurônios, respectivamente.
model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
                      max_iter=200, random_state=42)
model.fit(X_train_scaled, y_train)

# 4. Avaliar o modelo utilizando a métrica F1-Score (média harmônica entre precisão e revocação)
y_val_pred = model.predict(X_val_scaled)
f1 = f1_score(y_val, y_val_pred, average='macro')
print("F1-Score no conjunto de validação:", f1)

# 5. Carregar os dados de teste
# Agora carregando o arquivo "teste.csv"
test_df = pd.read_csv('teste.csv')
print("Amostra dos dados de teste:")
print(test_df.head())

# Armazenar os IDs para o arquivo de saída
ids = test_df['id']
# Preparar as features (removendo a coluna 'id')
X_test = test_df.drop(['id'], axis=1)
X_test_scaled = scaler.transform(X_test)

# 6. Gerar as previsões para o conjunto de teste
predictions = model.predict(X_test_scaled)

# 7. Criar o DataFrame de saída com as colunas "id" e "target"
output_df = pd.DataFrame({'id': ids, 'target': predictions})

# Verifica se o DataFrame possui o número correto de linhas (contando o cabeçalho)
print("Número de linhas no arquivo de saída (sem contar o cabeçalho):", output_df.shape[0])

# 8. Salvar o DataFrame em um arquivo CSV
output_df.to_csv('predictions.csv', index=False)
print("Arquivo 'predictions.csv' gerado com sucesso!")
