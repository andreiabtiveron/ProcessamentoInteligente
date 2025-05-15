import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Exemplo de dados
data = {'Paisagem': ['Mata Escura', 'Mata Clara', 'Solo Exposto', 'Cidade']}
df = pd.DataFrame(data)

# Aplicando Label Encoding
label_encoder = LabelEncoder()
df['Paisagem_Codificada'] = label_encoder.fit_transform(df['Paisagem'])

print(df)