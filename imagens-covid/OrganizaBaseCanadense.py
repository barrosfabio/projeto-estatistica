'''
SCRIPT PARA SEPARAR AS IMAGENS DA BASE DO CANADENSE
LINK: https://github.com/ieee8023/covid-chestxray-dataset

O que ele faz? Ele pega os metadados presente no .CSV disponibilizado na base e percorre linha por linha.
"Ignorando" as imagens de CT e apenas fazendo uma cópia delas para outro diretório.

** Estrutura do projeto **

- PROJETO PYTHON
|-- covid-chestray-dataset (pasta com as imagens do canadense)
|-- imagensOrganizadas (pasta que será gerada pelo script)
|   |----- (aqui dentro vão aparecer as imagens organizadas)
|-- IniciandoComOCSV.py (esse é o script)
|-- not_found_files.txt (arquivo que vai guardar imagens presentes no .csv que não existem na paasta)

Durante a copia das imagens, neste momento, é ignorado as imagens CT.
Todos os patógenos passam a ser escritos sem espaços (" ") e em formato Pascal.
"EntaoFicandoAssim" para evitar redundancias.

'''


import pandas as pd
import os.path
from os import path
import shutil


_IMAGES_ORIGINAL_PATH_ = "./covid-chestxray-dataset/images"
_IMAGES_FINAL_PATH_ = "./imagensOrganizadas"

df = pd.read_csv("./covid-chestxray-dataset/metadata.csv")



rows, cols = df.shape[0], df.shape[1]
print(f'Row = {rows}')
print(f'Cols = {cols}')

# Retira todas as colunas "inuteis"
df = df.drop(
    ['clinical_notes', 'license', 'url', 'doi', 'folder', 'location', 'date', 'lymphocyte_count', 'neutrophil_count',
     'leukocyte_count', 'pO2_saturation', 'temperature', 'extubated', 'needed_supplemental_O2', 'in_icu', 'went_icu',
     'intubation_present', 'intubated', 'survival', 'RT_PCR_positive', 'age', 'sex', 'other_notes'], axis=1)
# df = df.drop(['filename'], axis=1)


# Retira todas as imagens que nao sao RaioX
new_df = df[df['modality'] != "X-ray"]
df = df[df['modality'] == "X-ray"]

'''
print("XRay")
rows, cols = df.shape[0], df.shape[1]
print(f'Row = {rows}')
print(f'Cols = {cols}')

print("Not XRay")
rows, cols = new_df.shape[0], new_df.shape[1]
print(f'Row = {rows}')
print(f'Cols = {cols}')
'''


# Verifica se se um diretorio existe, e entao cria ele
def CreateNewFolder(folder):
    if not path.exists(str(folder)):
        os.makedirs(str(folder))
        #print("Pasta criada")
    #else:
        #print("Pasta já existe")

# Criar Diretorios
def CreateFolders():
    #Separa todos os virus existentes
    virusList = ['']
    for index, row in df.iterrows():
        v = row['finding']
        v = v.title().replace(" ", "")
        print(v)

        newOne = True
        for name in range(len(virusList)):
            if v == virusList[name]:
                #print('old --- ', v)
                newOne = False
                break

        if newOne:
            #print('NEW one ---', v)
            virusList.append(v)

    #Comeca a criacao dos dirtorios
    #CreateNewFolder(_IMAGES_FINAL_PATH_)

    for name in range(len(virusList)):
        newPath = _IMAGES_FINAL_PATH_ + "/" + virusList[name]
        CreateNewFolder(newPath)

# Copiar o arquivo e coloca em uma pasta
def File_To_Folders():
    notFoundCount = 0
    notFoundFiles = ['']
    for index, row in df.iterrows():

        #if row['finding'] == "Klebsiella":
        imageLocation = _IMAGES_ORIGINAL_PATH_ + "/" + row['filename']
        finalLocation = _IMAGES_FINAL_PATH_ + "/" + row['finding'].title().replace(" ", "")

        if path.exists(str(imageLocation)):
            shutil.copy2(imageLocation, finalLocation + "/" + row['filename'])
            print("." + str(index))
        else:
            #print("Not found id:" + str(row) + " - PATH: " + imageLocation)
            notFoundCount += 1
            notFoundFiles.append(imageLocation)


    with open('not_found_files.txt', 'w') as f:
        for item in notFoundFiles:
            f.write("%s\n" % item)

    print("End")

# -- Recomendavel que seja feito um de cada vez

CreateFolders()
File_To_Folders()




'''
Anotacoes

# Preview the first 5 lines of the loaded data
# print(df.head(3))

# Pegar o tipo de virus
# print(df[df['finding'] == 'COVID-19'])
# for file in df[df['finding'] == 'COVID-19']: #Pegas apenas as colunas :/



rows, cols = df.shape[0], df.shape[1]
print(f'Row = {rows}')
print(f'Cols = {cols}')

print(df.head(3))
#print(df.groupby('finding')['finding'].count())

'''