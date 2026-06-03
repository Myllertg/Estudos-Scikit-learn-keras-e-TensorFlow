# Estudos-Scikit-learn-keras-e-TensorFlow
# Classificação de Dígitos Manuscritos com MNIST

## 📖 Sobre o Projeto

Este projeto foi desenvolvido como estudo prático de Machine Learning utilizando o dataset **MNIST**, um dos conjuntos de dados mais conhecidos da área de Inteligência Artificial.

O objetivo é explorar conceitos fundamentais de classificação supervisionada utilizando a biblioteca Scikit-Learn, incluindo avaliação de modelos, métricas de desempenho, ajuste de threshold e comparação entre diferentes algoritmos.

---

## 🎯 Objetivos

Durante o projeto foram abordados os seguintes conceitos:

* Classificação Binária
* Classificação Multiclasse
* Validação Cruzada (Cross Validation)
* Matriz de Confusão
* Precision
* Recall
* F1-Score
* Threshold de Decisão
* Curva Precision-Recall
* Curva ROC
* Random Forest
* Support Vector Machine (SVM)
* One-vs-Rest (OvR)
* Padronização de Dados (StandardScaler)

---

## 📊 Dataset

Foi utilizado o dataset **MNIST (Modified National Institute of Standards and Technology)**.

Características do dataset:

* 70.000 imagens de dígitos manuscritos
* Imagens em escala de cinza
* Resolução de 28x28 pixels
* 10 classes (dígitos de 0 a 9)

Cada imagem é representada por:

* 784 atributos (28 × 28 pixels)

---

## 🛠️ Tecnologias Utilizadas

* Python
* NumPy
* Matplotlib
* Scikit-Learn

---

## 🚀 Etapas do Projeto

### 1. Carregamento dos Dados

Download do dataset MNIST utilizando a função `fetch_openml()`.

### 2. Visualização dos Dados

Exibição de imagens para compreender a estrutura do dataset.

### 3. Separação de Treino e Teste

* 60.000 imagens para treinamento
* 10.000 imagens para teste

### 4. Classificação Binária

Transformação do problema para identificar:

```text
É o dígito 5?
```

### 5. Treinamento com SGDClassifier

Treinamento de um classificador baseado em Stochastic Gradient Descent.

### 6. Avaliação com Cross Validation

Avaliação da acurácia utilizando validação cruzada com 3 folds.

### 7. Comparação com DummyClassifier

Utilização de um classificador simples como baseline para demonstrar as limitações da métrica Accuracy em datasets desbalanceados.

### 8. Métricas de Classificação

Análise das métricas:

* Precision
* Recall
* F1-Score

### 9. Threshold de Decisão

Estudo do impacto do threshold sobre Precision e Recall.

### 10. Curvas de Avaliação

Geração das curvas:

* Precision-Recall
* ROC (Receiver Operating Characteristic)

### 11. Comparação com Random Forest

Treinamento e avaliação de um modelo baseado em ensemble.

### 12. Classificação Multiclasse com SVM

Classificação direta dos dígitos de 0 a 9 utilizando Support Vector Machine.

### 13. Estratégia One-vs-Rest

Aplicação da abordagem One-vs-Rest para problemas multiclasse.

### 14. Padronização dos Dados

Utilização do StandardScaler para melhorar o desempenho do SGDClassifier.

### 15. Matriz de Confusão Multiclasse

Análise detalhada dos erros e acertos do modelo em todas as classes.

---

## 📈 Conceitos Aprendidos

Ao longo deste projeto foram praticados conceitos essenciais para problemas de classificação:

* Treinamento e avaliação de modelos
* Overfitting e generalização
* Métricas de desempenho
* Ajuste de limiares de decisão
* Comparação entre algoritmos
* Classificação binária e multiclasse
* Pré-processamento de dados

---

## ▶️ Como Executar

Clone o repositório:

```bash
git clone https://github.com/Myllertg/Estudos-Scikit-learn-keras-e-TensorFlow
```

Acesse a pasta:

```bash
cd Estudos-Scikit-learn-keras-e-TensorFlow
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute o script:

```bash
python mnist_classification.py
```

---

## 📚 Referência

Projeto desenvolvido com base nos conceitos apresentados no livro:

**Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow**
Autor: Aurélien Géron

---

## 👨‍💻 Autor

Myller T. G.

Projeto desenvolvido para estudo, prática e consolidação dos conceitos fundamentais de Machine Learning e Ciência de Dados.
