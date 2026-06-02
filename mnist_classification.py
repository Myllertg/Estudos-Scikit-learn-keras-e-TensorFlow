#%%

#Baixa o dataset mnist_784
from sklearn.datasets import fetch_openml

mnist = fetch_openml ('mnist_784', as_frame=  False)

X, y = mnist.data, mnist.target
#%%

#Aqui é desenhado uma das imagens existentes no dataset
import matplotlib.pyplot as plt

def plot_digit(image_data):
    image = image_data.reshape(28,28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")

some_digit = X[0]
plot_digit(some_digit)
plt.show()


# %%

#Separação de conjunto de Treino e Teste (60mil treino e 10 mil de teste)
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
# %%
#TESTE
#Realizando o testando numero 5
#Verdadeiro para todos 5, falso para o outros digitos

y_train_5 = (y_train == '5')  
y_test_5 = (y_test == '5')


# %%

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier (random_state=42)
sgd_clf.fit (X_train, y_train)



#%%
#Fazendo a avaliação do modelo atraves a da acuracia da cross validacion

from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf,X_train, y_train_5, cv=3, scoring="accuracy")

# %%

#Fazendo um DummyClassifier, pra provar que até ele consegue a mesma previsão ou proxima
from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier()
dummy_clf.fit(X_train, y_train_5)

print(any (dummy_clf.predict(X_train)))

cross_val_score(dummy_clf, X_train, y_train_5, cv=3, scoring= "accuracy")


# %%

#Matriz de Confusão, forma de avaliar a classificação
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix


y_train_pred = cross_val_predict (sgd_clf, X_train, y_train_5, cv=3)

cm= confusion_matrix(y_train_5, y_train_pred)

cm
# %%

#Precision e Recall
from sklearn.metrics import precision_score, recall_score

precision_score(y_train_5, y_train_pred)
recall_score(y_train_5, y_train_pred)

#%%

#Trabalhando o F1 Score, ele busca um equilibrio entre o Precision e Recall
from sklearn.metrics import f1_score

f1_score(y_train_5, y_train_pred)
# %%
# Ele Buscao Threshold, ele cria um score para cada julgamento

y_scores = sgd_clf.decision_function([some_digit])
y_scores
threshold = 0 # linha de corte
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred
# %%
threshold = 3000
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred



# %% 
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")
# %%
from sklearn.metrics import precision_recall_curve

#Grafico Recall X Precision
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
plt.figure(figsize=(8, 4))
plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
plt.vlines(threshold, 0, 1.0, "k", "dotted", label="threshold")
idx = (thresholds >= threshold).argmax()  
plt.plot(thresholds[idx], precisions[idx], "bo")
plt.plot(thresholds[idx], recalls[idx], "go")
plt.axis([-50000, 50000, 0, 1])
plt.grid()
plt.xlabel("Threshold")
plt.legend(loc="center right")


plt.show()
# %%
import matplotlib.patches as patches  

#Rurva Precision/Recall
plt.figure(figsize=(6, 5))  
plt.plot(recalls, precisions, linewidth=2, label="Precision/Recall curve")
plt.plot([recalls[idx], recalls[idx]], [0., precisions[idx]], "k:")
plt.plot([0.0, recalls[idx]], [precisions[idx], precisions[idx]], "k:")
plt.plot([recalls[idx]], [precisions[idx]], "ko",
         label="Point at threshold 3,000")
plt.gca().add_patch(patches.FancyArrowPatch(
    (0.79, 0.60), (0.61, 0.78),
    connectionstyle="arc3,rad=.2",
    arrowstyle="Simple, tail_width=1.5, head_width=8, head_length=10",
    color="#444444"))
plt.text(0.56, 0.62, "Higher\nthreshold", color="#333333")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.axis([0, 1, 0, 1])
plt.grid()
plt.legend(loc="lower left")


plt.show()
# %%

#Curva ROC

from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)


precisions, recalls, thresholds = precision_recall_curve(
    y_train_5,
    y_scores
)

idx_for_90_precision = (precisions[:-1] >= 0.90).argmax()
threshold_for_90_precision = thresholds[idx_for_90_precision]


idx_for_threshold_at_90 = (thresholds <= threshold_for_90_precision).argmax()
tpr_90, fpr_90 = tpr[idx_for_threshold_at_90], fpr[idx_for_threshold_at_90]

plt.figure(figsize=(6, 5))  
plt.plot(fpr, tpr, linewidth=2, label="ROC curve")
plt.plot([0, 1], [0, 1], 'k:', label="Random classifier's ROC curve")
plt.plot([fpr_90], [tpr_90], "ko", label="Threshold for 90% precision")
plt.gca().add_patch(patches.FancyArrowPatch(
    (0.20, 0.89), (0.07, 0.70),
    connectionstyle="arc3,rad=.4",
    arrowstyle="Simple, tail_width=1.5, head_width=8, head_length=10",
    color="#444444"))
plt.text(0.12, 0.71, "Higher\nthreshold", color="#333333")
plt.xlabel('False Positive Rate (Fall-Out)')
plt.ylabel('True Positive Rate (Recall)')
plt.grid()
plt.axis([0, 1, 0, 1])
plt.legend(loc="lower right", fontsize=13)


plt.show()
# %%
from sklearn.ensemble import RandomForestClassifier

#Testeando RandomForest

forest_clf= RandomForestClassifier(random_state=42)

y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                    method="predict_proba")

y_probas_forest[:2]

# Construção da Curva Precision x Recall do Random Forest
y_scores_forest = y_probas_forest[:, 1]
precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(
    y_train_5, y_scores_forest)


# %%

#Modelo SVM
from sklearn.svm import SVC

svm_clf = SVC(random_state=42)
svm_clf.fit(X_train[:2000], y_train[:2000])  

svm_clf.predict([some_digit])

# %%

#OneVsRest
from sklearn.multiclass import OneVsRestClassifier


ovr_clf = OneVsRestClassifier(SVC(random_state=42))
ovr_clf.fit(X_train[:2000], y_train[:2000])
# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype("float64"))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")

from sklearn.metrics import ConfusionMatrixDisplay

y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
plt.rc('font', size=9)  # extra code – make the text smaller
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred)
plt.show()
# %%
