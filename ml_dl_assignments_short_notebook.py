# %% [markdown]
# Short & Simple Jupyter Notebook: 15 Assignments (ML / DL / NLP / RL)
# Each cell is concise; run cells sequentially. Designed for clarity and quick demonstration.

# %%
# --- Imports used across assignments ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# For scikit-learn models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# For image augmentation / CNN
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# NLP
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Gym (for RL). If not installed, the cell will error; user can install gym.
try:
    import gym
except Exception:
    gym = None

# %% [markdown]
# Assignment 1: Supervised ML with scikit-learn (Iris dataset) — preprocessing + Logistic Regression & Decision Tree

# %%
# Load Iris, preprocess, train, evaluate
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sc = StandardScaler().fit(X_train)
X_train_s = sc.transform(X_train)
X_test_s = sc.transform(X_test)

# Logistic Regression
lr = LogisticRegression(max_iter=200)
lr.fit(X_train_s, y_train)
y_pred_lr = lr.predict(X_test_s)
print('LR acc:', accuracy_score(y_test, y_pred_lr))
print('LR conf_matrix:\n', confusion_matrix(y_test, y_pred_lr))

# Decision Tree
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print('\nDT acc:', accuracy_score(y_test, y_pred_dt))
print('DT conf_matrix:\n', confusion_matrix(y_test, y_pred_dt))

# %% [markdown]
# Assignment 2: Basic feedforward ANN (Breast Cancer dataset) — visualize loss & accuracy

# %%
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data; y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
sc = StandardScaler().fit(X_train)
X_train = sc.transform(X_train); X_test = sc.transform(X_test)

model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
hist = model.fit(X_train, y_train, validation_split=0.2, epochs=20, batch_size=32, verbose=0)

# Plot
plt.figure(); plt.plot(hist.history['loss'], label='loss'); plt.plot(hist.history['val_loss'], label='val_loss'); plt.legend(); plt.title('Loss')
plt.show()
plt.figure(); plt.plot(hist.history['accuracy'], label='acc'); plt.plot(hist.history['val_accuracy'], label='val_acc'); plt.legend(); plt.title('Accuracy')
plt.show()
print('Test acc:', model.evaluate(X_test, y_test, verbose=0)[1])

# %% [markdown]
# Assignment 3: Compare optimizers (SGD, Adam, RMSprop) — small network on Breast Cancer

# %%
opts = ['sgd','adam','rmsprop']
histories = {}
for opt in opts:
    m = keras.models.clone_model(model)
    m.set_weights(model.get_weights())  # start from same init
    m.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    h = m.fit(X_train, y_train, validation_split=0.2, epochs=15, batch_size=32, verbose=0)
    histories[opt] = h
    print(opt, 'final val_acc', h.history['val_accuracy'][-1])

plt.figure()
for opt in opts:
    plt.plot(histories[opt].history['val_accuracy'], label=opt)
plt.title('Validation Accuracy by Optimizer'); plt.legend(); plt.show()

# %% [markdown]
# Assignment 4: Regularization — L1, L2, Dropout on the same ANN

# %%
from tensorflow.keras import regularizers

def make_model(reg=None, dropout=0.0):
    reg_layer = regularizers.l2(1e-3) if reg=='l2' else (regularizers.l1(1e-4) if reg=='l1' else None)
    m = keras.Sequential()
    m.add(layers.Input(shape=(X_train.shape[1],)))
    m.add(layers.Dense(64, activation='relu', kernel_regularizer=reg_layer))
    if dropout>0: m.add(layers.Dropout(dropout))
    m.add(layers.Dense(32, activation='relu', kernel_regularizer=reg_layer))
    if dropout>0: m.add(layers.Dropout(dropout))
    m.add(layers.Dense(1, activation='sigmoid'))
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return m

configs = [('none',0.0), ('l2',0.0), ('l1',0.0), ('dropout',0.3)]
for name, d in configs:
    reg = None if name=='none' or name=='dropout' else name
    m = make_model(reg=reg, dropout=d)
    h = m.fit(X_train, y_train, validation_split=0.2, epochs=15, batch_size=32, verbose=0)
    print(name, 'val_acc', h.history['val_accuracy'][-1])

# %% [markdown]
# Assignment 5: Hyperparameter tuning with GridSearchCV (for scikit-learn) and simple manual search for Keras

# %%
from sklearn.model_selection import GridSearchCV
param_grid = {'C':[0.01,0.1,1,10]}
gs = GridSearchCV(LogisticRegression(max_iter=300), param_grid, cv=3)
gs.fit(X_train_s, y_train)
print('Best LR params:', gs.best_params_, 'best score:', gs.best_score_)

# Simple manual tuning for learning rate & batch size on small ANN
best = (None, -1)
for lr in [1e-3, 1e-2]:
    for bs in [16,32]:
        m = keras.Sequential([layers.Input(X_train.shape[1]), layers.Dense(32, activation='relu'), layers.Dense(1,activation='sigmoid')])
        m.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
        h = m.fit(X_train, y_train, epochs=8, batch_size=bs, validation_split=0.2, verbose=0)
        val = h.history['val_accuracy'][-1]
        if val>best[1]: best = ((lr,bs), val)
print('Best (lr,bs):', best)

# %% [markdown]
# Assignment 6: Image augmentation with ImageDataGenerator (Fashion-MNIST quick demo)

# %%
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train = x_train[...,None]/255.0; x_test = x_test[...,None]/255.0

datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
train_gen = datagen.flow(x_train, y_train, batch_size=64)

# Simple conv model
conv = keras.Sequential([layers.Input(shape=(28,28,1)), layers.Conv2D(16,3,activation='relu'), layers.MaxPool2D(), layers.Flatten(), layers.Dense(64,activation='relu'), layers.Dense(10,activation='softmax')])
conv.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
conv.fit(train_gen, epochs=3, steps_per_epoch=100, validation_data=(x_test,y_test))

# %% [markdown]
# Assignment 7: Simple CNN using Conv2D & MaxPooling (Fashion-MNIST)

# %%
model_cnn = keras.Sequential([
    layers.Input((28,28,1)),
    layers.Conv2D(32,3,activation='relu'), layers.MaxPool2D(),
    layers.Conv2D(64,3,activation='relu'), layers.MaxPool2D(),
    layers.Flatten(), layers.Dense(64,activation='relu'), layers.Dense(10,activation='softmax')
])
model_cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_cnn.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1)
print('Test acc', model_cnn.evaluate(x_test, y_test, verbose=0)[1])

# %% [markdown]
# Assignment 8: Transfer Learning (VGG16) — fine-tune top layers for Cats vs Dogs-like task (conceptual short demo)

# %%
# NOTE: For real Cats vs Dogs you'd load image files. Here we show how to assemble the model.
base = keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(128,128,3))
base.trainable = False
x = layers.GlobalAveragePooling2D()(base.output)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(2, activation='softmax')(x)
tl_model = keras.Model(base.input, outputs)
tl_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print('Transfer model ready (freeze base).')

# %% [markdown]
# Assignment 9: U-Net (simple version) for semantic segmentation (small toy model)

# %%
def conv_block(x, filters):
    x = layers.Conv2D(filters,3, padding='same', activation='relu')(x)
    x = layers.Conv2D(filters,3, padding='same', activation='relu')(x)
    return x

inputs = layers.Input((128,128,1))
# Encoder
c1 = conv_block(inputs, 16); p1 = layers.MaxPool2D()(c1)
c2 = conv_block(p1, 32); p2 = layers.MaxPool2D()(c2)
# Bottleneck
b = conv_block(p2, 64)
# Decoder
u1 = layers.UpSampling2D()(b); concat1 = layers.Concatenate()([u1, c2]); c3 = conv_block(concat1, 32)
u2 = layers.UpSampling2D()(c3); concat2 = layers.Concatenate()([u2 if ("u2" in locals()) else u2, c1])
# note: to keep it short, build a minimal decoder
u2 = layers.UpSampling2D()(c3); concat2 = layers.Concatenate()([u2, c1]); c4 = conv_block(concat2, 16)
outputs = layers.Conv2D(1,1, activation='sigmoid')(c4)
unet = keras.Model(inputs, outputs)
unet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print('U-Net summary:')
unet.summary()

# %% [markdown]
# Assignment 10: Q-learning / DQN for CartPole (very short DQN sketch)

# %%
if gym is None:
    print('gym not installed; install gym to run RL example')
else:
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    # simple DQN network
    qnet = keras.Sequential([layers.Input(state_size), layers.Dense(24,activation='relu'), layers.Dense(24,activation='relu'), layers.Dense(action_size)])
    qnet.compile(optimizer='adam', loss='mse')
    print('DQN network ready (sketch). Implement replay buffer and training loop to train agent).')

# %% [markdown]
# Assignment 11: NLTK — tokenization, stemming, lemmatization

# %%
ps = PorterStemmer()
wl = WordNetLemmatizer()
text = "Cats are running faster than the other cats."
tokens = word_tokenize(text)
print('Tokens:', tokens)
print('Stemmed:', [ps.stem(t) for t in tokens])
print('Lemmatized:', [wl.lemmatize(t) for t in tokens])

# %% [markdown]
# Assignment 12: POS tagging with NLTK and spaCy (compare outputs)

# %%
import spacy
try:
    nlp = spacy.load('en_core_web_sm')
except Exception:
    # if model not available, download
    from spacy.cli import download
    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

sent = "Apple is looking at buying U.K. startup for $1 billion"
print('NLTK POS:', nltk.pos_tag(word_tokenize(sent)))
print('spaCy POS:', [(t.text, t.pos_) for t in nlp(sent)])

# %% [markdown]
# Assignment 13: Named Entity Recognition with spaCy

# %%
doc = nlp(sent)
print('Entities:', [(ent.text, ent.label_) for ent in doc.ents])

# %% [markdown]
# Assignment 14: Train a sentiment classifier (very small demo using a toy dataset)

# %%
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

texts = ['I love this movie', 'I hate this film', 'Best product ever', 'Worst purchase', 'I like it', 'Not good']
labels = [1,0,1,0,1,0]
cv = CountVectorizer()
X = cv.fit_transform(texts)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.33, random_state=42)
clf = MultinomialNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('Sentiment acc:', accuracy_score(y_test, y_pred))
print('Confusion:\n', confusion_matrix(y_test, y_pred))

# %% [markdown]
# Assignment 15: Integrated pipeline: tokenization -> POS -> NER -> sentiment

# %%
def pipeline(text):
    toks = word_tokenize(text)
    pos = nltk.pos_tag(toks)
    doc = nlp(text)
    ents = [(e.text, e.label_) for e in doc.ents]
    # Simple sentiment using trained NB classifier
    x = cv.transform([text])
    sent = clf.predict(x)[0]
    return {'tokens': toks, 'pos': pos, 'entities': ents, 'sentiment': ('positive' if sent==1 else 'negative')}

print(pipeline('Google acquired a small startup. I love this news'))

# %% [markdown]
# End of notebook — short & simple implementations. 
# Notes for running: install packages (scikit-learn, tensorflow, nltk, spacy, gym) if missing. 
# For heavy tasks (training on large datasets), increase epochs and resources accordingly.
