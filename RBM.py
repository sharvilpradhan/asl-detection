import numpy as np
from sklearn.metrics.classification import accuracy_score
from sklearn.neural_network import BernoulliRBM
from sklearn.svm import LinearSVC, SVC
import cv2
import os
import random
from sklearn.pipeline import Pipeline
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

def RBMtuning(rbm, n_comp):
    rbm.fit(img_train)
    img_train_latent = rbm.transform(img_train)
    img_val_latent = rbm.transform(img_val)

    model = models.Sequential()
    model.add(layers.Dense(64, input_shape=(n_comp,)))
    model.add(layers.Dense(29, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    #Train model and save weights during training so that we don't have to re-train each time
    filepath = "model_weights_saved.hdf5"
    checkpoint = callbacks.ModelCheckpoint(filepath, save_weights_only=True)
    desired_callbacks = [checkpoint]
    history = model.fit(img_train_latent, label_train, epochs=5,
                        validation_data=(img_val_latent, label_val), callbacks=desired_callbacks)
    return np.amin(history.history['val_loss']), np.amax(history.history['val_acc'])



path_train = 'C:/Users/micha/Downloads/asl-alphabet/asl_alphabet_train/asl_alphabet_train/'
# path_test = 'C:/Users/micha/Downloads/asl-alphabet-test/'
path_test = 'C:/Users/micha/Downloads/asl-alphabet/asl_alphabet_test/asl_alphabet_test/'
folders = [f.name for f in os.scandir(path_train) if f.is_dir()]
logging.basicConfig(filename='rbm_output_set_29.log', level=logging.DEBUG)

#Create integer labels for each ASL character 0-28
label2id = {u: i for i, u in enumerate(folders)}

img_train = []
label_train = []
img_test = []
label_test = []
img_val = []
label_val = []
img_plot = []
label_plot = []

#Extract training set
x = 0
for sub in folders:
    for filename in os.listdir(path_train + sub):
        # if x == 1: break
        img = cv2.imread(os.path.join(path_train + sub, filename))
        img = cv2.resize(img, (32, 32))
        if img is not None: img_train.append(img.flatten())
        if x == 0:
            img_plot.append(img.flatten())
            label_plot.append(sub)
        label_train.append(label2id[sub])
        x += 1
    x = 0

training = [(img, lab) for img, lab in zip(img_train, label_train)]
random.shuffle(training)
img_train, label_train = zip(*training)
img_train = list(img_train)
label_train = list(label_train)

#Training with 1/10 of dataset for speed
img_train = np.asarray(img_train[:1000])/255.0
label_train = np.asarray(label_train[:1000])
# img_train = np.asarray(img_train)/255.0
# label_train = np.asarray(label_train)

#Extract test set
# for sub in folders:
#     for filename in os.listdir(path_test + sub):
#         img = cv2.imread(os.path.join(path_test + sub, filename))
#         img = cv2.resize(img, (32, 32))
#         if img is not None: img_test.append(img.flatten())
#         label_test.append(label2id[sub])
for filename in os.listdir(path_test):
    img = cv2.imread(os.path.join(path_test, filename))
    img = cv2.resize(img, (32, 32))
    if img is not None: img_test.append(img.flatten())
label_test = [x for x in range(29) if x != 4]

#Shuffle testing data so that validation split will contain representative data
testing = [(img, lab) for img, lab in zip(img_test, label_test)]
# random.shuffle(testing)
# img_test, label_test = zip(*testing)
# img_test = list(img_test)
# label_test = list(label_test)
#
# #Split 50% testing 50% validation
# split = round(len(img_test) * 0.50)
# img_val = np.asarray(img_test[split:])/255.0
# label_val = np.asarray(label_test[split:])
# img_test = np.asarray(img_test[:split])/255.0
# label_test = np.asarray(label_test[:split])
img_test = np.asarray(img_test)/255.0
label_test = np.asarray(label_test)
#
#
# train_X_full = np.concatenate((img_train, img_val))
# train_Y_full = np.concatenate((label_train, label_val))
rbm = BernoulliRBM(n_components=200, n_iter=40,
                   learning_rate=0.01, verbose=True)
logistic = LogisticRegression(C=1.0)

# train the classifier and show an evaluation report
classifier = Pipeline([("rbm", rbm), ("logistic", logistic)])
# classifier.fit(train_X_full, train_Y_full)
classifier.fit(img_train, label_train)
predictions = classifier.predict(img_test)
report = classification_report(label_test, predictions)
logging.debug(report)
logging.debug("\n\n\n\n\n")
matrix = confusion_matrix(label_test, predictions, range(29))
logging.debug(matrix)

# #2 RBMs stacked 50 -> 29
# rbm1 = BernoulliRBM(n_components=50, learning_rate=0.1, n_iter=40)
# rbm1.fit(img_train)
# img_train_latent = rbm1.transform(img_train)
# rbm2 = BernoulliRBM(n_components=29, learning_rate=0.1, n_iter=20)
# img_train_latent = rbm2.fit_transform(img_train_latent)
#
# #Classification model training
# img_val_latent = rbm1.transform(img_val)
# img_val_latent = rbm2.transform(img_val_latent)
#
# model = models.Sequential()
# model.add(layers.Dense(64, input_shape=(29,)))
# model.add(layers.Dense(29, activation='softmax'))
#
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# #Train model and save weights during training so that we don't have to re-train each time
# filepath = "model_weights_saved.hdf5"
# checkpoint = callbacks.ModelCheckpoint(filepath, save_weights_only=True)
# desired_callbacks = [checkpoint]
# history = model.fit(img_train_latent, label_train, epochs=5,
#                     validation_data=(img_val_latent, label_val), callbacks=desired_callbacks)
#
# #Classification model testing
# img_test_latent = rbm1.transform(img_test)
# img_test_latent = rbm2.transform(img_test_latent)
# acc = model.evaluate(img_test_latent, label_test)
# logging.debug("Accuracy Score: " + str(acc))
#
# #One image of each ASL letter from training set to transform and plot for RBM visualization
# img_plot = np.asarray(img_plot)/255.0
# hidden = rbm1.transform(img_plot)
# rbm_class_labels = rbm2.transform(hidden)
# # logging.debug(rbm_class_labels)
#
# #Plot each ASL character to show RBM output
# os.mkdir('plots/')
# os.chdir('plots/')
# for i, char in enumerate(rbm_class_labels):
#     plt.figure(i)
#     plt.plot(char)
#     plt.title(label_plot[i])
#     plt.savefig(label_plot[i] + ".png")
#     plt.close(i)
# units = [50, 100, 200]
# iter = [20, 40, 80]
# learn_rate = [0.1, 0.01, 0.001]
# unit_cost = []
# unit_acc = []
# learn_cost = []
# learn_acc = []
# iter_cost = []
# iter_acc = []
# for u in units:
#     rbm = BernoulliRBM(n_components=u)
#     loss, acc = RBMtuning(rbm, u)
#     unit_cost.append(loss)
#     unit_acc.append(acc)
# opt_unit = units[np.argmin(unit_cost)]
#
# plt.figure(1)
# plt.plot(units, unit_acc)
# plt.title("RBM Units Tuning - Accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("# of Units")
# plt.savefig("units_acc.png")
#
# plt.figure(2)
# plt.plot(units, unit_cost)
# plt.title("RBM Units Tuning - Loss")
# plt.ylabel("Loss")
# plt.xlabel("# of Units")
# plt.savefig("units_loss.png")
#
#
# for l in learn_rate:
#     rbm = BernoulliRBM(n_components=opt_unit, learning_rate=l)
#     loss, acc = RBMtuning(rbm, opt_unit)
#     learn_cost.append(loss)
#     learn_acc.append(acc)
# opt_learn = learn_rate[np.argmin(learn_cost)]
#
# plt.figure(3)
# plt.semilogx(learn_rate, learn_acc)
# plt.title("RBM Learning Rate Tuning - Accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("Learning Rate")
# plt.savefig("learn_acc.png")
#
# plt.figure(4)
# plt.semilogx(learn_rate, learn_cost)
# plt.title("RBM Learning Rate Tuning - Loss")
# plt.ylabel("Loss")
# plt.xlabel("Learning Rate")
# plt.savefig("learn_loss.png")
#
# for i in iter:
#     rbm = BernoulliRBM(n_components=opt_unit, learning_rate=opt_learn, n_iter=i)
#     loss, acc = RBMtuning(rbm, opt_unit)
#     iter_cost.append(loss)
#     iter_acc.append(acc)
# opt_iter = iter[np.argmin(iter_cost)]
#
# plt.figure(5)
# plt.plot(iter, iter_acc)
# plt.title("RBM Iteration Tuning - Accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("# of Iterations")
# plt.savefig("iter_acc.png")
#
# plt.figure(6)
# plt.plot(iter, iter_cost)
# plt.title("RBM Iteration Tuning - Loss")
# plt.ylabel("Loss")
# plt.xlabel("# of Iterations")
# plt.savefig("iter_loss.png")

# #Write outputs to csv file
# num_classes = 5
# result = np.zeros(num_classes)
# for i in predictions:
#     new_row = np.zeros(num_classes)
#     new_row[i] = 1
#     result = np.vstack((result, new_row))
# result = np.delete(result, 0, axis=0)
# np.savetxt("labels_new.csv", result.astype(int), fmt='%i', delimiter=",")