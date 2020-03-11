import os
import cv2
from tensorflow.keras import layers, models, callbacks
import random
import numpy as np
# from Baseline.tuning import *
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import logging

path_train = 'C:/Users/micha/Downloads/asl-alphabet/asl_alphabet_train/asl_alphabet_train/'
# path_test = 'C:/Users/micha/Downloads/asl-alphabet-test/'
path_test = 'C:/Users/micha/Downloads/asl-alphabet/asl_alphabet_test/asl_alphabet_test/'
folders = [f.name for f in os.scandir(path_train) if f.is_dir()]
logging.basicConfig(filename='baseline_output_set_29.log', level=logging.DEBUG)

#Create integer labels for each ASL character 0-28
label2id = {u: i for i, u in enumerate(folders)}

img_train = []
label_train = []
img_test = []
label_test = []
img_val = []
label_val = []

#Extract training set
x = 0
for sub in folders:
    for filename in os.listdir(path_train + sub):
        # if x == 10: break
        img = cv2.imread(os.path.join(path_train + sub, filename))
        img = cv2.resize(img, (100, 100))
        if img is not None: img_train.append(img)
        label_train.append(label2id[sub])
        x += 1
    x = 0

training = [(img, lab) for img, lab in zip(img_train, label_train)]
random.shuffle(training)
img_train, label_train = zip(*training)
img_train = list(img_train)
label_train = list(label_train)

#Training with 1/10 of dataset for speed
# img_train = np.asarray(img_train[:8700])/255.0
# label_train = np.asarray(label_train[:8700])
img_train = np.asarray(img_train)/255.0
label_train = np.asarray(label_train)

#Extract test set
# # x = 0
# for sub in folders:
#     for filename in os.listdir(path_test + sub):
#         # if x == 10: break
#         img = cv2.imread(os.path.join(path_test + sub, filename))
#         img = cv2.resize(img, (100, 100))
#         if img is not None: img_test.append(img)
#         label_test.append(label2id[sub])
#     #     x += 1
#     # x = 0

for filename in os.listdir(path_test):
    img = cv2.imread(os.path.join(path_test, filename))
    img = cv2.resize(img, (100, 100))
    if img is not None: img_test.append(img)
label_test = [x for x in range(29) if x != 4]

# #Shuffle testing data so that validation split will contain representative data
# testing = [(img, lab) for img, lab in zip(img_test, label_test)]
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

#Create model following referenced paper
model = models.Sequential()
model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(100, 100, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(64, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(128, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(256, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.Dense(256))
model.add(layers.Dense(29, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Train model and save weights during training so that we don't have to re-train each time
history = model.fit(img_train, label_train, epochs=3)
                    #validation_data=(img_val, label_val))


plt.figure(1)
plt.plot(range(3), history.history['loss'])
plt.title("Training Loss")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.savefig("train_loss.png")
plt.close(1)

plt.figure(2)
plt.plot(range(3), history.history['acc'])
plt.title("Training Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.savefig("train_acc.png")
plt.close(2)

predictions = model.predict(img_test)
predictions = [np.argmax(out) for out in predictions]
report = classification_report(label_test, predictions)
logging.debug(report)
logging.debug("\n\n\n\n\n")
matrix = confusion_matrix(label_test, predictions, range(29))
logging.debug(matrix)

print('finished')
# #Create model following referenced paper
# model = 0
# title = ""
# for i in range(5):
#     if i == 0:
#         model = model1()
#         title = "Model 1"
#     elif i == 1:
#         model = model2()
#         title = "Model 2"
#     elif i == 2:
#         model = model3()
#         title = "Model 3"
#     elif i == 3:
#         model = model4()
#         title = "Model 4"
#     elif i == 4:
#         model = model5()
#         title = "Model 5"
#
#     model.compile(optimizer='adam',
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])
#
#     #Train model
#     history = model.fit(img_train, label_train, epochs=3, validation_data=(img_val, label_val))
#     plt.figure(i+20)
#     plt.plot(range(3), history.history['val_loss'])
#     plt.title(title + " Loss")
#     plt.ylabel("Loss")
#     plt.xlabel("Epochs")
#     plt.savefig(title + " Loss" + ".png")
#     plt.close(i+20)
#
#     plt.figure(i+30)
#     plt.plot(range(3), history.history['val_acc'])
#     plt.title(title + " Accuracy")
#     plt.ylabel("Accuracy")
#     plt.xlabel("Epochs")
#     plt.savefig(title + " Accuracy" + ".png")
#     plt.close(i+30)




# #Evaluate model on test set and record accuracy
# test_loss, test_acc = model.evaluate(img_test, label_test, verbose=2)
# with open("eval.csv", 'w') as file:
#     file.write("Accuracy on Test Set: {}".format(test_acc))
# predictions = model.predict(img_test)
#
# #Record preictions labels to calculate confusion matrix
# with open('labels.csv', 'w') as file:
#     for output in predictions:
#         file.write(str(np.argmax(output)) + '\n')
