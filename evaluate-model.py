import os
#remove tensorflow warnings 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np 
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer

def plot_history(history, save_name):
	plt.plot(acc_h, label='train acc')
	plt.plot(acc_val, label='val acc')
	plt.plot(loss_h, label='train loss')
	plt.plot(loss_val, label='val loss')

	plt.title('Training Loss and Accuracy')
	plt.xlabel('epoch')
	plt.ylabel('Loss/Accuracy')

	plt.legend(loc='upper left')

	# plt.legend(['train', 'validation'])
	plt.savefig(save_name)
	plt.show()



def plot_cm(cm, save_name):
	sn.heatmap(cm/np.sum(cm), annot=False, cmap='Oranges') #font size

	plt.title("Confusion Matrix")
	plt.xlabel("Predicted labels")
	plt.ylabel("True labels")
	plt.savefig(save_name)
	plt.show()

def main():
	model_name = "trained_model20.h5"
	print("Loading testing data")
	#load X_test data
	with open("train_test_data/X_test.npy", "rb") as f:
		X_test = np.load(f)

	#load y_test data
	with open("train_test_data/y_test.npy", "rb") as f:
		y_test = np.load(f)

	print("X_test.shape: ", X_test.shape)
	print("y_test.shape: ", y_test.shape)
	print("Sucessfully loaded data")

	model = keras.models.load_model(model_name)
	if(not model):
		print("Couldn't load the model sucessfully :(")
		os.exit()

	print("Model sucessfully loaded :)")
	print(model.summary)

	print("Evaluating model")

	test_loss, test_accuracy = model.evaluate(X_test, y_test)
	print("Test Metrics")
	print("loss: %.4f  - accuracy: %.4f" %(test_loss, test_accuracy))

	y_prob = model.predict(X_test) 
	y_pred = y_prob.argmax(axis=-1)

	# Pre-process the labels - transform them in a one hot vector
	lb = LabelBinarizer()
	lb.fit(y_pred)
	y_test = lb.inverse_transform(y_test)

	print(classification_report(y_test, y_pred))
	cm = confusion_matrix(y_test, y_pred)
	print(cm)

	plot_name = model_name.split(".")[0] + ".png"
	plot_cm(cm, plot_name)

if __name__ == '__main__':
	main()