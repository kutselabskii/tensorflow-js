from loader import load_dataset

x_train, y_train, x_test, y_test = load_dataset()

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
