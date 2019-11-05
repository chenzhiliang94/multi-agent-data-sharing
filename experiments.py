import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

data_transformed = pd.read_csv("breast_cancer_after_transform.csv",index_col=False)
y = data_transformed["malignant"]
X = data_transformed.drop(columns=["malignant"])


def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin)*np.random.rand(n) + vmin

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
xs = X["trans_A"]
ys = X["trans_B"]
zs = X["trans_C"]
ax.scatter(xs, ys, zs, marker='^',c=y)

ax.set_xlabel('feature 1')
ax.set_ylabel('feature 2')
ax.set_zlabel('feature 3')

plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

print("dividing data into 3 players")
# divide into a few players
'''
auc_value = []
for sample_size in range(11,200):
    p_1 = X_train.sample(sample_size)
    p_1_y = y.loc[p_1.index]
    p_2 = X_train.sample(30)
    p_2_y = y.loc[p_2.index]
    p_3 = X_train.sample(30)
    p_3_y = y.loc[p_3.index]

    print ("obtaining parameters from all training data...")
    # fit logistic regression to all data to get good parameter estimate
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(random_state=0, penalty ='l2',multi_class='ovr').fit(p_1, p_1_y)
    parameters = (clf.coef_[0])
    y_pred = [x[1] for x in clf.predict_proba(X_test)]
    auc_value.append(roc_auc_score(y_test, y_pred))
plt.plot(range(11,200), auc_value)
plt.show()
exit()
'''
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0, penalty ='l1',multi_class='ovr').fit(X, y)
parameters = (clf.coef_[0])
def fisher_information(X,params):
    # get output prob
    outputs = (clf.predict_proba(X))
    prod = []
    for output in outputs:
        prod.append(output[0]*output[1])
    diagonal = np.diagflat(prod)

    # append a column of ones
    X["ones"] = np.array(len(X["trans_A"])*[1])
    print (np.trace(X.values.T @ diagonal @ X.values))
    return (np.trace(X.values.T @ diagonal @ X.values))

contribution_1 = []
contribution_2 = []
contribution_3 = []
for size in range(170,210):
    print (" ")
    sample_size = size
    print(size)
    outputs = (clf.predict_proba(X))

    y_proba = pd.Series([x[0] for x in outputs])
    y_boundary = y_proba[y_proba < 0.75]
    y_boundary = y_boundary[y_proba > 0.25]
    X_boundary = X.loc[y_boundary.index]

    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    xs = X_boundary["trans_A"]
    ys = X_boundary["trans_B"]
    zs = X_boundary["trans_C"]
    ax.scatter(xs, ys, zs, marker='^')
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    plt.show()
    '''

    p_1 = X_boundary.sample(sample_size, replace=True)
    p_1_y = y_boundary.loc[p_1.index]

    y_boundary = y_proba[y_proba < 0.95]
    y_boundary = y_boundary[y_proba > 0.15]
    X_boundary = X.loc[y_boundary.index]
    p_2 = X_boundary.sample(sample_size, replace=True)
    p_2_y = y_boundary.loc[p_2.index]

    p_3 = X.sample(sample_size, replace=True)
    p_3_y = y.loc[p_3.index]

    contribution_1.append(fisher_information(p_1,parameters))
    contribution_2.append(fisher_information(p_2,parameters))
    contribution_3.append(fisher_information(p_3,parameters))

plt.plot(range(170,210), contribution_1, label="player A")
plt.plot(range(170,210), contribution_2, label="player B")
plt.plot(range(170,210), contribution_3, label="player C")
plt.xlabel("number of data points sampled for each player")
plt.ylabel(r"$\phi$")
plt.legend()
plt.title(r"Contribution for each player derived from $\mathcal{I}(\theta)$")
plt.show()

exit()

df = pd.read_csv("Folds5x2_pp.csv")
y = df["PE"]
X = df.drop(columns=["PE"])
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)

def read_(csv_name, test_col):

    df = (pd.read_csv(csv_name,delimiter=','))
    y = df[test_col]
    X = df.drop(columns=[test_col,'id'])
    return X, y



X, y = read_("breast-cancer-wisconsin.txt", "dangerous")
X, X_test, y, y_test = train_test_split( X, y, test_size=0.7, random_state=42)
y=y.map({2:0,4:1})
X['f']=X['f'].replace({'?':1})
X['f'] = pd.to_numeric(X['f'])
print(X.dtypes)
'''
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
print(dummy_y)
'''
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.1, random_state=42)

def build_model():
  model = keras.Sequential([
    keras.layers.Dense(3, activation='linear', input_shape=[len(X_train.keys())],name="layer_one"),
    keras.layers.Dense(3, activation='relu',name="layer_two"),
    keras.layers.Dense(2,activation='softmax', name="layer_three")
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='sparse_categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
  return model

model = build_model()
model.summary()

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 150

history = model.fit(
  X_train, y_train,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])

def plot_history(history):
    hist = pd.DataFrame(history.history)

    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('crossentropy')
    plt.plot(hist['epoch'], hist['loss'],
           label='Train Error')
    plt.plot(hist['epoch'], hist['val_loss'],
           label = 'Val Error')

    plt.legend()
    plt.show()


plot_history(history)


print ("transforming data ...")
X, y = read_("breast-cancer-wisconsin.txt", "dangerous")
y=y.map({2:0,4:1})
X['f']=X['f'].replace({'?':1})
X['f'] = pd.to_numeric(X['f'])

layer_name = 'layer_two'
intermediate_layer_model = keras.models.Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(X)

transformed_data_X = pd.DataFrame(intermediate_output,columns=["trans_A","trans_B","trans_C"])
transformed_data_y = pd.DataFrame(y.values,columns=["malignant"])
data = pd.concat([transformed_data_X,transformed_data_y],axis=1)
data.to_csv("breast_cancer_after_transform.csv")
print ("dividing data:")
print(data.head(30))





