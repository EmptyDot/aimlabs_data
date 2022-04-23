import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import datetime


file = 'GridshotUltimate.csv'
df = pd.read_csv(file, index_col=False)

'''
print(df.agg({
    'accTotal': ['median', 'mean', 'max', 'min'],
    'killTotal': ['median', 'mean', 'max', 'min'],
    'rtTotal': ['median', 'mean', 'max', 'min'],
        }))

'''
print(df.head())

acc = ['accTotal']
rt = 'rtTotal'

df = df[df[rt] < 1000]




rev = df.sort_index(ascending=False)
idx = rev.index.values.reshape(len(rev.index), -1)
#times = rev['createDate'].values.reshape(-1, len(cr_time))

X = df[acc].values.reshape(-1, len(acc))
rts = rev[rt].values
accs = rev[acc].values


times = np.array([datetime.datetime.strptime(i, '%m/%d/%Y %H:%M:%S') for i in df['createDate'].sort_index(ascending=False)])
times = times.reshape(-1, len(times))


def relate_two(X, y, start, stop, xlab='x', ylab='y', prec=2):

    reg = LinearRegression()
    model = reg.fit(X, y)

    print(model.coef_)
    print(model.intercept_)

    x_pred = np.linspace(start, stop, (stop-start)*10**prec)
    x_pred = x_pred.reshape(-1, len(acc))

    y_pred = model.predict(x_pred)

    plt.style.use('default')
    plt.style.use('ggplot')

    fig, ax = plt.subplots(figsize=(7, 3.5))

    ax.plot(x_pred, y_pred, color='k', label='Regression model')
    ax.scatter(X, y, edgecolor='k', facecolor='grey', alpha=0.7, label='Sample data')
    ax.set_ylabel(ylab, fontsize=14)
    ax.set_xlabel(xlab, fontsize=14)
    ax.legend(facecolor='white', fontsize=11)
    ax.text(0.55, 0.25, '$y = %.2f x_1 - %.2f $' % (model.coef_[0], abs(model.intercept_)), fontsize=17, transform=ax.transAxes)

    fig.tight_layout()
    fig.show()






def heatmap(map_array, xticklabels=None, yticklabels=None, title=''):

    fig, ax = plt.subplots()
    im = ax.imshow(map_array)
    if xticklabels and yticklabels:
        # We want to show all ticks...
        ax.set_xticks(np.arange(len(xticklabels)))
        ax.set_yticks(np.arange(len(yticklabels)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(xticklabels)
        ax.set_yticklabels(yticklabels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    print(map_array.shape)
    for i in range(map_array.shape[0]):
        for j in range(map_array.shape[1]):
            text = ax.text(j, i, map_array[i, j],
                           ha="center", va="center", color="w")

    ax.set_title(title)
    fig.tight_layout()
    plt.show()



accmean = np.array(df[df.columns[7:15]].mean())

acc_mean = np.array([[accmean[7], accmean[3], accmean[0], accmean[4]],
                     [accmean[5], accmean[1], accmean[2], accmean[6]]]).round(2)

heatmap(acc_mean, title='Accuracy')
relate_two(idx, accs, 0, len(accs), xlab='Attempts', ylab='Accuracy(%)')
