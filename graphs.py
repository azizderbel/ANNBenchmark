import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as fg

def plotLoss(loss,label,epochs):
    n_iteration_per_epoch = 60000/100
    epochs_xaxis = []
    min = np.min(loss)
    #fig = fg.Figure(figsize=(10, 5))
    #fig,ax = plt.subplots(nrows=1,ncols=1,sharex=False, sharey=False)
    plt.title('Training loss',pad=20,fontsize=15)
    plt.xlabel('Iterations')
    plt.ylabel('Loss value')
    plt.subplots_adjust(top=0.8, bottom=0.2, left=0.1)
    plt.margins(0.2,0.1)
    plt.plot(loss,ls='--',label=label)
    plt.legend()
    for i in range(epochs * ((60000//100)//100)):
        if i % epochs == 0:
            plt.vlines(i,ymin=0,ymax=1,linestyles='dashdot',colors='red')


def plotAccuracy(dict):
    models_name = list(dict.keys())
    accuracy = list(dict.values())
    fig,ax = plt.subplots(nrows=1,ncols=1,sharex=False, sharey=False)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel('Models')
    ax.set_ylabel('Accuracy rate %')
    ax.bar(models_name,accuracy,width=0.3,align='center')
    title = plt.title('Model Accuracy', pad=20, fontsize=15)
    plt.subplots_adjust(top=0.8, bottom=0.2, left=0.1)
    ax.margins(0.2,0.1)
    for rect in ax.patches:
        # Get X and Y placement of label from rect
        y_value = rect.get_height()
        label = '{:.2f}'.format(y_value)
        x_value = rect.get_x() + rect.get_width() / 2
        
       
        #ax.grid(visible=True,which='major',axis='y')
        ax.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, 5),          # Vertically shift label by `space`
            textcoords='offset points', # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va='bottom')                      # Vertically align label differently for
                                        # positive and negative values




