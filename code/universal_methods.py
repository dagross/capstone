from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import load_img,img_to_array
from tensorflow.keras.models import Sequential, Model
import matplotlib.pyplot as plt
from numpy import expand_dims
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, plot_roc_curve,roc_curve
from sklearn.metrics import plot_confusion_matrix,ConfusionMatrixDisplay, confusion_matrix





class Universal:
    """
    Contains methods which are used through different docs.
    """
    def get_images(self, path='../data/real_and_fake_face/',size=600):
        """
        Gets images from folder, labels them and returns as X and y 
        """
        #load images and generate labels

        train_datagen = ImageDataGenerator(rescale = 1. / 255)
        itr = train_datagen.flow_from_directory(
        path,
        target_size = (size, size),
        batch_size = 15_000,
        class_mode = 'binary')
        X,y = itr.next()
       

        return X,y,list(itr.class_indices.keys())


    def get_images_multiclass(self, path='../data/Multiclass/',size=600):
        """
        Gets images from folder, labels them and returns as X and y 
        """
        #load images and generate labels

        train_datagen = ImageDataGenerator(rescale = 1. / 255)
        itr = train_datagen.flow_from_directory(
        path,
        target_size = (size, size),
        batch_size = 15_000,
        class_mode = 'categorical')
        X,y = itr.next()
      
        return X,y,list(itr.class_indices.keys())

    def get_feat(self,model,image,layer_num,savename="features"):
        """
        Recieves model and picture, generates plot with features of convolutional layer
        """
        # expand dimensions so that it represents a single 'sample'
        plt.tight_layout()
        img = expand_dims(image, axis=0)
        # check for convolutional layer
        if 'conv' not in model.layers[layer_num].name:
            return "No Convolutional layer"
        model_filter = Model(inputs=model.inputs, outputs=model.layers[layer_num].output)
        feature_maps = model_filter.predict(img)
	    # taking amount of rows for plot
        rows = model.layers[layer_num].output.shape[-1]
        ix = 1
        fig, axs = plt.subplots(nrows=1, ncols=rows, figsize=(10, 2))
        plt.subplots_adjust(hspace=0.1)
        fig.suptitle("Features of convolutional layer", fontsize=18, y=0.95)
        for _ in range(rows):
            # specify subplot and turn of axis
            axs[ix-1].imshow(feature_maps[0, :, :, ix-1])
            axs[ix-1].set_xticks([])
            axs[ix-1].set_yticks([])
            ix += 1
        plt.savefig(f'../resources/{savename}.jpg')
        # show the figure
        return axs

    def get_class(self,df,label,class_num):
        """
        Method to get dataset of images for chosen class 
        """
        plt.tight_layout()
        res=[]
        stop=0
        for x,y in zip(df,label):
            if stop==80:
             break
            if y==class_num:            
                res .append(x)
            stop+=1

        return np.array(res)

    def plot_samples(self,faces,amount,side,name,savename="samples_plot"): 
        """
        Method to plot images received as numpy array
        """
        plt.tight_layout()
        fig = plt.figure(figsize=(15, 15))
        for i in range(amount):
            plt.subplot(side, side, i+1)
            plt.imshow(faces[i])
            plt.suptitle(f"{name} images", fontsize=40)
            plt.axis('off')
        plt.savefig(f'../resources/{savename}.jpg')
        return fig

    def plot_proportions(self,y,name):
        plt.tight_layout()
        fake_class=np.count_nonzero(y==0)
        true_class=np.count_nonzero(y==1)
        plt.figure(figsize = (10, 6))
        plt.title(name)
        plt.xlabel('Observation count')

        plt.barh("True images",true_class,.5)
        plt.barh("Fake images",fake_class,.5)
        plt.savefig(f'../resources/{name}.jpg')



    def plot_results(self,history_,loss):
        plt.tight_layout()
        if (loss=="loss"):        
            train_loss = history_.history['loss']
            test_loss = history_.history['val_loss']
            label="Loss"
        elif (loss=="acc"):
            train_loss = history_.history['acc']
            test_loss = history_.history['val_acc']
            label="Accuracy"
        
        epoch_labels = history_.epoch

        # Set figure size.
        plt.figure(figsize=(12, 8))

        # Generate line plot of training, testing loss over epochs.
        plt.plot(train_loss, label=f'Training {label}', color='#185fad')
        plt.plot(test_loss, label=f'Testing {label}', color='orange')

        # Set title
        plt.title(f'Training and Testing {label} by Epoch', fontsize=25)
        plt.xlabel('Epoch', fontsize=18)
        plt.ylabel('Binary Crossentropy', fontsize=18)
        plt.xticks(epoch_labels, epoch_labels)    # ticks, labels

        plt.legend(fontsize=18);
   

    def plot_results_duo(self,history_,model,X_test,y_test,savename="model_result"):
               
        train_loss_l = history_.history['loss']
        test_loss_l = history_.history['val_loss']
        label_l="Loss"
        train_loss_a = history_.history['acc']
        test_loss_a = history_.history['val_acc']
        label_a="Accuracy"
        
        epoch_labels = history_.epoch

        # Set figure size.
        fig=plt.figure(figsize=(20,20))
        ax1 = plt.subplot2grid((8, 6), (0, 0), colspan=6,rowspan=2)
        ax2 = plt.subplot2grid((8, 6), (2, 0), colspan=6,rowspan=2)
        ax3 = plt.subplot2grid((8, 6), (4, 0), colspan=2,rowspan=2)

        ax5 = plt.subplot2grid((8, 6), (4, 3))
        ax6 = plt.subplot2grid((8, 6), (4, 4))
        ax7 = plt.subplot2grid((8, 6), (4, 5))

        ax9 = plt.subplot2grid((8, 6), (5, 3))
        ax10 = plt.subplot2grid((8, 6), (5, 4))
        ax11 = plt.subplot2grid((8, 6), (5, 5))
        ax12 = plt.subplot2grid((8, 6), (6, 2),colspan=4,rowspan=2)
        ax13 = plt.subplot2grid((8, 6), (6, 0),colspan=2,rowspan=2)
      
        # Generate line plot of training, testing loss over epochs.
        ax1.plot(train_loss_l, label=f'Training {label_l}', color='#185fad')
        ax1.plot(test_loss_l, label=f'Testing {label_l}', color='orange')
        ax2.plot(train_loss_a, label=f'Training {label_a}', color='#185fad')
        ax2.plot(test_loss_a, label=f'Testing {label_a}', color='orange')

        # Set title
        ax1.set_title(f'Training and Testing {label_l} by Epoch', fontsize=20)
        ax1.set_xlabel('Epoch', fontsize=18)
        ax1.set_ylabel('Binary Crossentropy', fontsize=18)
        ax1.set_xticks(epoch_labels, epoch_labels)    # ticks, labels
        ax1.legend(fontsize=18);
        
        ax2.set_title(f'Training and Testing {label_a} by Epoch', fontsize=20)
        ax2.set_xlabel('Epoch', fontsize=18)
        ax2.set_ylabel('Binary Crossentropy', fontsize=18)
        ax2.set_xticks(epoch_labels, epoch_labels)    # ticks, labels
        ax2.legend(fontsize=18);

        preds=model.predict(X_test)
        class_labels=["Fake", "True"]
        preds_labeled=(preds>0.5).astype(int)
        true_preds=y_test.astype(int)  

        matrix_trans= confusion_matrix(y_test.astype(int), (preds>0.5).astype(int))
        disp=ConfusionMatrixDisplay(matrix_trans,display_labels=class_labels)
        disp.plot(ax=ax3)

        mismatches = [i for i in range(0, len(preds)) if preds_labeled[i]==0 and true_preds[i]!=0]
        matches = [i for i in range(0, len(preds)) if preds_labeled[i]==0 and true_preds[i]==0]


        if len(matches)==0:
            ax4=plt.subplot2grid((8, 6), (4,3),colspan=4)
            ax4.text(0.3, 0.5, 'No correctly predicted fakes!', horizontalalignment='center', verticalalignment='center',fontsize=25)
            ax4.axis('off')

        else:
            ax4 = plt.subplot2grid((8, 6), (4, 2))
            ax4.imshow(X_test[matches[0]])
            ax4.axis('off')
            ax5.imshow(X_test[matches[1]])
            ax5.axis('off')
            ax5.set_title("Correctly predicted fakes",fontsize=25,x=1.3)
            ax6.imshow(X_test[matches[2]])
            ax6.axis('off')
            ax7.imshow(X_test[matches[3]])
            ax7.axis('off')
        if len(mismatches)==0:
            ax8=plt.subplot2grid((8, 6), (5,3),colspan=3)
            ax8.text(0.3, 0.5, 'No incorrectly predicted fakes!', horizontalalignment='center', verticalalignment='center',fontsize=25)
            ax8.axis('off')
        else:
            ax8 = plt.subplot2grid((8, 6), (5, 2))  
            ax8.imshow(X_test[mismatches[0]])
            ax8.axis('off')
            ax9.imshow(X_test[mismatches[1]])
            ax9.spines["bottom"].set_position(("axes", -0.15))
            ax9.axis('off')
            ax9.set_title("Incorrectly predicted fakes",fontsize=25,x=1.3)
            ax10.imshow(X_test[mismatches[2]])
            ax10.axis('off')
            ax11.imshow(X_test[mismatches[3]])
            ax11.axis('off')
        accuracy = accuracy_score(y_test, preds_labeled)
        # precision tp / (tp + fp)
        precision = precision_score(y_test, preds_labeled)
        # recall: tp / (tp + fn)
        recall = recall_score(y_test, preds_labeled)
        # f1: 2 tp / (2 tp + fp + fn)
        f1 = f1_score(y_test, preds_labeled)
        # ROC AUC
        auc = roc_auc_score(y_test, preds_labeled)
        fpr , tpr , thresholds = roc_curve ( y_test , preds_labeled)
      
        ax12.text(0.5, 0.5, f'F1 Score : {f1} \n Precision : {precision} \n Recall : {recall} \n AUC : {auc}', horizontalalignment='center', verticalalignment='center', fontsize=35)
        ax12.axis('off')
        ax13.plot(fpr,tpr)
        ax13.plot([-1, -1], [1, 1], 'red', linewidth=10)
        ax13.axis([0,1,0,1]) 
        ax13.set_xlabel('False Positive Rate') 
        ax13.set_ylabel('True Positive Rate')
        fig.tight_layout()
        plt.savefig(f'../resources/{savename}.jpg')


        return matches,mismatches,fig
        
    def get_filters(self,model, layer_num,savename="model_filter"):
      
        if 'conv' not in str(model.layers[layer_num]):
            return "Not a convalutional layer"
        filters, biases = model.layers[layer_num].get_weights()
        # normalize filter values to 0-1 so we can visualize them
        f_min, f_max = filters.min(), filters.max()
        filters = (filters - f_min) / (f_max - f_min)
        # plot first few filters
        n_filters, ix = 6, 1
        # plot each channel separately
        for i in range(n_filters):	
            # # get the filter
            f = filters[:, :, :, i]
            for j in range(2):
                ax = plt.subplot(n_filters, 3, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.imshow(f[:, :, 0], cmap='gray')
		        # plot filter channel in grayscale
                ix += 1
        # show the figure
        plt.savefig(f'../resources/{savename}.jpg')
        plt.show()




