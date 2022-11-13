# standard imports
import numpy as np
from numpy import expand_dims
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# CNN
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import load_img,img_to_array
from tensorflow.keras.models import Sequential, Model
import tensorflow as tf
# sklearn metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, plot_roc_curve,roc_curve
from sklearn.metrics import plot_confusion_matrix,ConfusionMatrixDisplay, confusion_matrix

# image libraries
from PIL import Image
import cv2
# finds pathnames matching a pattern to collect images
import glob
# For reproducibility
import random
seed = 42
np.random.seed(seed)
# random.seed(seed)
tf.random.set_seed(seed)


class Universal:
    """
    Contains methods which are used through different docs.
    """
    def get_images(self, path='../data/real_and_fake_face/',size=600):
        """
        Gets images from folder, labels them and returns as X and y 
        """
        # creating an instance of a generator, and automatically rescale it
        train_datagen = ImageDataGenerator(rescale = 1. / 255)
        #load images and generate labels
        itr = train_datagen.flow_from_directory(
        # source folder with images
        path,
        # converts images to given size
        target_size = (size, size),
        # number of images being processed
        batch_size = 15_000,
        # sets needed type of label array
        class_mode = 'binary')
        X,y = itr.next()
        return X,y,list(itr.class_indices.keys())

    def get_feat(self,model,image,layer_num,savename="features"):
        """
        Recieves model and picture, generates plot with features of convolutional layer
        """
        # rebuild this code https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/
        plt.tight_layout()
        # expand dimensions so that it represents a single 'sample'
        img = expand_dims(image, axis=0)
        # check for convolutional layer
        if 'conv' not in model.layers[layer_num].name:
            return "No Convolutional layer"
        #building a simple model with inputs and outputs of original one
        model_filter = Model(inputs=model.inputs, outputs=model.layers[layer_num].output)
        # obtain features
        feature_maps = model_filter.predict(img)
	    # taking amount of rows for plot
        rows = model.layers[layer_num].output.shape[-1]
        ix = 1        
        fig, axs = plt.subplots(nrows=1, ncols=rows, figsize=(16, 3))
        plt.subplots_adjust(hspace=0.1)
        fig.suptitle("Features of convolutional layer", fontsize=18, y=0.99)
        #adds resulting features to subplots
        for _ in range(rows):
            # specify subplot and turn of axis
            axs[ix-1].imshow(feature_maps[0, :, :, ix-1],aspect='equal')
            axs[ix-1].set_xticks([])
            axs[ix-1].set_yticks([])
            ix += 1
        # saves resulting plot
        plt.savefig(f'../resources/layers/{savename}.jpg',bbox_inches = 'tight')
        # show the figure
        return axs

    def get_class(self,df,label,class_num):
        """
        Method to get subset of images for chosen class 
        """
        plt.tight_layout()
        res=[]
        stop=0
        for x,y in zip(df,label):
            # setting limit of iterations to 80
            if stop==80:
             break
            #adding image to resulting array, if it matches given class
            if y==class_num:            
                res.append(x)
            stop+=1
            #converts result to numpy array before return
        return np.array(res)

    def generate_samples(self, amount_for_each_class=3):
        """
        Generates given number of samples for each class (easy/mid/hard) of fakes
        """
        image_list = []
        samples=np.array([])

        for filename in glob.glob('../data/real_and_fake_face/fake/*.jpg'): #looking for all .jpg files in the folder
            image_list.append(filename)
        # looks for all images with "easy" in the name
        matching = [s for s in image_list if "easy" in s]
        # adds them to results
        samples=np.append(samples,random.choices(matching,k=amount_for_each_class))
        # looks for all images with "mid" in the name
        matching = [s for s in image_list if "mid" in s]
        # adds them to results
        samples=np.append(samples,random.choices(matching,k=amount_for_each_class))
        # looks for all images with "hard" in the name
        matching = [s for s in image_list if "hard" in s]
        # adds them to results
        samples=np.append(samples,random.choices(matching,k=amount_for_each_class))
        #opens image and converts to numpy array before return
        samples=[np.asarray(Image.open(path)) for path in samples]
        return samples

    def plot_samples(self,faces,amount,side,name,savename="samples_plot"): 
        """
        Method to plot given number images received as numpy array 
        """
        plt.tight_layout()
        #creates figure, with fixed dimensions
        fig = plt.figure(figsize=(15, 15))
        #loops through needed amount of images and adds them to plot
        for i in range(amount):
            plt.subplot(side, side, i+1)
            plt.imshow(faces[i])
            # adds suptitle
            plt.suptitle(f"{name} images", fontsize=40)
            # turns off axis at the image
            plt.axis('off')
            #saves image
        plt.savefig(f'../resources/{savename}.jpg')
        return fig

    def plot_proportions(self,y,name):
        """
        Plots classes proportion of y dataset
        """
        plt.tight_layout()
        # counts fakes
        fake_class=np.count_nonzero(y==0)
        # counts true images
        true_class=np.count_nonzero(y==1)
        # creates figure with fixed dimensions
        plt.figure(figsize = (10, 6))
        # adds a given title
        plt.title(name)
        plt.xlabel('Observation count')
        # adds both classes to plot
        plt.barh("True images",true_class,.5)
        plt.barh("Fake images",fake_class,.5)
        # saves resulting plot
        plt.savefig(f'../resources/{name}.jpg')



    def plot_results_duo(self,history_,model,X_test,y_test,savename="model_result"):
        """
        United method that plots all necessary results of modeling
        """
        # gets array of loss and val_loss scores
        train_loss_l = history_.history['loss']
        test_loss_l = history_.history['val_loss']
        label_l="Loss"
        # gets array of accuracy and val_accuracy scores
        train_loss_a = history_.history['acc']
        test_loss_a = history_.history['val_acc']
        label_a="Accuracy"
        #gets array of epochs
        epoch_labels = history_.epoch
        # Set figure size
        fig=plt.figure(figsize=(20,20))
        # creating subplots through grid, setting col and row span
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
        # Set title of Train/Test loss plot
        ax1.set_title(f'Training and Testing {label_l} by Epoch', fontsize=20)
        ax1.set_xlabel('Epoch', fontsize=18)
        ax1.set_ylabel('Binary Crossentropy', fontsize=18)
        ax1.set_xticks(epoch_labels, epoch_labels)    # ticks, labels
        ax1.legend(fontsize=18);        
        # Set title of Train/Test accuracy plot
        ax2.set_title(f'Training and Testing {label_a} by Epoch', fontsize=20)
        ax2.set_xlabel('Epoch', fontsize=18)
        ax2.set_ylabel('Binary Crossentropy', fontsize=18)
        ax2.set_xticks(epoch_labels, epoch_labels)    # ticks, labels
        ax2.legend(fontsize=18);
        # generating predictions
        preds=model.predict(X_test)
        # list of classes names
        class_labels=["Fake", "True"]
        # converting class probabilities to classes
        preds_labeled=(preds>0.5).astype(int)
        # converting real labels to int, just in case
        true_preds=y_test.astype(int)  
        # building confusion matrix
        matrix_trans= confusion_matrix(y_test.astype(int), (preds>0.5).astype(int))
        disp=ConfusionMatrixDisplay(matrix_trans,display_labels=class_labels)
        # setting confusion matrix to ax at the plot 
        disp.plot(ax=ax3)
        # generating match and mismatches images num
        # incorrectly predicted  fakes
        mismatches_fakes = [i for i in range(0, len(preds)) if preds_labeled[i]==0 and true_preds[i]!=0]
        # incorrectly predicted real images
        mismatches_real = [i for i in range(0, len(preds)) if preds_labeled[i]==1 and true_preds[i]==0]
        # correctly predicted real fakes
        matches_fakes = [i for i in range(0, len(preds)) if preds_labeled[i]==0 and true_preds[i]==0]
        # setting exception for cases where there are no matches
        if len(matches_fakes)==0:
            # in that case subplot will contain only text phrase
            ax4=plt.subplot2grid((8, 6), (4,3),colspan=4)
            ax4.text(0.3, 0.5, 'No correctly predicted fakes!', horizontalalignment='center', verticalalignment='center',fontsize=25)
            ax4.axis('off')
        else:
            # if match contains values, then adds first 4 images for 4 subplots respectively
            ax4 = plt.subplot2grid((8, 6), (4, 2))
            ax4.imshow(X_test[matches_fakes[0]])
            # turns off axis
            ax4.axis('off')
            ax5.imshow(X_test[matches_fakes[1]])
            # turns off axis
            ax5.axis('off')
            # general title for all match subplots
            ax5.set_title("Correctly predicted fakes",fontsize=25,x=1.3)
            ax6.imshow(X_test[matches_fakes[2]])
            # turns off axis
            ax6.axis('off')
            ax7.imshow(X_test[matches_fakes[3]])
            # turns off axis
            ax7.axis('off')
        # setting exception for cases where there are no mismatches
        if len(mismatches_fakes)==0:
            # in that case subplot will contain only text phrase
            ax8=plt.subplot2grid((8, 6), (5,3),colspan=3)
            ax8.text(0.3, 0.5, 'No incorrectly predicted fakes!', horizontalalignment='center', verticalalignment='center',fontsize=25)
            ax8.axis('off')
        else:
            # if mismatch contains values, then adds first 4 images for 4 subplots respectively
            ax8 = plt.subplot2grid((8, 6), (5, 2))  
            ax8.imshow(X_test[mismatches_fakes[0]])
            # turns off axis
            ax8.axis('off')
            ax9.imshow(X_test[mismatches_fakes[1]])
            ax9.spines["bottom"].set_position(("axes", -0.15))
            # turns off axis
            ax9.axis('off')
            # general title for all mismatch subplots
            ax9.set_title("Incorrectly predicted fakes",fontsize=25,x=1.3)
            ax10.imshow(X_test[mismatches_fakes[2]])
            # turns off axis
            ax10.axis('off')
            ax11.imshow(X_test[mismatches_fakes[3]])
            # turns off axis
            ax11.axis('off')
        # calculates accuracy
        accuracy = accuracy_score(y_test, preds_labeled)
        # calculates precision tp / (tp + fp)
        precision = precision_score(y_test, preds_labeled)
        # calculates recall: tp / (tp + fn)
        recall = recall_score(y_test, preds_labeled)
        # calculates f1: 2 tp / (2 tp + fp + fn)
        f1 = f1_score(y_test, preds_labeled)
        # calculates ROC AUC
        auc = roc_auc_score(y_test, preds_labeled)
        fpr , tpr , thresholds = roc_curve ( y_test , preds_labeled)
        # adds previously predicted scores to one subplot 
        ax12.text(0.5, 0.5, f'F1 Score : {f1} \n Precision : {precision} \n Recall : {recall} \n AUC : {auc}', horizontalalignment='center', verticalalignment='center', fontsize=35)
        # turns off axis
        ax12.axis('off')
        ax13.plot(fpr,tpr)
        ax13.plot([-1, -1], [1, 1], 'red', linewidth=10)
        ax13.axis([0,1,0,1])
        # adds titles 
        ax13.set_xlabel('False Positive Rate') 
        ax13.set_ylabel('True Positive Rate')
        fig.tight_layout()
        #saves final plot with given name
        plt.savefig(f'../resources/{savename}.jpg',bbox_inches = 'tight')        
        return matches_fakes,mismatches_fakes,mismatches_real
        
    def get_filters(self,model, layer_num,savename="model_filter"):
        """
        Generates filters for the received model
        """
        # checks if there are convolution layers in the model
        if 'conv' not in str(model.layers[layer_num]):
            return "Not a convalutional layer"
        # obtaining weights from recieved model
        filters, biases = model.layers[layer_num].get_weights()
        # normalize filter values to 0-1 so we can visualize them
        f_min, f_max = filters.min(), filters.max()
        filters = (filters - f_min) / (f_max - f_min)
        # plot first few filters
        n_filters, ix = 6, 1
        # plot each channel separately
        for i in range(n_filters):	
            # get the filter
            f = filters[:, :, :, i]
            for j in range(2):
                ax = plt.subplot(n_filters, 3, ix)
                ax.set_xticks([])
                ax.set_yticks([])
		        # plot filter channel in grayscale
                plt.imshow(f[:, :, 0], cmap='gray')
                ix += 1
        # saves resulting plot
        plt.savefig(f'../resources/layers/{savename}.jpg')
        # show the figure
        plt.show()
   
    def get_conv(self,model,img_get_grad,model_name,num=-1,pred_index=None,alpha=0.6):
        """
        Generates Grad-Cam for the given image
        """
        # rebuild this code https://keras.io/examples/vision/grad_cam/
        model.layers[-1].activation = None
        img = img_get_grad
        # obtains list of all convolution layers and adds them to an array
        conv_layers=[]
        for layer in model.layers:
            if 'conv' in str(layer):
                conv_layers.append(layer.name)
        # creates simple model with identical to initial model inputs/outputs and only conv layers
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(conv_layers[num]).output, model.output]
        )
        img_array=np.expand_dims(img_get_grad, axis=0)
        # computes the gradient of the top predicted class for input image 
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
        # gets the gradient of the output neuron
        grads = tape.gradient(class_channel, last_conv_layer_output)
        # converts to vector
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        # obtains the heatmap of class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        # normalizes heatmap between 0 and 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap=heatmap.numpy()
        # rebuilding heatmap to uint8      
        heatmap =  cv2.normalize(heatmap, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8U)
        # colorizes heatmap with jet colormap
        jet = cm.get_cmap("jet")
        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        # creates an image with RGB colorized heatmap
        jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)
        # adds heatmap to original image
        superimposed_img = jet_heatmap * alpha + img
        superimposed_img = tf.keras.utils.array_to_img(superimposed_img)
        # drops axis
        plt.gca().set_axis_off()
        # removing white contour 
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
        plt.margins(0,0)
        # remove spines
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plots images
        plt.imshow(img_get_grad)
        plt.imshow(superimposed_img,alpha=alpha)
        # saves resulting image
        plt.savefig(f"../resources/grid_cam/{model_name}_final.jpg",bbox_inches = 'tight')








