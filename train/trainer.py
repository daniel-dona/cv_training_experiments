import numpy as np
from tensorflow.keras.callbacks import TerminateOnNaN, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras_tqdm import TQDMNotebookCallback as OutdatedTQDMCallback
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import keras_tuner

class TQDMNotebookCallback(OutdatedTQDMCallback):
    def build_tqdm_inner(self, *args, **kwargs):
        self.inner_description_update = self.inner_description_initial
        return super().build_tqdm_inner(*args, **kwargs)

    def append_logs(self, logs):
        return

    def format_metrics(self, logs):
        return


class Trainer:
    
    def __init__(self, dataset, experiment):
        
        self.dataset = dataset
        self.experiment = experiment
        self.callbacks = []
        
    def default_callbacks(self):
        self.enable_checkpoints()
        #self.enable_early_stop()
        self.enable_tensorboard()
        self.enable_tqdm()
           
    def enable_tensorboard(self):

        tensorboard = TensorBoard(
            log_dir=f"{self.experiment.folder}/logs",
            histogram_freq=0,
            write_graph=True,
            write_images=False,
            write_steps_per_second=True,
            update_freq="epoch",
            profile_batch=0,
            embeddings_freq=0,
            embeddings_metadata=None,
        )
        
        self.callbacks.append(tensorboard)
        
    def enable_tqdm(self):
        tqdm_callback = TQDMNotebookCallback(leave_inner=False, leave_outer=True)
        self.callbacks.append(tqdm_callback)
        
    def enable_checkpoints(self, monitor="val_accuracy"):
        model_checkpoint = ModelCheckpoint(f"{self.experiment.folder}/model.keras", monitor=monitor, verbose=0, save_best_only=True)
        self.callbacks.append(model_checkpoint)
        
    def enable_early_stop(self, patience=50):
        early_stop = EarlyStopping('val_accuracy', patience=patience, verbose=1)
        self.callbacks.append(early_stop)
        
    def enable_lr_plateau(self, patience=25):
        reduce_lr = ReduceLROnPlateau('val_accuracy=25', factor=0.75, patience=patience, verbose=0)
        self.callbacks.append(reduce_lr)

    def tuner_train(self, tunable_model, max_trials=1000):

        self.tunable_model = tunable_model

        self.tuner = keras_tuner.GridSearch(
        self.tunable_model,
        objective='val_accuracy',
        max_trials=max_trials,
        directory="./tuner/",
        project_name=self.experiment.name)

        self.tuner.save_model = lambda x: None

        self.tuner.search_space_summary(extended=True)
        
        #best_model = tuner.get_best_models()[0]

        train_generator, train_steps = self.dataset.train_data(
            batch_size=self.experiment.parameters["batch_size"]
        )
        valid_generator, valid_steps = self.dataset.valid_data(
            batch_size=self.experiment.parameters["batch_size"]
        )

        results = self.tuner.search(train_generator,
            steps_per_epoch=train_steps,
            validation_data=valid_generator,
            validation_steps=valid_steps,
            epochs=self.experiment.parameters["epochs"],
            callbacks=self.callbacks,
            verbose=1)

        return results

    
        #x_train, y_train, epochs=200, validation_data=(x_val, y_val))

    def train(self):
        
        train_generator, train_steps = self.dataset.train_data(
            batch_size=self.experiment.parameters["batch_size"]
        )
        valid_generator, valid_steps = self.dataset.valid_data(
            batch_size=self.experiment.parameters["batch_size"]
        )
        
        self.train_result = self.experiment.model.fit(
            train_generator,
            steps_per_epoch=train_steps,
            validation_data=valid_generator,
            validation_steps=valid_steps,
            epochs=self.experiment.parameters["epochs"],
            callbacks=self.callbacks,
            verbose=0
        )
        
        best_idx = int(np.argmax(self.train_result.history['val_accuracy']))
        best_value = np.max(self.train_result.history['val_accuracy'])
        
        print('Best validation model: epoch ' + str(best_idx+1), ' - val_accuracy ' + str(best_value))

    def validate(self):
        
    
        def draw_confusion_matrix(cm, categories):
            # Draw confusion matrix
            fig = plt.figure(figsize=[6.4*pow(len(categories), 0.5), 4.8*pow(len(categories), 0.5)])
            ax = fig.add_subplot(111)
            cm = cm.astype('float') / np.maximum(cm.sum(axis=1)[:, np.newaxis], np.finfo(np.float64).eps)
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.get_cmap('Blues'))
            ax.figure.colorbar(im, ax=ax)
            ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=list(categories.values()), yticklabels=list(categories.values()), ylabel='Annotation', xlabel='Prediction')
            # Rotate the tick labels and set their alignment
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            # Loop over data dimensions and create text annotations
            thresh = cm.max() / 2.0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], '.2f'), ha="center", va="center", color="white" if cm[i, j] > thresh else "black", fontsize=int(20-pow(len(categories), 0.5)))
            fig.tight_layout()
            plt.show(fig)


        self.experiment.model.load_weights(f"{self.experiment.folder}/model.keras")
        y_true, y_pred = [], []
        for ann in self.anns_valid:
            # Load image
            for obj_pred in ann.objects:
                # Generate prediction
                warped_image = np.expand_dims(ann.img, 0)
                predictions = self.experiment.model.predict(warped_image, verbose=0)
                # Save prediction
                pred_category = list(self.dataset.categories.values())[np.argmax(predictions)]
                pred_score = np.max(predictions)
                y_true.append(obj_pred.category)
                y_pred.append(pred_category)


        # Compute the confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=list(self.dataset.categories.values()))
        draw_confusion_matrix(cm, self.dataset.categories)


        # Compute the accuracy
        correct_samples_class = np.diag(cm).astype(float)
        total_samples_class = np.sum(cm, axis=1).astype(float)
        total_predicts_class = np.sum(cm, axis=0).astype(float)
        print('Mean Accuracy: %.3f%%' % (np.sum(correct_samples_class) / np.sum(total_samples_class) * 100))
        acc = correct_samples_class / np.maximum(total_samples_class, np.finfo(np.float64).eps)
        print('Mean Recall: %.3f%%' % (acc.mean() * 100))
        acc = correct_samples_class / np.maximum(total_predicts_class, np.finfo(np.float64).eps)
        print('Mean Precision: %.3f%%' % (acc.mean() * 100))
        for idx in range(len(self.dataset.categories)):
            # True/False Positives (TP/FP) refer to the number of predicted positives that were correct/incorrect.
            # True/False Negatives (TN/FN) refer to the number of predicted negatives that were correct/incorrect.
            tp = cm[idx, idx]
            fp = sum(cm[:, idx]) - tp
            fn = sum(cm[idx, :]) - tp
            tn = sum(np.delete(sum(cm) - cm[idx, :], idx))
            # True Positive Rate: proportion of real positive cases that were correctly predicted as positive.
            recall = tp / np.maximum(tp+fn, np.finfo(np.float64).eps)
            # Precision: proportion of predicted positive cases that were truly real positives.
            precision = tp / np.maximum(tp+fp, np.finfo(np.float64).eps)
            # True Negative Rate: proportion of real negative cases that were correctly predicted as negative.
            specificity = tn / np.maximum(tn+fp, np.finfo(np.float64).eps)
            # Dice coefficient refers to two times the intersection of two sets divided by the sum of their areas.
            # Dice = 2 |A∩B| / (|A|+|B|) = 2 TP / (2 TP + FP + FN)
            f1_score = 2 * ((precision * recall) / np.maximum(precision+recall, np.finfo(np.float64).eps))
            print('> %s: Recall: %.3f%% Precision: %.3f%% Specificity: %.3f%% Dice: %.3f%%' % (list(categories.values())[idx], recall*100, precision*100, specificity*100, f1_score*100))

    
    def test(self):
        pass