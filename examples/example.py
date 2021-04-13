# DeepCASE imports
from deepcase                 import DeepCASE
from deepcase.preprocessing   import Preprocessor
from deepcase.context_builder import ContextBuilder
from deepcase.interpreter     import Interpreter

# Additional imports used in example
import numpy as np
import torch
from sklearn.metrics import classification_report
from deepcase.utils  import confusion_report

if __name__ == "__main__":
    ########################################################################
    #                              Load data                               #
    ########################################################################

    # Create preprocessor
    preprocessor = Preprocessor(
        context = 20,    # 10 events in context
        timeout = 86400, # Ignore events older than 1 day (60*60*24 = 86400 seconds)
    )

    # Load data from file
    events, context, label, mapping = preprocessor.csv('data/atlas.csv')

    # Events must be of shape (n_samples, 1)
    events = events.reshape(-1, 1)

    # In case no labels are provided, set labels to 0
    if label is None:
        label = torch.zeros(events.shape[0], dtype = torch.long)

    # Cast to cuda if available
    if torch.cuda.is_available():
        events  = events .to('cuda')
        context = context.to('cuda')
        label   = label  .to('cuda')

    ########################################################################
    #                        Split into Train/Test                         #
    ########################################################################

    # Split into train and test sets (20:80) by time - assuming events are ordered chronologically
    events_train  = events [:events.shape[0]//5 ]
    events_test   = events [ events.shape[0]//5:]

    context_train = context[:events.shape[0]//5 ]
    context_test  = context[ events.shape[0]//5:]

    label_train   = label  [:events.shape[0]//5 ]
    label_test    = label  [ events.shape[0]//5:]

    ########################################################################
    #                           Create DeepCASE                            #
    ########################################################################

    # Create DeepCASE object
    deepcase = DeepCASE(
        n_features  = 60,                                             # Set maximum number of expected events (60 is enough for the ATLAS dataset)
        complexity  = 128,                                            # Default complexity used in DeepCASE, dimension of hidden layer
        context     = 20,                                             # 10 events in context, same as in preprocessor
        device      = 'cuda' if torch.cuda.is_available() else 'cpu', # Or manually set 'cpu'/'cuda'
        eps         = 0.1,                                            # Default epsilon     used in DeepCASE, used for DBSCAN clustering
        min_samples = 5,                                              # Default min_samples used in DeepCASE, used for DBSCAN clustering
        threshold   = 0.2,                                            # Default threshold   used in DeepCASE, minimum required confidence
    )

    ########################################################################
    #                             Fit DeepCASE                             #
    ########################################################################

    # Fit ContextBuilder
    deepcase.context_builder.fit(
        X          = context_train,
        y          = events_train,
        batch_size = 128,           # Batch size you want to train with
        epochs     = 10,            # Number of epochs to train
        verbose    = True,          # If True, prints training progress
    )

    # Fit Interpreter
    deepcase.interpreter.fit(
        X          = context_train,
        y          = events_train,
        score      = label_train.unsqueeze(1),
        verbose    = True,
    )

    ########################################################################
    #                          Save/Load DeepCASE                          #
    ########################################################################

    # Save DeepCASE components
    deepcase.context_builder.save('context.save')     # Or specify a different filename
    deepcase.interpreter    .save('interpreter.save') # Or specify a different filename

    # Load DeepCASE components
    deepcase.context_builder.load(
        infile = 'context.save',                                 # File from which to load ContextBuilder
        device = 'cuda' if torch.cuda.is_available() else 'cpu', # Or manually set 'cpu'/'cuda'
    )

    deepcase.interpreter.load(
        infile          = 'interpreter.save',       # File from which to load Interpreter
        context_builder = deepcase.context_builder, # Used to link Interpreter to ContextBuilder. IMPORTANT: an Interpreter is specific to a ContextBuilder, so using a different ContextBuilder than used for training the Interpreter may yield bad results.
    )

    ########################################################################
    #               DeepCASE - Manual mode - Print evaluation              #
    ########################################################################

    # Get all clusters
    clusters = deepcase.interpreter.clusters # Get clusters

    # Count number of non-matched sequences
    non_matches = np.sum(clusters == -1)

    # Count unique clusters
    clusters_unique = set(np.unique(clusters).tolist()) - {-1}          # Get unique clusters without "non-cluster": -1

    # Print statistics
    print("\nManual mode - Statistics")
    print('━'*40)
    print("\tClusters: {}"               .format(len(clusters_unique)))
    print("\tCoverage: {}/{} sequences = {:8.4f}%\n".format(
        clusters.shape[0] - non_matches,
        clusters.shape[0],
        100 * (clusters.shape[0] - non_matches) / clusters.shape[0],
    ))

    ########################################################################
    #                    DeepCASE - Semi-automatic mode                    #
    ########################################################################

    # Use deepcase to predict labels
    label_predict = deepcase.interpreter.predict(
        X       = context_test,
        y       = events_test,
        verbose = True,
    )

    # The labels give an 'average' label, our original labels are 0 or 1, so we
    # compute the optimal threshold, in practice this will be set by an expert.
    label_predict[np.logical_and(label_predict >= 0, label_predict >  0)] = 1
    label_predict[np.logical_and(label_predict >= 0, label_predict <= 0)] = 0
    label_predict = label_predict.astype(int).reshape(-1)

    ########################################################################
    #           DeepCASE - Semi-automatic mode - Print evaluation          #
    ########################################################################

    # Print classification report
    print("\nClassification report")
    print('━'*40)
    print(classification_report(
        y_true        = label_test   [label_predict >= 0].cpu().numpy(),
        y_pred        = label_predict[label_predict >= 0],
        digits        = 4, # Specify required precision
        zero_division = 0,
    ), end='\n\n')

    # Get unique labels
    unique_labels = np.unique(
        [ -3,  -2,  -1] + np.unique(label_predict).tolist()
    ).tolist()

    # Print confusion matrix
    print("Confusion matrix")
    print('━'*40)
    print(confusion_report(
        y_true       = label_test.cpu().numpy(),
        y_pred       = label_predict,
        labels       = unique_labels,
        target_names = [
            'New cluster'   , # -3 means a new cluster was formed
            'Unknown label' , # -2 means an unknown label was found
            'Low confidence', # -1 means confidence for sample was too low
        ] + unique_labels[3:],
    ))
