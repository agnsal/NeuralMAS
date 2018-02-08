

'''
Copyright 2018 Agnese Salutari.
Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on 
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and limitations under the License
'''

__author__ = 'Agnese Salutari'


def main():
    '''
    To install Neurolab Python package see https://pythonhosted.org/neurolab/install.html
    (use pip and python for Python2 or pip3 and python3 for Python3).
    It uses Numpy and Scipy.
    To install Keras see https://keras.io/.
    It uses TensorFlow.
    '''

    import numpy as np

    from keras.models import Model
    from keras.layers import Dense, Input

    import NeuralRedis

    '''
    The following ones are configuration variables.
    You can change them to change the configuration.
    datasetPath is a string containing the path of the dataset txt file..
    foldDim is an integer containing the dimension of the fold.
    epochNumber is an integer containing the number of epochs (steps) we want the training to last.
    rowsToConsider is an integer in [1, infinite) containing the number of the dataset rows we want to use.
        Actual rows number is equal to rowsToConsider + rowsToConsider * questionVars if combined = False ,
        and it's equal to rowsToConsider * 2 otherwise.
    hiddenLayerProportion is an integer in [1, infinite) containing the proportion of neurons of the Hidden Layer
        (given the Input and Output Layers neuron number) of the net.
        hiddenLayerNeurons = int((inputNeurons + outputNeurons)/2) * hiddenLayerProportion
        If the Input has more that 1 neurons and the Output Layer has only 1 neuron:
            hiddenLayerProportion < 2 gives a "pyramid" structure;
                = 2 a rectangle;
                otherwise an armonica.
    '''
    ###############################
    datasetPath = "LetterDataset/LetterDataset.txt"
    epochsNumber = 500
    rowsForTraining = 19900
    hiddenLayerProportion = 100
    ################################

    nr = NeuralRedis.NeuralRedis()
    nr.datasetManager
    # nr.waitForOldestInRedisQueue(redisQueueName='prova')
    # nr.getDatasetMatrixFromRedisQueue(queueName='prova', stop=False, outputColumnsPositions=[], randomShuffle=False)
    # print(nr.datasetManager.getInputMatrix())
    nr.getDatasetMatrixFromTXT(txtPath=datasetPath, separator=',', stop=False, elemToFloat=True,
                               outputColumnsPositions=[0], randomShuffle=False)
    # nr.printTotalMatrix()  # Test
    # nr.printInputMatrix()  # Test
    # nr.printOutputMatrix()  # Test
    assert rowsForTraining <= len(nr.getDatasetTotalMatrix())
    print('Input Matrix Space:')
    print(nr.getInputSpace())
    print('Output Matrix Space:')
    print(nr.getOutputSpace())
    inputEq, inputRevEq = nr.createEquivalenceDictFromSpace(space=nr.getInputSpace())
    print('Input Matrix Equivalence and Reverse Equivalence:')
    print(inputEq)
    print(inputRevEq)
    inputMatrix = np.asarray(nr.convertMatrix(equivalences=inputEq, matrix=nr.getDatasetInputMatrix(), reverse=False))
    print('Converted Input Matrix:')  # Test
    print(inputMatrix)  # Test
    outputEq, outputRevEq = nr.createEquivalenceDictFromSpace(space=nr.getOutputSpace())
    print('Output Matrix Equivalence and Reverse Equivalence:')
    print(outputEq)
    print(outputRevEq)
    inputLneurons = nr.assignNeurons(matrix=inputMatrix)
    print('Input Layer Neurons: ', inputLneurons)
    outputMatrix = nr.convertMatrix(equivalences=outputEq, matrix=nr.getDatasetOutputMatrix(), reverse=False)
    outputLneurons = nr.assignNeurons(matrix=outputMatrix)
    print('Output Layer Neurons: ',  outputLneurons)
    hiddenLneurons0 = int((inputLneurons + outputLneurons) / 2) * hiddenLayerProportion
    hiddenLneurons1 = int(hiddenLneurons0/2)
    hiddenLneurons2 = int(hiddenLneurons1/2)
    hiddenLneurons3 = int(hiddenLneurons2/2)
    print('Hidden Layer Neurons:', [hiddenLneurons0, hiddenLneurons1, hiddenLneurons2, hiddenLneurons2, hiddenLneurons3])


    # Start defining the input tensor:
    inputLayer = Input((inputLneurons,))
    # create the layers and pass them the input layer to get the output layer:
    hiddenLayer0 = Dense(units=hiddenLneurons0, activation='relu')(inputLayer)
    hiddenLayer1 = Dense(units=hiddenLneurons1, activation='relu')(hiddenLayer0)
    hiddenLayer2 = Dense(units=hiddenLneurons2, activation='relu')(hiddenLayer1)
    hiddenLayer3 = Dense(units=hiddenLneurons2, activation='relu')(hiddenLayer2)
    hiddenLayer4 = Dense(units=hiddenLneurons3, activation='relu')(hiddenLayer3)
    outputLayer = Dense(outputLneurons, activation='relu')(hiddenLayer4)
    # Define the model's start and end points :
    neuralNet = Model(inputLayer, outputLayer)

    # weights = neuralNet.layers[0].get_weights()  # Test
    # print('The net is initialized with weights:')  # Test
    # print(weights)  # Test


    neuralNet.compile(optimizer='Nadam', loss='mean_absolute_percentage_error', metrics=['accuracy'])
    reshapedOutputMatrix = nr.reshape3DMatrixTo2DForNeuralNet(outputMatrix, outputLneurons)
    print('Reshaped Output Matrix:')
    print(reshapedOutputMatrix)
    neuralNet.fit(x=inputMatrix[0:rowsForTraining], y=reshapedOutputMatrix[0:rowsForTraining], epochs=epochsNumber,
                  verbose=1, shuffle=False)

    # weights = neuralNet.layers[0].get_weights()  # Test
    # print('The net is trained with weights:')  # Test
    # print(weights)  # Test

    netLoss = neuralNet.evaluate(x=inputMatrix[rowsForTraining:-1], y=reshapedOutputMatrix[rowsForTraining:-1], verbose=1)
    print('Score (netLoss): ', netLoss)

    print(nr.getDatasetTotalMatrix()[rowsForTraining:-1])
    prediction = neuralNet.predict(inputMatrix[rowsForTraining:-1], verbose=1)
    prediction = nr.roundNetResult(prediction, minValue=-2, bit0Value=-1, middleValue=0, bit1Value=1, maxValue=2)
    prediction = nr.shapeBack2DMatrixTo3DFromNeuralNet(matrix=prediction, numberOfColumns=outputLneurons)
    prediction = nr.convertMatrix(equivalences=outputRevEq, matrix=prediction, reverse=True)
    print('Prediction: ')
    print(prediction)



if __name__ == '__main__':
    '''
    To install Neurolab Python package see https://pythonhosted.org/neurolab/install.html
    (use pip and python for Python2 or pip3 and python3 for Python3).
    It uses Numpy and Scipy.
    '''
    main()
