
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
    '''
    ###############################
    datasetPath = "Dataset/BreastCancerDataset.txt"
    epochsNumber = 1000
    rowsForTraining = 630
    hiddenLayerProportion = 1000
    ################################

    nr = NeuralRedis.NeuralRedis()
    nr.datasetManager
    # nr.waitForOldestInRedisQueue(redisQueueName='prova')
    # nr.getDatasetMatrixFromRedisQueue(queueName='prova', stop=False, outputColumnsPositions=[], randomShuffle=False)
    # print(nr.datasetManager.getInputMatrix())
    nr.getDatasetMatrixFromTXT(txtPath=datasetPath, separator=',', stop=False, elemToFloat=True,
                               outputColumnsPositions=[10], randomShuffle=False)
    nr.removeColumnsFromInputMatrix([0])
    # nr.printTotalMatrix()  # Test
    # nr.printInputMatrix()  # Test
    # nr.printOutputMatrix()  # Test
    assert rowsForTraining <= len(nr.getDatasetTotalMatrix())
    print('Input Matrix Space:')
    print(nr.getInputSpace())
    print('Output Matrix Space:')
    print(nr.getOutputSpace())
    inputEq, inputRevEq = nr.addToEquivalenceDict(dict={}, revDict={}, key='?', value='100')  # 100 is a not defined value
    print('Input Matrix Equivalence and Reverse Equivalence:')
    print(inputEq)
    print(inputRevEq)
    convertedInputMatrix = np.asarray(nr.convertMatrix(equivalences=inputEq, matrix=nr.getDatasetInputMatrix()))
    inputMatrix = convertedInputMatrix
    # print('Converted Input Matrix:')  # Test
    # nr.printMatrix(convertedInputMatrix)  # Test
    outputEq, outputRevEq = nr.createEquivalenceDictFromSpace(space=nr.getOutputSpace())
    print('Output Matrix Equivalence and Reverse Equivalence:')
    print(outputEq)
    print(outputRevEq)
    inputLneurons = nr.assignNeurons(matrix=inputMatrix)
    print('Input Layer Neurons: ', inputLneurons)
    outputMatrix = nr.getDatasetOutputMatrix()
    outputLneurons = nr.assignNeurons(matrix=outputMatrix)
    print('Output Layer Neurons: ',  outputLneurons)
    hiddenLneurons = int((inputLneurons + outputLneurons) / 2) * hiddenLayerProportion
    print('Hidden Layer Neurons:', hiddenLneurons)

    # Start defining the input tensor:
    inputLayer = Input((inputLneurons,))
    # create the layers and pass them the input layer to get the output layer:
    hiddenLayer0 = Dense(units=hiddenLneurons, activation='relu')(inputLayer)
    hiddenLayer1 = Dense(units=int(hiddenLneurons/2), activation='relu')(hiddenLayer0)
    hiddenLayer2 = Dense(units=int(hiddenLneurons/4), activation='relu')(hiddenLayer1)
    outputLayer = Dense(units=outputLneurons, activation='relu')(hiddenLayer2)
    # Define the model's start and end points :
    neuralNet = Model(inputLayer, outputLayer)

    # weights = neuralNet.layers[0].get_weights()  # Test
    # print('The net is initialized with weights:')  # Test
    # print(weights)  # Test

    neuralNet.compile(optimizer='Nadam', loss='mean_absolute_percentage_error', metrics=['accuracy'])
    neuralNet.fit(x=inputMatrix[0:rowsForTraining], y=outputMatrix[0:rowsForTraining], epochs=epochsNumber,
                  verbose=1, shuffle=False)

    # weights = neuralNet.layers[0].get_weights()  # Test
    # print('The net is trained with weights:')  # Test
    # print(weights)  # Test

    netLoss = neuralNet.evaluate(x=inputMatrix[rowsForTraining:-1], y=outputMatrix[rowsForTraining:-1], verbose=1)
    print('Score (netLoss): ', netLoss)

    print(nr.getDatasetTotalMatrix()[rowsForTraining:-1])
    prediction = neuralNet.predict(inputMatrix[rowsForTraining:-1], verbose=1)
    print('Prediction: ', prediction)




if __name__ == '__main__':
    '''
    To install Neurolab Python package see https://pythonhosted.org/neurolab/install.html
    (use pip and python for Python2 or pip3 and python3 for Python3).
    It uses Numpy and Scipy.
    '''
    main()
