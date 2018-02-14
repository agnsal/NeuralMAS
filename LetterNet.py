
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

    import numpy as np

    from keras.models import load_model
    from keras.models import Model
    from keras.layers import Dense, Input

    import NeuralRedis

    '''
    The following ones are configuration variables.
    You can change them to change the configuration.
    datasetPath is a string containing the path of the dataset txt file.
    outputColumnsPositions is the list containing the positions of the output columns.
    elementsXConvertedOutputs is the list containing the lengths of the output columns after symbols convertion.
    epochNumber is an integer containing the number of epochs (steps) we want the training to last.
    rowsForTraining is an integer in [1, infinite) containing the number of the dataset rows we want to use.
    hiddenLayerProportion is an integer in [1, infinite) containing the proportion of neurons of the Hidden Layer
        (given the Input and Output Layers neuron number) of the net.
        hiddenLayerNeurons = int((inputNeurons + outputNeurons)/2) * hiddenLayerProportion
    '''
    ###############################
    datasetPath = "Letter Dataset/Letter Dataset.txt"
    outputColumnsPositions = [0]
    elementsXConvertedOutputs = [26]
    epochsNumber = 100
    rowsForTraining = 19950
    hiddenLayerProportion = 100
    ################################

    nr = NeuralRedis.NeuralRedis()
    nr.datasetManager
    nr.getDatasetMatrixFromTXT(txtPath=datasetPath, separator=',', stop=False, elemToFloat=True,
                               outputColumnsPositions=outputColumnsPositions, randomShuffle=False)
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
    print('Hidden Layer Neurons:', [hiddenLneurons0, hiddenLneurons1, hiddenLneurons1, hiddenLneurons2])
    reshapedOutputMatrix = nr.reshape3DMatrixTo2DForNeuralNet(matrix=outputMatrix)
    print('Reshaped Output Matrix:')
    print(reshapedOutputMatrix[0:10])
    print('...')

    '''
    # PAY ATTENTION!!! THE FOLLOWING LINES OF CODE ARE NEEDED FOR NET CREATION AND TRAINING
    # COMMENT THIS CODE IF YOU IMPORT THE TRAINED NET FROM A FILE
    # Start defining the input tensor:
    inputLayer = Input((inputLneurons,))
    # create the layers and pass them the input layer to get the output layer:
    hiddenLayer0 = Dense(units=hiddenLneurons0, activation='relu')(inputLayer)
    hiddenLayer1 = Dense(units=hiddenLneurons1, activation='relu')(hiddenLayer0)
    hiddenLayer2 = Dense(units=hiddenLneurons1, activation='relu')(hiddenLayer1)
    hiddenLayer3 = Dense(units=hiddenLneurons2, activation='relu')(hiddenLayer2)
    outputLayer = Dense(outputLneurons, activation='softmax')(hiddenLayer3)
    # Define the model's start and end points :
    neuralNet = Model(inputLayer, outputLayer)
    neuralNet.compile(optimizer='Nadam', loss='binary_crossentropy', metrics=['accuracy'])
    neuralNet.fit(x=inputMatrix[0:rowsForTraining], y=reshapedOutputMatrix[0:rowsForTraining], epochs=epochsNumber,
                  verbose=1, shuffle=False)
    neuralNet.save(filepath='LetterNet.h5', overwrite=True, include_optimizer=True)
    '''

    neuralNet = load_model('LetterNet.h5')

    netLoss = neuralNet.evaluate(x=inputMatrix[rowsForTraining:-1], y=reshapedOutputMatrix[rowsForTraining:-1], verbose=1)
    print('Score (netLoss): ', netLoss)

    nr.printMatrix(nr.getDatasetTotalMatrix()[rowsForTraining:-1])
    prediction = neuralNet.predict(inputMatrix[rowsForTraining:-1], verbose=1)
    print('Prediction: ')
    print(prediction)
    prediction = nr.roundNetResult(prediction, minValue=-1, bit0Value=0, middleValue=0.5, bit1Value=1, maxValue=2)
    prediction = nr.shapeBack2DMatrixTo3DFromNeuralNet(matrix=prediction,
                                                       elementsXColumnList=elementsXConvertedOutputs)
    prediction = nr.convertMatrix(equivalences=outputRevEq, matrix=prediction, reverse=True)
    print('Converted Prediction: ')
    print(prediction)



if __name__ == '__main__':
    '''
        Numpy is needed.
        To install Keras see https://keras.io/.
        It uses TensorFlow.
        h5py required: sudo pip3 install h5py.
    '''
    main()
    
