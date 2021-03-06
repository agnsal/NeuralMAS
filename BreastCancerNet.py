
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
    foldDim is an integer containing the dimension of the fold.
    epochNumber is an integer containing the number of epochs (steps) we want the training to last.
    rowsForTraining is an integer in [1, infinite) containing the number of the dataset rows we want to use for training.
    hiddenLayerProportion is an integer in [1, infinite) containing the proportion of neurons of the Hidden Layer
        (given the Input and Output Layers neuron number) of the net.
        hiddenLayerNeurons = int((inputNeurons + outputNeurons)/2) * hiddenLayerProportion
    '''
    ###############################
    redisNetQueryList = 'neuralNetQueryList'
    redisNetSolutionList = 'neuralNetSolutionList'
    netOutChannel = 'LINDAchannel'
    netPauseSeconds = 1
    datasetPath = "Dataset/BreastCancerDataset.txt"
    outputColumnsPositions = [10]
    epochsNumber = 1000
    rowsForTraining = 650
    hiddenLayerProportion = 100
    ################################

    nr = NeuralRedis.NeuralRedis(outChannel=netOutChannel)
    nr.datasetManager
    # nr.waitForOldestInRedisQueue(redisQueueName='prova')
    # nr.getDatasetMatrixFromRedisQueue(queueName='prova', stop=False, outputColumnsPositions=[], randomShuffle=False)
    # print(nr.datasetManager.getInputMatrix())
    nr.getDatasetMatrixFromTXT(txtPath=datasetPath, separator=',', stop=False, elemToFloat=True,
                               outputColumnsPositions=outputColumnsPositions, randomShuffle=False)
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
    convertedInputMatrix = np.asarray(nr.convertMatrix(equivalences=inputEq, matrix=nr.getDatasetInputMatrix(), reverse=False))
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
    hiddenLneurons0 = int((inputLneurons + outputLneurons) / 2) * hiddenLayerProportion
    hiddenLneurons1 = int(hiddenLneurons0/2)
    hiddenLneurons2 = int(hiddenLneurons1/2)
    hiddenLneurons3 = int(hiddenLneurons2/2)
    print('Hidden Layer Neurons:', [hiddenLneurons0, hiddenLneurons1, hiddenLneurons2, hiddenLneurons2, hiddenLneurons3])

    '''
    # PAY ATTENTION!!! THE FOLLOWING LINES OF CODE ARE NEEDED FOR NET CREATION AND TRAINING
    # COMMENT THIS CODE IF YOU IMPORT THE TRAINED NET FROM A FILE
    # Start defining the input tensor:
    inputLayer = Input((inputLneurons,))
    # create the layers and pass them the input layer to get the output layer:
    hiddenLayer0 = Dense(units=hiddenLneurons0, activation='relu')(inputLayer)
    hiddenLayer1 = Dense(units=hiddenLneurons1, activation='relu')(hiddenLayer0)
    hiddenLayer2 = Dense(units=hiddenLneurons2, activation='relu')(hiddenLayer1)
    hiddenLayer3 = Dense(units=hiddenLneurons2, activation='relu')(hiddenLayer2)
    hiddenLayer4 = Dense(units=hiddenLneurons3, activation='relu')(hiddenLayer3)
    outputLayer = Dense(units=outputLneurons, activation='relu')(hiddenLayer4)
    # Define the model's start and end points :
    neuralNet = Model(inputLayer, outputLayer)
    neuralNet.compile(optimizer='Nadam', loss='mean_absolute_percentage_error', metrics=['accuracy'])
    neuralNet.fit(x=inputMatrix[0:rowsForTraining], y=outputMatrix[0:rowsForTraining], epochs=epochsNumber,
                  verbose=1, shuffle=False)
    neuralNet.save(filepath='BreastCancerNet.h5', overwrite=True, include_optimizer=True)
    '''

    neuralNet = load_model('BreastCancerNet.h5')
    netLoss = neuralNet.evaluate(x=inputMatrix[rowsForTraining:-1], y=outputMatrix[rowsForTraining:-1], verbose=1)
    print('Score (netLoss): ', netLoss)
    print(nr.getDatasetTotalMatrix()[rowsForTraining:-1])
    prediction = neuralNet.predict(inputMatrix[rowsForTraining:-1], verbose=1)
    prediction = nr.roundNetResult(prediction, minValue=0, bit0Value=2, middleValue=3, bit1Value=4, maxValue=6)
    print('Prediction: ')
    print(prediction)

    while True:
        firstRedisListElem = nr.waitForOldestInRedisQueue(redisQueueName=redisNetQueryList, stop=None,
                                                          pauseSeconds=netPauseSeconds)
        id = nr.redis2DQueueToMatrix([firstRedisListElem])[0][0]
        print('ID: ', id)
        query = [nr.redis2DQueueToMatrix([firstRedisListElem])[0][1:inputLneurons+1]]
        convertedQuery = np.asarray(nr.convertMatrix(equivalences=inputEq, matrix=query, reverse=False))
        print('Query for the Neural Net: ', convertedQuery)
        if len(query[0]) != inputLneurons:
            print('Invalid Query Format!')
        else:
            try:
                solution = neuralNet.predict(x=convertedQuery, verbose=1)
                print('Solution given by the Neural Net: ', solution)
                roundedSolution = nr.roundNetResult(solution, minValue=0, bit0Value=2, middleValue=3, bit1Value=4, maxValue=6)[0][0]
                if isinstance(roundedSolution, str) and 'OutOfRange' in roundedSolution:
                    roundedSolution = 0
                completeSolution = [id, 'breastcancer', roundedSolution]
                print('Rounded  Complete Solution: ', completeSolution)
                # nr.writeOnRedisQueue(redisQueueName=redisNetSolutionList, item=completeSolution)
                nr.writeOnRedisChannel(completeSolution)
            except:
                print('Solution not found')


if __name__ == '__main__':
    '''
        Numpy is needed.
        To install Keras see https://keras.io/.
        It uses TensorFlow.
        h5py required: sudo pip3 install h5py.
    '''
    main()
    
