
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

import numpy as np

import NeuralRedis.datasetManager as dm
import NeuralRedis.redisManager as rm

class NeuralRedis:
    datasetManager = None  # Public
    redisManager = None  # Public
    neuralNet = None  # Public

    def __init__(self,  host='127.0.0.1', password='', db=0, port=6379, inChannel='NeuInCh', outChannel='NeuOutCh'):
        self.datasetManager = dm.DatasetManager()
        self.neuralNet = None  # The perceptron
        self.redisManager = rm.RedisManager(host, password, db, port, inChannel, outChannel)
        self.redisManager.connect()

    def getDatasetInputMatrix(self):
        return self.datasetManager.getInputMatrix()

    def getDatasetOutputMatrix(self):
        return self.datasetManager.getOutputMatrix()

    def getDatasetTotalMatrix(self):
        return self.datasetManager.getTotalMatrix()

    def getInputSpace(self):
        return self.datasetManager.getInputSpace()

    def getOutputSpace(self):
        return self.datasetManager.getOutputSpace()

    def getTotalSpace(self):
        return self.datasetManager.getTotalSpace()

    def printMatrix(self, matrix):
        self.datasetManager.printMatrix(matrix)

    def printInputMatrix(self):
        print('Dataset Input Matrix:')
        self.datasetManager.printMatrix(self.datasetManager.getInputMatrix())

    def printOutputMatrix(self):
        print('Dataset Output Matrix:')
        self.datasetManager.printMatrix(self.datasetManager.getOutputMatrix())

    def printTotalMatrix(self):
        print('Dataset Total Matrix:')
        self.datasetManager.printMatrix(self.datasetManager.getTotalMatrix())

    def getDatasetMatrixFromRedisQueue(self, queueName, stop=False, outputColumnsPositions=[], randomShuffle = False):
        redisList = self.redisManager.getRedisQueue(queueName=queueName)
        # print('RedisList: ', redisList)  # Test
        redisMatrix = self.redisManager.redis2DQueueToMatrix(redisList=redisList)
        # print('RedisMatrix: ', redisMatrix)  # Test
        dataset = np.asarray(redisMatrix)
        # print('Dataset: ', dataset)  # Test
        self.datasetManager.importDatasetFrom2DMatrix(dataset, stop=stop, outputColumnsPositions=outputColumnsPositions,
                                                      randomShuffle=randomShuffle)
        return self.datasetManager.getInputMatrix(), self.datasetManager.getOutputMatrix(), \
               self.datasetManager.getTotalMatrix(), self.datasetManager.getInputSpace(), \
               self.datasetManager.getOutputSpace(), self.datasetManager.getTotalSpace()

    def getDatasetMatrixFromTXT(self, txtPath, separator="", stop=False, elemToFloat=False, outputColumnsPositions=[],
                                randomShuffle = False):
        self.datasetManager.importDatasetFromTXT(txtPath=txtPath, separator=separator, stop=stop, elemToFloat=elemToFloat,
                                                 outputColumnsPositions=outputColumnsPositions, randomShuffle=randomShuffle)
        return self.datasetManager.getInputMatrix(), self.datasetManager.getOutputMatrix(), \
               self.datasetManager.getTotalMatrix(), self.datasetManager.getInputSpace(), \
               self.datasetManager.getOutputSpace(), self.datasetManager.getTotalSpace()

    def removeColumnsFromInputMatrix(self, columnsToDel):
        self.datasetManager.delColumnsFromInputMatrix(columnPositions=columnsToDel)
        return self.datasetManager.getInputMatrix(), self.datasetManager.getTotalMatrix(),\
               self.datasetManager.getInputSpace(), self.datasetManager.getTotalSpace()

    def waitForOldestInRedisQueue(self, redisQueueName, stop=None):
        '''
        Returns the last element of a Redis queue. If the queue is empty, it waits while the list gains an element.
        If the stop param is setted, it waits only for a number of cycles equal to stop param.
        :param redisQueueName: the name of the queue, a string.
        :param stop: the number of cycles we want to wait, an integer.
        :return: the oldest element of the Redis queue.
        '''
        assert isinstance(redisQueueName, str)
        query = False
        if stop is None:
            while not query:
                query = self.redisManager.getOldestFromRedisQueue(queueName=redisQueueName)
                print(query)
        else:
            assert isinstance(stop, int)
            while not query and stop > 0:
                query = self.redisManager.getOldestFromRedisQueue(queueName=redisQueueName)
                print(query)
                stop -=1
        return query

    def createEquivalenceDictFromSpace(self, space):
        eq, revEq = self.datasetManager.createSpaceEquilalenceArrayDictionary(spaceList=space)
        return eq, revEq

    def addToEquivalenceDict(self, dict, revDict, key, value):
        return self.datasetManager.addToEquivalenceDict(eqDict=dict, revEqDict=revDict, key=key, value=value)

    def convertMatrix(self, equivalences, matrix):
        converted = self.datasetManager.matrixSymbolsConversion(equivalenceDictionary=equivalences,
                                                                matrixToConvert=matrix, reverse=False)
        return converted

    def assignNeurons(self, matrix):
        '''
        Says how many neurons are needed in the input/output layer, given the input/output matrix.
        :param matrix: The input or the output matrix.
        :return: neurons: An integer.
        '''
        assert str(type(matrix)) == "<class 'numpy.ndarray'>" \
               or str(type(matrix)) == "<class 'numpy.matrixlib.defmatrix.matrix'>" or isinstance(matrix, list)
        assert len(matrix) > 0
        nestedList = matrix[0]
        neurons = 0
        for l in nestedList:
            try:
                if isinstance(l, str) or str(type(l)) == 'numpy.str_':
                    neurons += 1
                else:
                    neurons = neurons + len(l)
            except:
                neurons = 1
        return neurons




#TODO TAKE NET CONFIG OUT OF THERE, DO TRAINING METHOD
    def createNet(self, netModel, inputLayerType, inputNeurons, inputActivationFunc=None,
                  hiddenLayerNeurons=[], hiddenActivationFunc=[], hiddenLayerType=[],
                  outputLayerType=None, outputNeurons=1, outputActivationFunc=None):
        assert isinstance(netModel, str) # For example: Sequential
        assert isinstance(inputLayerType, str)   # For example: Dense
        assert isinstance(outputLayerType, str)
        assert isinstance(inputNeurons, int)
        assert isinstance(outputNeurons, int)
        assert isinstance(hiddenLayerNeurons, list)
        assert isinstance(hiddenActivationFunc, list)
        assert isinstance(hiddenLayerType, list)
        for l in hiddenLayerNeurons:
            assert isinstance(l, int)
        for l in hiddenActivationFunc:
            assert isinstance(l, str)
        for l in hiddenLayerType:
            assert isinstance(l, str)
        assert isinstance(inputActivationFunc, str)
        assert isinstance(outputActivationFunc, str)
        assert callable(getattr(models, netModel.split('(')[0], None))
        assert callable(getattr(layers, inputLayerType.split('(')[0], None))
        assert callable(getattr(layers, outputLayerType.split('(')[0], None))
        assert callable(getattr(activations, inputActivationFunc.split('(')[0], None))
        assert callable(getattr(activations, outputActivationFunc.split('(')[0], None))

        if '(' not in netModel:
            netModel = netModel + '()'
        net = eval('models.' + netModel)
        inputL = eval(inputLayerType.split('(')[0] + '(' + str(inputNeurons))
        net.add()

        lCount = 0


        net.add(De)
        net.add(Dense(input_dim=inputNeurons, activation='tanh', units=inputNeurons))
        net.add(Dense(activation='tanh', units=hiddenLayerNeurons))
        net.add(Dense(activation='tanh', units=outputNeurons))
        input = redimDatasetIn[0: totalLength - fold]
        output = redimDatasetOut[0: totalLength - fold]
        '''
        net is a 2 layer (including hidden layer and output layer) perceptron.
        hiddenLayerNeurons is the number of neurons we have in the internal layer and we change it to try different net
        configurations.
        If we have a 10-fold validation, the Piece of dataset we use for training is made of 9/10 of rows of the dataset.
        '''
        print('########## ' + str(foldDim) + '-fold validation of a 2 Layers Perceptron with ' +
              str(inputNeurons) + ' Neurons in the Input Layer (and in the Output Layer) and with '
              + str(hiddenLayerNeurons) +
              ' Neurons in the Hidden Layer (Training Epochs = ' + str(epochsNumber) + ') ##########')
        print('-- Training on the First ' + str(rowsToConsider - int(rowsToConsider / foldDim)) +
              ' Dataset Rows, that correspond to '
              + str(len(input)) + ' Rows (with the same fold dimension): it can take several minutes --')
        # sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)
        net.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy'])
        net.fit(input, output, epochs=epochsNumber, verbose=1, shuffle=False)
        # Epochs (the number of repetitions of the training) is set as epochsNumber.
        print('-- Simulating on the last ' + str(foldDim) + ' Rows of the Dataset, that correspond to ' + str(fold)
              + ' Rows (with the same fold dimension) : it can take several minutes --')
        inputTest = redimDatasetIn[-(fold + 1): -1]
        outputTest = redimDatasetOut[-(fold + 1): -1]
        score = net.evaluate(inputTest, outputTest, verbose=1)
        print(' ')
        print('Score: ' + str(score))
        prediction = net.predict(inputTest, verbose=1)
        print('Predictions:')
        d.printMatrix(prediction)
        reconvPred = d.netResultBackConversion(reverseEquivalenceDict, prediction)
        print('Reconverted Predictions:')
        print(reconvPred)


# ############################### TEST #################################
'''
nr = NeuralRedis()
nr.getDatasetMatrixFromRedisQueue('prova')

nr.waitForOldestInRedisQueue(redisQueueName='prova')
'''