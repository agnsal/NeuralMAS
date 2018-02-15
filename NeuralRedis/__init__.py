
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
import time

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

    def waitForOldestInRedisQueue(self, redisQueueName, stop=None, pauseSeconds=0):
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
                time.sleep(pauseSeconds)
        else:
            assert isinstance(stop, int)
            while not query and stop > 0:
                query = self.redisManager.getOldestFromRedisQueue(queueName=redisQueueName)
                print(query)
                stop -= 1
                time.sleep(pauseSeconds)
        return query

    def writeOnRedisQueue(self, redisQueueName, item):
        self.redisManager.addToRedisQueue(queueName=redisQueueName, item=item)

    def writeOnRedisChannel(self, item):
        self.redisManager.redisPublish(toPublish=item)

    def redis2DQueueToMatrix(self, redisList):
        return self.redisManager.redis2DQueueToMatrix(redisList=redisList)

    def redisQueueOfListsToMatrix(self, redisList):
        return self.redisManager.redisQueueOfListsToMatrix(redisList=redisList)

    def createEquivalenceDictFromSpace(self, space):
        eq, revEq = self.datasetManager.createSpaceEquilalenceArrayDictionary(spaceList=space)
        return eq, revEq

    def addToEquivalenceDict(self, dict, revDict, key, value):
        return self.datasetManager.addToEquivalenceDict(eqDict=dict, revEqDict=revDict, key=key, value=value)

    def convertMatrix(self, equivalences, matrix, reverse):
        converted = self.datasetManager.matrixSymbolsConversion(equivalenceDictionary=equivalences,
                                                                matrixToConvert=matrix, reverse=reverse)
        return converted

    def roundNetResult(self, netResult, minValue, bit0Value, middleValue, bit1Value, maxValue):
        return self.datasetManager.roundNetResult(netResult=netResult, minValue=minValue, bit0Value=bit0Value,
                                                  middleValue=middleValue, bit1Value=bit1Value, maxValue=maxValue)

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
                neurons += 1
        return neurons

    def reshape3DMatrixTo2DForNeuralNet(self, matrix):
        return self.datasetManager.reshape3DMatrixTo2DForNeuralNet(matrix=matrix)

    def shapeBack2DMatrixTo3DFromNeuralNet(self, matrix, elementsXColumnList):
        return self.datasetManager.shapeBack2DMatrixTo3DFromNeuralNet(matrix=matrix, elementsXColumnList=elementsXColumnList)



# ############################### TEST #################################
'''
nr = NeuralRedis()
nr.getDatasetMatrixFromRedisQueue('prova')

nr.waitForOldestInRedisQueue(redisQueueName='prova')
'''
