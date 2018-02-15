
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



import redis
import numpy as np # Test
import ast

class RedisManager:

    __host = None  # The default value is localhost
    __password = None
    __db = None  # This is the database we are using
    __port = None
    __inputChannel = None  # The channel we eventually want to listen to
    __outputChannel = None  # The channel we have to put outputs into
    __redis = redis  # Redis object

    def __init__(self, host='127.0.0.1', password='', db=0, port=6379, inChannel='NeuInCh', outChannel='NeuOutCh'):
        assert isinstance(host, str)
        assert isinstance(password, str)
        assert isinstance(db, int)
        assert isinstance(port, int)
        assert isinstance(inChannel, str)
        assert isinstance(outChannel, str)
        self.__host = host
        self.__password = password
        self.__db = db
        self.__port = port
        self.__inputChannel = inChannel
        self.__outputChannel = outChannel


    def getHost(self):
        return self.__host

    def setHost(self, newHost):
        assert isinstance(newHost, str)
        self.__host = newHost

    def getPassword(self):
        return self.__password

    def setPassword(self, newPassword):
        assert isinstance(newPassword, str)
        self.__password = newPassword

    def getDB(self):
        return self.__db

    def setDB(self, newDB):
        assert isinstance(newDB, int)
        self.__db = newDB

    def getPort(self):
        return self.__port

    def setPort(self, newPort):
        assert isinstance(newPort, int)
        self.__port = newPort

    def getInputChannel(self):
        return self.__inputChannel

    def setInputChannel(self, newInputChannel):
        assert isinstance(newInputChannel, str)
        self.__inputChannel = newInputChannel

    def getOutputChannel(self):
        return self.__outputChannel

    def setOutputChannel(self, newOutputChannel):
        assert isinstance(newOutputChannel, str)
        self.__outputChannel = newOutputChannel

    def connect(self):
        self.__redis = redis.Redis(host=self.__host, port=self.__port, db=self.__db, password=self.__password,
                                   charset="utf-8", decode_responses=True)
        if self.__redis.info():
            print('Successfully Connected to Redis.')
        else:
            print('Redis Connection Failed!')

    def redisPublish(self, toPublish):
        str(toPublish)
        self.__redis.publish(channel=self.__outputChannel, message=toPublish)

    def addToRedisQueue(self, queueName, item):
        assert isinstance(queueName, str)
        str(item)
        self.__redis.rpush(queueName, item)

    def countRedisQueueElements(self, queueName):
        '''
        Counts the elements inside a Redis queue.
        :param queueName: the name of the queue, a string.
        :return: length: the length of the queue, an integer.
        '''
        assert isinstance(queueName, str)
        length = self.__redis.llen(name=queueName)
        return length

    def getRedisQueue(self, queueName):
        '''
        Returns a Redis queue.
        :param queueName: the name of the Redis queue.
        :return: qList: the Redis queue.
        '''
        assert isinstance(queueName, str)
        length = self.countRedisQueueElements(queueName=queueName)
        qList = self.__redis.lrange(name=queueName, start=0, end=length)
        # print(qList)  # Test
        return qList

    def redis2DQueueToMatrix(self, redisList):
        '''
        Change a Redis 2D queue into a 2D matrix.
        :param redisList: the Redis queue.
        :return: result: a 2D matrix.
        '''
        assert isinstance(redisList, list) or str(type(redisList)) == "<class 'numpy.ndarray'>"
        result = []
        for row in redisList:
            # print('Row: ', row)  # Test
            rowList = str(row).replace(' ', '').split("],[")
            # print(rowList) # Test
            for elem in rowList:
                elemList = elem.replace('[', '').replace(']', ''). split(',')
                # print('ElemList: ', elemList)  # Test
                result.append(elemList)
        # print('res: ', result)  # Test
        return result

    def redisQueueOfListsToMatrix(self, redisList):
        '''
        Change a Redis queue of lists into a 2D matrix.
        :param redisList: the Redis queue.
        :return: result: a matrix of lists.
        '''
        assert isinstance(redisList, list)
        result = []
        for row in redisList:
            rowElem = list(ast.literal_eval(row))
            # print('RowElem: ', rowElem)  # Test
            result.append(rowElem)
        return result

    def getOldestFromRedisQueue(self, queueName):
        '''
        Returns the oldest element of a Redis queue.
        :param queueName: the name of the queue, a string.
        :return: the oldest element of the queue.
        '''
        assert isinstance(queueName, str)
        oldest = None
        if self.countRedisQueueElements(queueName=queueName) != 0:
            oldest = self.__redis.lpop(name=queueName)
        return oldest



# ####################### Test ############################
'''
    rpush prova [0,0],[0,1]
    rpush prova [1,0],[1,1]
'''
'''
redism = RedisManager()
redism.connect()
queue = redism.getRedisQueue('prova')
result = redism.redis2DQueueToMatrix(queue)
print(result)
print(len(result))
print(np.asarray(result))
print(len(np.asarray(result)))
print(redism.getOldestFromRedisQueue('prova'))
'''

