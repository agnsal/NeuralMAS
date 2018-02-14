
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


# To make this work, you need numpy
import numpy as np

class DatasetManager:
    '''
    The dataset class allows to manage the dataset.
    It has 2 attributes, __inputsMatrix and __outputMatrix, that are numpy.ndarray matrixes.
    __inputMatrix corresponds to the X matrix and __outputMatrix corresponds to the Y matrix.
    The other attribute, __TotalMatrx, is the union of the previous ones.
    The various __Space are the space of each matrix.
    '''
    __inputMatrix = []
    __outputMatrix = []
    __totalMatrix = []
    __inputSpace = []
    __outputSpace = []
    __totalSpace = []

    def setInputMatrix(self, newInputMatrix):
        '''
        Sets the __inputMatrix.
        :param newInputMatrix: a numpy.ndarray matrix
        :return:
        '''
        assert str(type(newInputMatrix)) == "<class 'numpy.ndarray'>"
        self.__inputMatrix = newInputMatrix

    def getInputMatrix(self):
        '''
        To read the matrix:
            matrix[:,0] gives as output the first column
            matrix[0] or matrix[0,:] give as output the first row
        :return: The input matrix (X), a numpy.ndarray matrix
        '''
        return self.__inputMatrix

    def setOutputMatrix(self, newOutputMatrix):
        '''
        Sets the __outputMatrix.
        :param newOutputMatrix: a numpy.ndarray mattrix
        :return:
        '''
        assert str(type(newOutputMatrix)) == "<class 'numpy.ndarray'>"
        self.__outputMatrix = newOutputMatrix

    def getOutputMatrix(self):
        '''
        To read the matrix:
            matrix[:,0] gives as output the first column
            matrix[0] or matrix[0,:] give as output the first row
        :return: The output matrix (Y), a numpy.ndarray matrix
        '''
        return self.__outputMatrix

    def setTotalMatrix(self, newTotalMatrix):
        '''
        Sets the __totalMatrix.
        :param newTotalMatrix: a numpy.ndarray mattrix
        :return:
        '''
        assert str(type(newTotalMatrix)) == "<class 'numpy.ndarray'>"
        self.__totalMatrix = newTotalMatrix

    def getTotalMatrix(self):
        '''
        To read the matrix:
            matrix[:,0] gives as output the first column
            matrix[0] or matrix[0,:] give as output the first row
        :return: The dataset matrix, a numpy.ndarray matrix
        '''
        return self.__totalMatrix

    def setInputSpace(self, newInputSpace):
        '''
        Sets the __inputSpace.
        :param newInputSpace: a list.
        :return:
        '''
        assert isinstance(newInputSpace, list)
        self.__inputSpace = newInputSpace

    def addToInputSpace(self, elem):
        '''
        Adds an element to the __inputSpace list.
        :param elem: the element to add.
        :return:
        '''
        self.__inputSpace.append(elem)

    def getInputSpace(self):
        return self.__inputSpace

    def setOutputSpace(self, newOutputSpace):
        '''
        Sets the __outputSpace.
        :param newOutputSpace: a list.
        :return:
        '''
        assert isinstance(newOutputSpace, list)
        self.__outputSpace = newOutputSpace

    def addToOutputSpace(self, elem):
        '''
        Adds an element to the __outputSpace list.
        :param elem: the element to add.
        :return:
        '''
        self.__outputSpace.append(elem)

    def getOutputSpace(self):
        return self.__outputSpace

    def setTotalSpace(self, newTotalSpace):
        '''
        Sets the __totalSpace.
        :param newTotalSpace: a list.
        :return:
        '''
        assert isinstance(newTotalSpace, list)
        self.__totalSpace = newTotalSpace

    def addToTotalSpace(self, elem):
        '''
        Adds an element to the __totalSpace list.
        :param elem:
        :return:
        '''
        self.__totalSpace.append(elem)

    def getTotalSpace(self):
        return self.__totalSpace

    def printMatrix(self, matrix):
        '''
        Prints the matrix.
        :param matrix: a numpy.ndarray, matrix or a list matrix.
        :return:
        '''
        assert str(type(matrix)) == "<class 'numpy.ndarray'>" \
               or str(type(matrix)) == "<class 'numpy.matrixlib.defmatrix.matrix'>" or isinstance(matrix, list)
        if matrix is not None and len(matrix) > 0:
            for row in matrix:
                print(row)
        else:
            print('Empty Matrix')

    def createSpace(self, matrix):
        '''
        Creates a space of values, given a matrix.
        :param matrix: a numpy.ndarray or a list matrix.
        :return: space: a sorted list.
        '''
        space = []
        # self.printMatrix(matrix)  # Test
        assert str(type(matrix)) == "<class 'numpy.ndarray'>" or isinstance(matrix, list)
        for row in matrix:
            # print('Row: ', row)  # Test
            for elem in row:
                # print('Elem: ', elem)  # Test
                if elem not in space:
                    # print('Not present')  # Test
                    space.append(elem)
        space.sort()
        return space

    def createSpaceEquivalenceDictionary(self, spaceList):
        '''
        Creates an equivalence dictionary given a list containing a space.
        :param spaceList: a list.
        :return: dictionary: a dictionary of equivalences.
        :return: reverseDictionary: the inverse of dictionary.
        '''
        assert isinstance(spaceList, list)
        dictionary = {}
        reverseDictionary = {}
        if len(spaceList) == 0:
            return dictionary, reverseDictionary
        else:
            counter = 0
            for elem in spaceList:
                dictionary[elem] = counter
                reverseDictionary[str(counter)] = str(elem)
                counter += 1
            return dictionary, reverseDictionary

    def createSpaceEquilalenceAllArrayDictionary(self, spaceList):
        '''
        Creates an equivalence array dictionary given a list containing a space.
        :param spaceList: a list.
        :return: dictionary: a dictionary of equivalences.
        :return: reverseDictionary: the inverse equivalence dictionary.
        '''
        assert isinstance(spaceList, list)
        dictionary = {}
        reverseDictionary = {}
        if len(spaceList) == 0:
            return dictionary, reverseDictionary
        else:
            counter = 0
            zeros = [0] * len(spaceList)
            for elem in spaceList:
                dictionary[elem] = zeros[:]
                dictionary[elem][counter] = 1
                vector = np.asarray(dictionary[elem])
                stringVector = str(vector).replace('\n', '')
                # print('String Vector:')  # Test
                # print(stringVector)  # Test
                reverseDictionary[stringVector] = elem
                counter += 1
            return dictionary, reverseDictionary

    def countSymbols(self, spaceList):
        '''
        Given a space (a list), returns the number of string elements (symbols) that are present in it.
        :param spaceList: a list.
        :return: count: an integer.
        '''
        assert isinstance(spaceList, list)
        count = 0
        for elem in spaceList:
            try:
                float(elem)
            except ValueError:
                count += 1
        return count

    def createSpaceEquilalenceArrayDictionary(self, spaceList):
        '''
        Creates an equivalence dictionary of arrays for string elements (symbols) only given a list containing a space.
        :param spaceList: a list.
        :return: dictionary: a dictionary of equivalences.
        :return: reverseDictionary: the inverse equivalence dictionary.
        '''
        assert isinstance(spaceList, list)
        dictionary = {}
        reverseDictionary = {}
        if len(spaceList) == 0:
            return dictionary, reverseDictionary
        else:
            counter = 0
            symbolNumber = self.countSymbols(spaceList)  # The number of symbols in the spaceList
            zeros = [0] * symbolNumber
            for elem in spaceList:
                try:
                    float(elem)
                except ValueError:
                    dictionary[elem] = zeros[:]
                    dictionary[elem][counter] = 1
                    vector = np.asarray(dictionary[elem])
                    stringVector = str(vector.tolist()).replace('\n', '')
                    # print('String Vector:')  # Test
                    # print(stringVector)  # Test
                    reverseDictionary[stringVector] = elem
                    counter += 1
            return dictionary, reverseDictionary

    def matrixStringify(self, matrix):
        '''
        Creates a matrix of string elements, given a matrix.
        :param matrix: a numpy.ndarray or a list matrix.
        :return: resultMatrix: a numpy.ndarray matrix.
        '''
        assert str(type(matrix)) == "<class 'numpy.ndarray'>" or isinstance(matrix, list)
        resultMatrix = []
        for row in matrix:
            resultMatrixRow = []
            for elem in row:
                resultMatrixRow.append(str(elem))
            resultMatrix.append(resultMatrixRow)
        return np.asarray(resultMatrix)

    def matrixSymbolsConversion(self, equivalenceDictionary, matrixToConvert, reverse=False):
        '''
        Converts a matrix in an equivalent numpy.ndarray one, given a dictionary of equivalences.
        :param equivalenceDictionary: A dictionary containing equivalences.
        :param matrixToConvert: The matrix to convert.
        :param reverse: A boolean saying if you are going to perform a back-conversion (using a reverse equivalence dictionary).
        :return: The matrix obtained by converting matrixToConvert, a numpy.ndarray matrix.
        '''
        assert isinstance(equivalenceDictionary, dict)
        assert len(matrixToConvert) > 0
        resultMatrix = []
        for row in matrixToConvert:
            resultRow = []
            elemCount = 0
            if reverse == False:
                for elem in row:
                    # print('ElemCount: ' + str(elemCount)) # Test
                    # print('Elem: ' + str(elem))  # Test
                    if elem in equivalenceDictionary:
                        equivalent = equivalenceDictionary[elem]
                        resultRow.append(equivalent)
                        # print('Equivalent to: ' + str(equivalent))  # Test
                    else:
                        resultRow.append(elem)
                    elemCount += 1
            else:
                for elem in row:
                    # print('ElemCount: ' + str(elemCount)) # Test
                    # print('Elem: ', str(elem))  # Test
                    if str(elem) in equivalenceDictionary:
                        equivalent = equivalenceDictionary[str(elem)]
                        resultRow.append(equivalent)
                        # print('Equivalent to: ' + str(equivalent))  # Test
                    else:
                        resultRow.append(elem)
                    elemCount += 1
            resultMatrix.append(resultRow)
            # print(resultMatrix)
        return resultMatrix

    def createAllVarTrainingsetInAndOut(self, dataset, questionVars=0, together=False):
        '''
        Expandes a dataset by making equal to 0 a variable at a time.
        :param dataset: a numpy.ndarray or a list matrix.
        :return: trainingsetIn: a np.ndarray matrix containing the inputs.
        :return: trainingsetOut: a np.ndarray matrix containing the outputs.
        '''
        assert str(type(dataset)) == "<class 'numpy.ndarray'>" or isinstance(dataset, list)
        assert isinstance(questionVars, int)
        assert questionVars >= 0
        assert questionVars <= len(dataset[0])
        assert isinstance(together, bool)
        assert questionVars < len(dataset[0]) or together == False
        if questionVars == 1:
            assert together == False
        if questionVars == len(dataset[0]):
            print('WARNING: The variables are NON linearly independent!')
        if isinstance(dataset, list):
            np.asarray(dataset)
        trainingsetIn = []
        trainingsetOut = []
        if questionVars == 0:
            trainingsetIn = dataset
            trainingsetOut = dataset
            return trainingsetIn, trainingsetOut
        if together == False:
            for row in dataset:
                trainingInRow = []
                inputCount = 0
                # print(row)  # Test
                trainingsetIn.append(row)
                trainingsetOut.append(row)  # Append a row for input without missing variable
                for input in row:
                    zeros = [0] * len(input)
                    if inputCount < questionVars:
                        # print('TEST')  # Test
                        trainingInRow = row
                        trainingInRow = np.delete(trainingInRow, inputCount, 0)
                        trainingInRow = np.insert(trainingInRow, inputCount, zeros, axis=0)
                        trainingsetOut.append(row)  # Append a row for input without missing variable
                        # print(trainingInRow)  # Test
                        trainingsetIn.append(trainingInRow)
                    inputCount += 1
                # self.printMatrix(trainingsetIn)  # Test
        else:
            for row in dataset:
                # print('Row:')  # Test
                # print(row)  # Test
                trainingsetIn.append(row)
                trainingsetOut.append(row)  # Append a row for input without missing variable
                trainingsetOut.append(row)  # Append a row for input with missing variables
                zeros = [0] * len(row[0])
                q = 0
                trainingInRow = row
                while q < questionVars:
                    # print(q)
                    trainingInRow = np.delete(trainingInRow, q, 0)
                    trainingInRow = np.insert(trainingInRow, q, zeros, axis=0)
                    q += 1
                # print('TrainingInRow:')  # Test
                # print(trainingInRow)  # Test
                trainingsetIn.append(trainingInRow)
        return np.asarray(trainingsetIn), np.asarray(trainingsetOut)

    def listOfListsMatrixToListMatrix(self, matrix):
        '''
        Converts a list of lists matrix into a list matrix by appending the nestes lists.
        :param matrix: a np.ndarray or a list matrix.
        :return: result: a np.ndarray matrix.
        '''
        assert str(type(matrix)) == "<class 'numpy.ndarray'>" or isinstance(matrix, list)
        result = []
        for row in matrix:
            unifiedList = []
            for list in row:
                for elem in list:
                    unifiedList.append(elem)
            result.append(unifiedList)
        return np.asarray(result)

    def writeMatrixOnFile(self, matrix, path):
        assert str(type(matrix)) == "<class 'numpy.ndarray'>" or isinstance(matrix, list)
        assert isinstance(path, str)
        with open(path, mode='wt', encoding='utf-8') as myfile:
            for line in matrix:
                myfile.write(str(line))
                myfile.write('\n')

    def convertMatrixElementsToFloat(self, matrix):
        convertedM = []
        for row in matrix:
            convertedRow = []
            for elem in row:
                # print('Elem: ', elem)  # Test
                try:
                    convElem = float(elem)
                except:
                    convElem = elem
                # print('Converted to: ', type(convElem))  # Test
                convertedRow.append(convElem)
            convertedM.append(convertedRow)
        return convertedM

    def reshape3DMatrixTo2DForNeuralNet(self, matrix):
        assert isinstance(matrix, list) or str(type(matrix)) == "<class 'numpy.ndarray'>"
        matrix2D = []
        for row in matrix:
            row2D = []
            for elem in row:
                if isinstance(elem, list) or str(type(elem)) == "<class 'numpy.ndarray'>":
                    row2D.extend(elem)
                else:
                    row2D.append(elem)
            matrix2D.append(row2D)
        return matrix2D

    def shapeBack2DMatrixTo3DFromNeuralNet(self, matrix, elementsXColumnList):
        assert isinstance(matrix, list) or str(type(matrix)) == "<class 'numpy.ndarray'>"
        assert isinstance(elementsXColumnList, list) or str(type(elementsXColumnList)) == "<class 'numpy.ndarray'>"
        assert len(elementsXColumnList) > 0
        reshapedMatrix = []
        for row in matrix:
            reshapedRow = [row[0:elementsXColumnList[0] - 1].tolist()]
            count = 1
            while count < len(elementsXColumnList):
                reshapedRow.append([row[(elementsXColumnList[count - 1]):(elementsXColumnList[count - 1] + elementsXColumnList[count])].tolist()])
                count += 1
            reshapedMatrix.append(reshapedRow)
        return reshapedMatrix

    def importDatasetFromTXT(self, txtPath, separator="", stop=False, elemToFloat=False, outputColumnsPositions=[], randomShuffle = False):
        '''
        Constructs a dataset (both inputs and outputs) by reading a txt file, and sets __totalMatrix,
        __inputMatrix, __outputMatrix __totalSpace, __inputSpace and __outputSpace.
        :param txtPath: a string containing the path of the txt file.
        :param separator: a string containing the separator between elements.
        :param stop: the lines we need: it is an integer or False if we want all the lines.
        :param elemToFloat: Boolean saying if you need elements to be floats (instead of strings).
        :param outputColumnsPositions: a list containing integers, that correspond to the positions
            of the columns of the output.
        :return:
        '''
        assert isinstance(txtPath, str)
        assert isinstance(outputColumnsPositions, list)
        txt = open(txtPath).read()
        if len(outputColumnsPositions) == 0:
            print('No output columns in the Dataset!')
            if not stop:
                inputMatrixRows = np.matrix([item.split(str(separator)) for item in txt.split('\n')[:-1]])
            else:
                assert isinstance(stop, int)
                inputMatrixRows = np.matrix([item.split(str(separator)) for item in txt.split('\n')[:stop - 1]])
            newInputMatrix = np.asarray(inputMatrixRows)
            if randomShuffle == True:
                np.random.shuffle(newInputMatrix)
            self.setInputMatrix(newInputMatrix)  # InputMatrix and TotalMatrix are the same
            self.setTotalMatrix(newInputMatrix)
            inputSpace = self.createSpace(newInputMatrix)
            self.setInputSpace(inputSpace)
            self.setTotalSpace(inputSpace)
        else:
            inputMatrixRows = []
            outputMatrixRows = []
            totalMatrixRows = []
            if not stop:
                text = txt.split('\n')[:-1]
            else:
                assert isinstance(stop, int)
                text = np.matrix([item.split(str(separator)) for item in txt.split('\n')[:stop - 1]])
            if randomShuffle == True:
                np.random.shuffle(text)
            for item in text:
                # print('Item: ',  item)  # Test
                totalMatrixRows.append(item.split(separator))
                inputRow = []
                outputRow = []
                inputPiece = item.split(separator)
                delcount = 0  # Deleted elements number
                for outputCol in outputColumnsPositions:
                    outputElem = item.split(separator)[outputCol]
                    if outputElem not in self.getOutputSpace():
                        self.addToOutputSpace(outputElem)
                    del inputPiece[outputCol - delcount]
                    inputRow = inputPiece
                    outputRow.append(outputElem)
                    delcount += 1
                outputMatrixRows.append(outputRow)
                inputMatrixRows.append(inputRow)
                # print('outputRow: ' + str(outputRow))  # Test
                # print('inputRow: ' + str(inputRow))  # Test
            if elemToFloat:
                self.setInputMatrix(np.asarray(self.convertMatrixElementsToFloat(inputMatrixRows)))
                self.setOutputMatrix(np.asarray(self.convertMatrixElementsToFloat(outputMatrixRows)))
                self.setTotalMatrix(np.asarray(self.convertMatrixElementsToFloat(totalMatrixRows)))
            else:
                self.setInputMatrix(np.asarray(inputMatrixRows))
                self.setOutputMatrix(np.asarray(outputMatrixRows))
                self.setTotalMatrix(np.asarray(totalMatrixRows))
            inputSpace = self.createSpace(inputMatrixRows)
            self.setInputSpace(inputSpace)
            outputSpace = self.createSpace(outputMatrixRows)
            self.setOutputSpace(outputSpace)
            totalSpace = self.createSpace(totalMatrixRows)
            self.setTotalSpace(totalSpace)

    def importDatasetFrom2DMatrix(self, matrixToImport, stop=False, elemToFloat=False, outputColumnsPositions=[], randomShuffle = False):
        '''
        Constructs a dataset (both inputs and outputs) by reading a list of lists matrix (2d matrix), and sets __totalMatrix,
        __inputMatrix, __outputMatrix __totalSpace, __inputSpace and __outputSpace.
        :param matrixToImport: a matrix containing all the dataset; it is a list of lists.
        :param stop: the lines we need: it is an integer or False if we want all the lines.
        :param elemToFloat: Boolean saying if you need elements to be floats (instead of strings).
        :param outputColumnsPositions: a list containing integers, that correspond to the positions
            of the columns of the output.
        :return:
        '''
        # print('MatrixToImport: ', matrixToImport)  # Test
        assert isinstance(matrixToImport, list) or str(type(matrixToImport)) == "<class 'numpy.ndarray'>"
        assert len(matrixToImport[0]) >= 0
        if elemToFloat:
            matrixToImport = np.asarray(self.convertMatrixElementsToFloat(matrixToImport))
        else:
            matrixToImport = np.asarray(matrixToImport)
        if len(outputColumnsPositions) == 0:
            print('No output columns in the Dataset!')
            if not stop:
                inputMatrixRows = matrixToImport
            else:
                assert isinstance(stop, int)
                inputMatrixRows = matrixToImport[:stop - 1]
            newInputMatrix = inputMatrixRows
            if randomShuffle == True:
                np.random.shuffle(newInputMatrix)
            self.setInputMatrix(newInputMatrix)  # InputSpace in equal to TotalSpace
            self.setTotalMatrix(newInputMatrix)
            inputSpace = self.createSpace(newInputMatrix)
            self.setInputSpace(inputSpace)
            self.setTotalSpace(inputSpace)
        else:
            if not stop:
                rows = matrixToImport
            else:
                assert isinstance(stop, int)
                rows = matrixToImport[:stop]
            if randomShuffle == True:
                np.random.shuffle(rows)
            # print('Rows: ', rows)  # Test
            outputMatrixRows = rows[:, outputColumnsPositions]
            inputMatrixRows = np.delete(rows, outputColumnsPositions, axis=1)
            # print('InputMatrixRows: ', inputMatrixRows)  # Test
            self.setInputMatrix(inputMatrixRows)
            self.setOutputMatrix(outputMatrixRows)
            self.setTotalMatrix(rows)
            inputSpace = self.createSpace(inputMatrixRows)
            self.setInputSpace(inputSpace)
            outputSpace = self.createSpace(outputMatrixRows)
            self.setOutputSpace(outputSpace)
            totalSpace = self.createSpace(rows)
            self.setTotalSpace(totalSpace)

    def importTotalDatasetFromTXT(self, txtPath, separator="", stop=False, randomShuffle = False):
        '''
        Constructs a single dataset matrix by reading a txt file, and sets __totalMatrix and __totalSpace.
        :param txtPath: a string containing the path of the txt file.
        :param separator: a string containing the separator between elements.
        :param stop: the lines we need: it is an integer or False if we want all the lines.
        :param randomShuffle: a boolean saying if you want to shuffle dataset rows.
        :return:
        '''
        assert isinstance(txtPath, str)
        txt = open(txtPath).read()
        if not stop:
            totalMatrixRows = np.matrix([item.split(str(separator)) for item in txt.split('\n')[:-1]])
        else:
            assert isinstance(stop, int)
            totalMatrixRows = np.matrix([item.split(str(separator)) for item in txt.split('\n')[:stop-1]])
        newTotalMatrix = np.asarray(totalMatrixRows)
        # self.printMatrix(newTotalMatrix)  # Test
        if randomShuffle == True:
            np.random.shuffle(newTotalMatrix)
        # self.printMatrix(newTotalMatrix)  # Test
        self.setTotalMatrix(newTotalMatrix)
        totalSpace = self.createSpace(newTotalMatrix)
        self.setTotalSpace(totalSpace)

    def delColumnsFromInputMatrix(self, columnPositions):
        assert isinstance(columnPositions, list)
        for p in columnPositions:
            assert isinstance(p, int) and p < len(self.getInputMatrix())
        newInMatrix = np.delete(self.getInputMatrix(), columnPositions, axis=1)
        self.setInputMatrix(newInputMatrix=newInMatrix)
        newInputSpace = self.createSpace(newInMatrix)
        self.setInputSpace(newInputSpace=newInputSpace)
        # print(self.getTotalMatrix())  # Test
        newTotalMatrix = np.delete(self.getTotalMatrix(), columnPositions, axis=1)
        self.setTotalMatrix(newTotalMatrix=newTotalMatrix)
        newTotalSpace = newInputSpace + self.getOutputSpace()
        self.setTotalSpace(newTotalSpace=newTotalSpace)


    def addToEquivalenceDict(self, eqDict, revEqDict, key, value):
        assert isinstance(eqDict, dict)
        assert isinstance(revEqDict, dict)
        eqDict[key] = value
        revEqDict[str(value)] = key
        return eqDict, revEqDict

    def roundNetResult(self, netResult, minValue, bit0Value, middleValue, bit1Value, maxValue):
        assert isinstance(netResult, list) or str(type(netResult)) == "<class 'numpy.ndarray'>"
        assert isinstance(minValue, int) or isinstance(minValue, float)
        assert isinstance(bit0Value, int) or isinstance(bit0Value, float)
        assert isinstance(middleValue, int) or isinstance(middleValue, float)
        assert isinstance(bit1Value, int) or isinstance(bit1Value, float)
        assert isinstance(maxValue, int) or isinstance(maxValue, float)
        convertedRes = []
        for row in netResult:
            convertedRow = []
            for elem in row:
                if elem >= minValue and elem <= maxValue:
                    if elem < middleValue:
                        convertedElem = bit0Value
                    elif elem == middleValue:
                        convertedElem = middleValue
                    else:  # if convertedElem > middleValue
                        convertedElem = bit1Value
                else:
                    if elem < minValue:
                        convertedElem = '- OutOfRange'
                    else:
                        convertedElem = '+ OutOfRange'
                convertedRow.append(convertedElem)
            convertedRes.append(convertedRow)
        return np.asarray(convertedRes)


# ################################## TEST #################################
'''
a = np.asarray([['ciao','salve',2],[4,3,4]])
ds = DatasetManager()
a = ds.matrixStringify(a)
print('Matrix a:')
ds.printMatrix(a)
spaceA = ds.createSpace(a)
print('Space of A:')
print(spaceA)
equivalence, reverseEquivalence = ds.createSpaceEquilalenceArrayDictionary(spaceA)
print('Equivalence dict:')
print(equivalence)
print('Reverse Equivalence Dict:')
print(reverseEquivalence)
newA = ds.matrixSymbolsConversion(equivalence,a)
print('Converted to:')
ds.printMatrix(newA)
oldA = ds.matrixSymbolsConversion(reverseEquivalence, newA, reverse=True)
print('Back Converted to:')
ds.printMatrix(np.asarray(oldA))
'''
