# NeuralMAS
> Neural Network and Multi-Agent System Hybrid Framework.

## The Neural Network
The Neural Network has been developed using Keras library, running on top of TensorFlow.
See: 
- https://keras.io/
- https://www.tensorflow.org/

Keras is very powerful and can run on top of Theano or CNTK too, but you can use a different technology of your choise insted of it.



## The Multi-Agent System
The Multi-Agent System (MAS) has been developed using DALI, or better Koinè DALI, running on top of SICStus Prolog.
See: 
- https://github.com/agnsal/KOINE-DALI
- https://github.com/AAAI-DISIM-UnivAQ/DALI
- https://sicstus.sics.se/
You can choose to develop the MAS with a different technology instead of it.



## The Communication
The communication channel between the Neural Network and the MAS is made using Redis.
See:
- https://redis.io/



## The Environment
You can use it on both Linux and Windows, and it is compatible with Docker too:
See:
- https://github.com/agnsal/docker-PyRedis
- https://www.docker.com/

## Instructions
1. To install Redis see: https://redis.io/
2. To install Keras and Tensorflow see: https://keras.io/ and https://www.tensorflow.org/
3. To install SISCtus Prolog see: https://sicstus.sics.se/
4. To get KOINE-DALI see: https://github.com/agnsal/KOINE-DALI
5. To install numpy and Keras dependencies:
```sh
  sudo pip3 install h5py
  sudo pip3 install numpy
```
6. To join the trained net file pieces and decompress it:
```sh
  cat LetterNet.h5* > LetterNet.h5.tar.gz
  tar -xvzf LetterNet.h5.tar.gz
```

