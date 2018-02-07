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
