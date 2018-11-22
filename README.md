# infinite-hmm
Proof-of-concept implementation of an Infinite Hidden Markov Model.

## Requirements
(See `requirements.txt`)

## TODOs
+ [X] Matrix class
+ [X] HDP class
+ [X] InfiniteHMM class
+ [ ] BeamSampler class
+ [ ] GibbsSampler class
+ [ ] ParticleFilter class
+ make the sliced indexing operations in Matrix class more intuitive

## Further Reading
+ https://github.com/mblondel/soft-dtw/blob/master/sdtw/soft_dtw_fast.pyx
+ https://github.com/cythonbook/examples/blob/master/10-numpy-typed-memviews/01-typed-memviews/memviews.pyx
+ https://cs.nyu.edu/courses/spring12/CSCI-GA.3033-014/Assignment1/function_pointers.html
+ https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html#declaring-the-numpy-arrays-as-contiguous
+ https://yuen.fr/2015/02/27/cython-cheatsheet.html
+ http://mlg.eng.cam.ac.uk/zoubin/talks/BayesHMMs10.pdf
+ https://github.com/tmills/ihmm