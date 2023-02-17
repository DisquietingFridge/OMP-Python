# OMP-Python

The file "OMP.py" is an implementation I made of the Orthogonal Matching Pursuit algorithm. In particular, it implements an optimized version of the algorithm called KSVD-OMP, described [here](https://www.cs.technion.ac.il/~ronrubin/Publications/KSVD-OMP-v2.pdf).

It's from a group project I was part of, as assessment for the ANU course "ENGN8535 Data Analytics". The assessment involved the implementation of a real-world application of data analytics- in our case, image denoising.

OMP is a greedy algorithm that seeks to add a set of "dictionary" vectors together, such that the error-margin between an input vector and the resulting output is within a particular bound. It was instrumental in the image-denoising algorithm we ended up implementing.

I may add more files that I contributed to from the project here, but I also wouldn't want others' contribution mistaken for my own, and I'd want to take care that it's presented in a coherent way. For now, I've included the OMP algorithm as a self-contained example of

* My experience with Python
* Implementation of a non-trivial algorithm in code from its description in an academic paper
* Familiarity with matrix-processing operations
