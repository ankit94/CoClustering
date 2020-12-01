# Sematic Web Mining : Co-Clustering
In this project, we aim to perform Co-clustering on various datasets. Co-clustering, or two-mode clustering is a data mining technique which allows simultaneous clustering of the rows and columns of a matrix.

    Install the required python libraries using the command  
    pip install -r requirements.txt
        
For the execution, we've implemented 3 algorithms over 3 datasets. run code/clustering_interface.py to use any combination

    $ python clustering_interface.py -c <algorithm> -d <dataset>
    
    Required arguments:
        -c, --clustering_technique select algo to run [spectral, subspace, infoth]
        -d, --dataset select dataset to run on [classic3, cstr, mnist]
