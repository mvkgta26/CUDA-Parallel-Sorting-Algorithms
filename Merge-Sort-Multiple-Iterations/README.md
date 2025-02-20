# Merge-Sort-Multiple-Iterations


#### MAIN: "kernel.cu" 
This is the parallel CUDA implementation of merge sort.


# CORE ALGORITHM:
    1)Split the input array into sections of specific size      
    2)Parallelly sort each individual section using a single block of threads                
    3)Iteratively merge sorted sections in order to produce bigger sections, until finally entire array is sorted                        
        All merges in a stage are parallelly done by multiple blocks of threads (Each block responsible for one merge)      
        
            
### References:   
    1)CUDA Zone | NVIDIA Developer (https://developer.nvidia.com/cuda-zone)   
    2)Udacity CS344: Intro to Parallel Programming (http://www.udacity.com/wiki/CS344)
    3)Bozidar, Darko & Dobravec, Tomaž. (2015). Comparison of parallel sorting algorithms. 

## STEP-COMPLEXITY AND WORK-COMPLEXITY:
    ## STAGE-1:
        1) STEP-COMPLEXITY: O(1)
        2) WORK-COMPLEXITY: O(n)
    ## STAGE-2:
        1) STEP-COMPLEXITY: O(log2(n))
        2) WORK-COMPLEXITY : O(n)

# IMPORTANT NOTE:
  #### * Please refer "DESCRIPTION-MERGE-SORT.pdf" document for DETAILED EXPLANATION of the Project
  #### * Open "Merge-Sort-Small-Size.sln" to open the entire project
  #### * Please view the **"Merge Sort.pptx" file for detailed description and explanation** of project.
