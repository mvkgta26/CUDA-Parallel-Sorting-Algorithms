# 2-Way RadixSort


## CUDA Implementation of 2-Way Radix Sort for integers
#### MAIN: "kernel.cu"

## Core Algorithm: Array arr[] of size n:
### Repeat for i = 0 --> i = 31
  1. *Compact* all the elements of arr[] array based on following *predicate*: Check if bit-i is 1
  2. Produce a *Blelloch Scan* of the Compact array
  3. Use the scan array as *Scatter-Addresses* to do the *Split For Bit-i*:   
        a. All the elements with bit-i == 0 are grouped in the beginning. Original order between them is retained  
        b. All elements with bit-i == 1 are grouped in the end. Original order between them is retained
        
        
        
### References: 
  1. The paper by Ha, Krüger, and Silva : https://vgc.poly.edu/~csilva/papers/cgf.pdf
  2. Blelloch Scan Implementation from : https://github.com/mark-poscablo  , (https://github.com/mark-poscablo/gpu-prefix-sum) 
  3. Udacity CS344: Intro to Parallel Programming (http://www.udacity.com/wiki/CS344)
  4. Bozidar, Darko & Dobravec, Tomaž. (2015). Comparison of parallel sorting algorithms. (https://www.researchgate.net/publication/283761857_Comparison_of_parallel_sorting_algorithms)
        

### Step and Work Complexity:
       Step Complexity : O(1)
       Work Complexity : O(n)

# IMPORTANT NOTE:
  #### * Please refer "DESCRIPTION-RADIX-SORT.pdf" document for DETAILED EXPLANATION of the Project
  #### * Open "RadixSort.sln" to open the entire project
