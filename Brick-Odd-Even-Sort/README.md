# Odd-Even-Sort
### Parallel CUDA implementation of ODD-EVEN-SORT , a variant of bubble-sort

 #### MAIN: "kernel.cu"

# Core Algorithm:
## Array of size n
1. Split the array into n-pairs and number them
2. Parallelly sorth the even pairs 
3. Parallelly sort the odd pairs
4. Repeat steps 3) and 4) n times 

### References :
  1)https://www.cpp.edu/~gsyoung/CS370/14Sp/parallel_sorting_kla%20Danny.pdf    
  2)Udacity CS344: Intro to Parallel Programming (http://www.udacity.com/wiki/CS344)

### Step and Work Complexity:
       Step Complexity : O(n)
       Work Complexity : O(n^2)

# IMPORTANT NOTE:
  #### * Please refer "DESCRIPTION-ODD-EVEN SORT.pdf" document for DETAILED EXPLANATION of the Project
  #### * Open "CudaRuntime1.sln" to open the entire project
