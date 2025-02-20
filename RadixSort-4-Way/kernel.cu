#ifndef __CUDACC__
	#define __CUDACC__
#endif

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cuda.h>
#include "device_functions.h"
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <windows.h>
#include < time.h >
//#include <iostream>

#include "scan.h"
#include "kernel.h"


//-------------------------------------------------------CPU TIMER LIBRARY-------------------------------------------------------

#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
#define DELTA_EPOCH_IN_MICROSECS  116444736000000000Ui64 // CORRECT
#else
#define DELTA_EPOCH_IN_MICROSECS  116444736000000000ULL // CORRECT
#endif

struct timezone
{
	int  tz_minuteswest; /* minutes W of Greenwich */
	int  tz_dsttime;     /* type of dst correction */
};

// Definition of a gettimeofday function
int gettimeofday(struct timeval* tv, struct timezone* tz)
{
	// Define a structure to receive the current Windows filetime
	FILETIME ft;

	// Initialize the present time to 0 and the timezone to UTC
	unsigned __int64 tmpres = 0;
	static int tzflag = 0;

	if (NULL != tv)
	{
		GetSystemTimeAsFileTime(&ft);

		// The GetSystemTimeAsFileTime returns the number of 100 nanosecond 
		// intervals since Jan 1, 1601 in a structure. Copy the high bits to 
		// the 64 bit tmpres, shift it left by 32 then or in the low 32 bits.
		tmpres |= ft.dwHighDateTime;
		tmpres <<= 32;
		tmpres |= ft.dwLowDateTime;

		// Convert to microseconds by dividing by 10
		tmpres /= 10;

		// The Unix epoch starts on Jan 1 1970.  Need to subtract the difference 
		// in seconds from Jan 1 1601.
		tmpres -= DELTA_EPOCH_IN_MICROSECS;

		// Finally change microseconds to seconds and place in the seconds value. 
		// The modulus picks up the microseconds.
		tv->tv_sec = (long)(tmpres / 1000000UL);
		tv->tv_usec = (long)(tmpres % 1000000UL);
	}

	if (NULL != tz)
	{
		if (!tzflag)
		{
			_tzset();
			tzflag++;
		}

		// Adjust for the timezone west of Greenwich
		tz->tz_minuteswest = _timezone / 60;
		tz->tz_dsttime = _daylight;
	}

	return 0;
}






//Takes arr[] as input and produces compact_0[] for bit-00, compact_1[] for bit-01, compact_2[] for bit-10 and compact_3[] for bit-11 , in bit-poisiton 'i' and 'i+1'
__global__ void compacter(unsigned int* arr, unsigned int* compact_0, unsigned int* compact_1, 
							unsigned int* compact_2, unsigned int* compact_3, int i)
{
	int idx = threadIdx.x;

	unsigned int num = arr[idx];

	//Predicate: Check if bit-0 is 1, then output is 1
	compact_0[idx] = (!((num >> (i + 1)) & 1)) && ( !((num >> i) & 1) ) ;    // '00' in bit-i+1 and bit-i
	compact_1[idx] =  ( !((num >> (i+1) ) & 1) ) && ( (num >> i ) & 1 ) ;    // '01' in bit-i+1 and bit-i
	compact_2[idx] =  ( (num >> (i + 1)) & 1) &&  ( ! ( (num >> i) & 1 ) )  ;    // '10' in bit-i+1 and bit-i
	compact_3[idx] =  ((num >> (i + 1)) & 1) && ((num >> i) & 1);				// '11' in bit-i+1 and bit-i
}


//Takes arr[], scatter addresses (scan of compact_0, 1, 2, 3, 4), compact_0,1,2,3[] as input, and produces sorted array for the respective bit, in arr[] itself
	//arr[] = Input  array to be sorted according to respective bit
	//scan_0[] = Scan array for compact of bit-0 in respective position
	//scan_1[] = Scan array for compact of bit-1 in respective position
	//compact_0[] = Compact array for bit-0 in respective position for arr[]
	//n = Size of arr[]
	//temp_out[] is temporary array used to store sorted array before copying back to arr[]
__global__ void scatter_sort(unsigned int* arr, unsigned int* scan_0, unsigned int* scan_1, unsigned int* scan_2, unsigned int* scan_3,
							 unsigned int* compact_0, unsigned int* compact_1, unsigned int* compact_2, unsigned int* compact_3,
							  int n, unsigned int* temp_out)
{
	int idx = threadIdx.x;

	int num[4];    // num[k]represents total number of elements with 'b1 b0' (binary of k), as bits at position i+1, i
	
	num[0] = scan_0[n];
	num[1] = scan_1[n];
	num[2] = scan_2[n];
	num[3] = scan_3[n];

	int pos;

	if (compact_0[idx] == 1)    //If element arr[idx] has bit-0 in current position
	{
		pos = scan_0[idx];    //The index position in output array where the arr[idx] is supposed to go
		temp_out[pos] = arr[idx];
	}

	else if (compact_1[idx] == 1)  //If element arr[idx] has bit-1 in current position
	{
		pos = scan_1[idx] + num[0];    //num_of_0 : Offset
		temp_out[pos] = arr[idx];
	}

	else if (compact_2[idx] == 1)  //If element arr[idx] has bit-1 in current position
	{
		pos = scan_2[idx] + num[0] + num[1];    //num_of_0 + num_of_1 : Offset
		temp_out[pos] = arr[idx];
	}

	else if (compact_3[idx] == 1)  //If element arr[idx] has bit-1 in current position
	{
		pos = scan_3[idx] + num[0] + num[1] + num[2];    //num_of_0 + num_of_1 + num_of_2 : Offset
		temp_out[pos] = arr[idx];
	}

	__syncthreads;   //Barrier


	//Copy from temp_out[] back to arr[]
	arr[idx] = temp_out[idx];

}


//Calculate scan for bit-1: 
	//scan_1[] is scan array for bit-1 that is to be calculated
	//scan_0[] is scan array for bit-0, given as input
	//n = size of array that is to be sorted
__global__ void scan_1_calculator(unsigned int* scan_1, unsigned int* scan_0, int n)
{
	int idx = threadIdx.x;
	int num_of_0_elements = scan_0[n];               //Number of elements with 0 in respective bit position
	int num_of_1_elements = n - num_of_0_elements;   //Number of elements with 1 in respective bit position = Total number of elements - Number of elements with 0 in respective bit position

	//Last element (at index n) of scan_1[] = num_of_1_elements 
	if ( idx == (n) )
	{
		scan_1[idx] = num_of_1_elements;
	}


	else
	{
		//Formula to calculate scan_1[i] using scan_1[i+1] and compact_1[]. 
		//Logic:
			//scan_1[i] = Total Number of 1-elements (elements with 1 at respective bit-position), till index i- in arr[]
		    //  i  -->index = Number of elements in arr[] before current element
			// scan_0[i]  --> Number of 0-elements till index-i in arr[]
		scan_1[idx] = idx - scan_0[idx];   
	}

}

void seq_bubble_sort(unsigned int* arr, int n)      //In-Place Sequential Bubble Sort
{
	int i, k, flag, temp;
	for (k = 1; k < (n - 1); k++)
	{
		flag = 0;
		for (i = 0; i < (n - k); i++)
		{
			if (arr[i] > arr[i + 1])
			{
				temp = arr[i];    //
				arr[i] = arr[i + 1];  //  Swapping A[i+1] and A[i]
				arr[i + 1] = temp;  //
			}
		}
	}
}

void radix_sort()
{

	//---------------------------------Create input arr[] and h_compact[] --------------------------------------------------------------------	

	//unsigned int h_arr[] = { 1989, 1124, 9701, 2900, 5241, 6702, 1784, 1096, 3382, 863, 8966, 2830, 2043, 9889, 3414, 2810, 644, 1420, 1065, 8597, 4419, 1388, 8796, 6139, 1158, 7689, 4114, 1865, 8485, 9190, 564, 5409, 9016, 3515, 2993, 6536, 7524, 2982, 9953, 3228, 1678, 2770, 7072, 3091, 9815, 7426, 363, 5139, 7481, 6183, 3392, 5808, 2407, 5425, 3353, 4484, 1825, 6621, 3046, 2710, 1533, 118, 7109, 2917, 4677, 641, 4802, 9366, 5029, 59, 9204, 9173, 5875, 7417, 9070, 2929, 1529, 6985, 5923, 9271, 1907, 8870, 7774, 3496, 2258, 5167, 7164, 9915, 2854, 1089, 1275, 1258, 7372, 6088, 375, 8333, 840, 3010, 5606, 2534, 138, 7266, 1339, 1714, 3615, 9601, 9817, 6208, 9483, 5269, 7835, 4681, 772, 7000, 6922, 6833, 4163, 6246, 3913, 5617, 7160, 2678, 8284, 8454, 7671, 8963, 4080, 4745, 6173, 8950, 4646, 1182, 4264, 3858, 2221, 270, 7341, 8382, 2889, 5722, 3947, 784, 4280, 439, 5331, 6080, 7296, 4075, 8380, 8049 };
	//unsigned int h_arr[] = { 26338, 1102, 43406, 62440, 7116, 75423, 95442, 31454, 43177, 27829, 37527, 20088, 48786, 33484, 91084, 98038, 93852, 64998, 84009, 91983, 12691, 8128, 90209, 64050, 64526, 63539, 31716, 52617, 55245, 69338, 96600, 62877, 9070, 89736, 64799, 45587, 22108, 78332, 86473, 6608, 39686, 67631, 90273, 41757, 57024, 65715, 6247, 35982, 49869, 83823, 50126, 83367, 58983, 78590, 50726, 61549, 96979, 9315, 99939, 92868, 42634, 72012, 75395, 66760, 97571, 67335, 50690, 55839, 7193, 68868, 52778, 50767, 38817, 73182, 73173, 18570, 59518, 75198, 43944, 81905, 17130, 30499, 62531, 67788, 10212, 93307, 33485, 35421, 41800, 91439, 7128, 5244, 4857, 19666, 55995, 93197, 16521, 70814, 65956, 29551, 41088, 90651, 8364, 81397, 65599, 21928, 79453, 43220, 30210, 89518, 46937, 39088, 3064, 8272, 95080, 72370, 38519, 80624, 73855, 14338, 26999, 797, 7389, 22675, 29441, 91215, 57751, 20691, 73766, 19449, 65664, 4911, 51829, 52694, 59465, 18815, 47909, 6960, 71232, 31887, 33836, 75839, 77099, 72696, 66743, 23003, 67416, 79256, 92027, 14854, 67547, 66948, 73557, 81593, 40373, 36769, 91653, 55675, 91273, 82840, 12390, 55618, 98120, 60072, 10352, 26671, 53834, 67112, 11591, 66281, 4508, 74787, 76274, 50136, 87370, 33150, 30572, 77426, 22738, 93988, 76158, 88211, 56058, 61264, 49590, 64482, 76270, 58166, 83076, 3572, 53961, 96103, 69219, 56519, 39247, 47900, 96419, 20127, 21434, 83026, 53972, 12275, 41420, 14470, 83492, 98815, 11219, 28147, 40874, 75086, 58471, 14116, 12447, 59360, 86793, 53957, 74870, 56028, 78651, 78761, 68152, 31585, 86419, 73739, 13907, 47057, 52096, 17777, 46550, 14465, 97001, 75372, 82870, 2510, 54067, 53592, 49157, 50366, 72633, 69869, 6182, 91356, 76672, 13479, 20844, 35748, 16099, 52512, 89438, 59887, 65339, 85400, 64340, 34066, 77991, 38365, 79942, 7545, 94624, 3357, 87348, 79218, 70170, 80566, 9464, 64659, 95658, 58290, 44600, 13764, 83274, 46833, 13879, 72497, 61917, 16609, 77559, 58483, 24873, 69414, 11725, 52316, 53275, 13306, 35063, 87894, 5532, 53202, 24268, 58043, 33283, 6492, 45431, 30406, 32716, 88618, 33055, 78184, 98307, 6890, 8994, 89487, 11234, 52622, 64764, 11958, 44815, 1532, 79736, 47961, 14542, 77658, 75199, 45576, 19542, 7035, 20812, 91288, 51135, 4249, 98847, 6941, 38347, 64487, 15925, 99375, 8921, 90455, 17026, 97407, 29745, 56171, 32655, 30296, 25287, 92752, 33062, 54188, 59874, 18971, 80723, 14297, 57746, 85341, 15519, 2459, 10977, 97487, 88749, 88970, 57457, 89542, 45360, 455, 65658, 10186, 96217, 87608, 45218, 90238, 76122, 54096, 77985, 97251, 8675, 9146, 38822, 80023, 87709, 7900, 80506, 24446, 51014, 14603, 94082, 43508, 43274, 67228, 56128, 37014, 43127, 43953, 96690, 35271, 67673, 71158, 50788, 69814, 76142, 791, 86797, 15482, 98654, 52733, 31128, 64146, 96229, 11191, 23915, 99656, 78797, 15635, 9645, 48483, 2935, 50237, 14411, 8588, 61111, 84174, 17764, 11618, 94105, 21335, 81908, 82632, 72680, 56202, 83381, 39008, 80899, 11145, 72512, 74094, 12690, 1350, 20388, 46408, 69910, 79823, 90323, 22544, 99050, 10961, 5284, 18371, 45704, 86015, 35965, 98364, 62855, 51699, 8593, 95676, 85731, 66224, 19520, 49572, 26118, 64236, 86854, 68097, 28618, 30063, 73433, 37923, 96022, 9800, 4261, 52449, 32161, 35946, 82185, 70005, 57610, 48663, 3915, 88572, 93349, 1709, 89097, 72236, 62070, 64663, 25783, 54343, 376, 57853, 6775, 81468, 32931, 44625, 91909, 52921, 84552, 94976, 65099, 37356, 86361, 34262, 571, 32338, 22506, 3641, 79365, 17411, 4160, 47729, 21059, 55742, 69135, 25617, 81408, 45984, 76631, 11955, 57728, 72173, 51879, 76821, 2146, 7880, 21508, 94189, 19944, 90717, 42518, 77249, 66967, 54725, 30968, 39205, 35117, 86598, 60500, 8489, 72419, 62481, 72656, 76243, 65476, 16709, 13908, 20416, 72724, 5089, 8265, 40097, 79298, 40171, 10218, 67885, 62477, 3068, 41719, 62229, 60152, 78163, 59021, 73703, 94992, 14615, 50868, 41622, 59259, 41659, 86480, 85254, 19229, 13745, 87784, 98089, 99842, 10269, 27592, 89481, 68467, 56126, 32884, 92184, 17521, 85446, 58156, 80504, 92966, 73456, 11176, 26435, 72496, 32129, 49957, 8609, 30759, 28815, 84018, 81335, 13435, 68014, 78931, 4503, 7175, 25168, 60915, 74655, 80498, 31761, 82006, 14226, 30640, 75321 };

	unsigned int h_arr[600] = { 3871, 3742, 1863, 2402, 904, 3951, 2762, 1795, 3884, 2894, 3505, 2386, 1871, 2802, 3890, 1051, 2738, 975, 1875, 1372, 107, 2, 1800, 1453, 742, 2564, 2012, 1332, 3359, 327, 291, 2288, 137, 3779, 1268, 2358, 3608, 134, 2031, 3770, 2817, 2307, 1463, 383, 1796, 3704, 2222, 2432, 2338, 416, 921, 3702, 114, 3375, 3191, 1854, 2810, 3331, 2087, 1763, 1842, 957, 3514, 2675, 1513, 202, 2436, 677, 2873, 2504, 2744, 103, 2964, 2997, 141, 285, 3333, 3497, 3512, 2861, 3278, 1384, 3009, 1874, 1536, 1062, 280, 365, 2119, 1809, 3669, 3963, 2274, 2334, 2773, 457, 1528, 1045, 3825, 1548, 223, 3063, 2724, 323, 1722, 1018, 1412, 3920, 3665, 2048, 616, 1245, 406, 124, 2231, 666, 840, 1519, 2572, 279, 2884, 316, 209, 1621, 3412, 963, 1315, 177, 980, 3475, 43, 1630, 1725, 680, 1627, 3497, 3552, 1401, 190, 102, 3254, 463, 2854, 663, 3984, 2604, 511, 2353, 65, 3900, 2476, 3144, 3997, 621, 633, 2867, 2250, 2958, 3031, 3063, 3605, 863, 752, 1443, 459, 2689, 3500, 278, 643, 56, 1934, 1140, 3868, 2692, 3287, 3134, 3575, 1961, 2400, 1960, 3054, 3911, 3811, 1242, 3866, 3677, 501, 725, 2042, 1306, 957, 1312, 2180, 3708, 1932, 2619, 3720, 3681, 2887, 153, 513, 360, 2032, 2050, 1086, 3553, 3594, 213, 950, 21, 1435, 610, 1369, 877, 367, 2557, 3444, 98, 3635, 1839, 2828, 2167, 3989, 3850, 3549, 1882, 2498, 2263, 2718, 3522, 1180, 3080, 3004, 3645, 1582, 2193, 3087, 3977, 2647, 10, 2440, 1054, 1785, 1441, 1383, 2020, 616, 3122, 2255, 1362, 3668, 3369, 1419, 69, 1843, 2458, 125, 2226, 1856, 2916, 428, 1703, 1279, 2236, 2016, 2963, 791, 165, 2535, 2819, 3806, 2728, 2820, 2929, 3665, 1422, 2733, 1430, 3988, 1571, 2716, 1236, 1689, 2262, 2746, 2162, 2322, 951, 2029, 2890, 3607, 2771, 2867, 2015, 3205, 1011, 3197, 3367, 1089, 1776, 1537, 1611, 298, 1067, 2537, 86, 549, 2708, 3010, 407, 3120, 2722, 1028, 239, 1881, 805, 1612, 2568, 789, 3755, 1045, 3483, 1495, 927, 660, 262, 28, 163, 2483, 2423, 3596, 2391, 3082, 220, 3378, 2752, 3209, 1616, 3405, 2639, 653, 3237, 3015, 3417, 3875, 961, 3932, 759, 118, 1441, 1001, 760, 3980, 1114, 981, 2681, 1182, 3707, 2761, 881, 2693, 323, 936, 2473, 1046, 1889, 3058, 2468, 2381, 2609, 910, 1847, 318, 2155, 3776, 1705, 3334, 2009, 1538, 2966, 2929, 435, 197, 21, 3489, 3751, 2209, 2426, 2329, 835, 3059, 412, 92, 383, 1317, 795, 3456, 2615, 2760, 3586, 3129, 1183, 3369, 1940, 316, 3004, 748, 3793, 3917, 1241, 3145, 1540, 1591, 3403, 3771, 627, 3213, 2179, 2199, 957, 745, 364, 3649, 1294, 101, 1027, 2682, 3508, 3262, 2357, 2062, 1704, 370, 841, 3553, 1029, 2438, 620, 320, 3368, 3810, 3373, 2617, 3072, 1057, 2023, 2138, 2712, 3769, 456, 3717, 321, 2757, 1618, 387, 1083, 2582, 1073, 2509, 2856, 3041, 2374, 2319, 1225, 125, 1783, 195, 1203, 3195, 2142, 3441, 2096, 113, 2113, 3151, 815, 3224, 2175, 2590, 2986, 3608, 486, 720, 2276, 553, 1279, 650, 825, 988, 2737, 310, 1944, 1225, 1205, 681, 3886, 3097, 3350, 79, 1900, 1863, 2072, 1851, 2926, 2507, 3396, 2226, 1035, 377, 3929, 3050, 3643, 51, 827, 1652, 3409, 1942, 2810, 2639, 1955, 2317, 1068, 1847, 1240, 1275, 2173, 2844, 1174, 3445, 3502, 482, 2693, 353, 2840, 3291, 548, 3480, 1767, 648, 1745, 2854, 8, 3780, 2827, 1243, 1917, 1184, 1559, 1524, 673, 1723, 3387, 1583, 2526, 2677, 2364, 1933, 3310, 2922, 3932, 2571, 98, 1228, 2162, 108, 974, 3168, 3369, 2674, 1049, 3714, 2725, 2340, 2224, 3641, 240, 1073, 469, 1580, 1385, 931, 963, 3474, 183, 2598, 1804, 373, 2318, 1195, 1308, 3331, 3946, 2104, 3146, 64, 901, 3478, 1527, 3516, 1525 };
	int n = sizeof(h_arr) / sizeof(unsigned int);   //Size of input array
	unsigned int* h_compact = new unsigned int[n];


	//------------------Create d_arr[], d_compact_0[], d_compact_1[], d_compact_2[], d_compact_3[] d_temp_out[]------------------------------------------ 
	unsigned int* d_arr, *d_compact_0, *d_compact_1, * d_compact_2, * d_compact_3, *d_temp_out;
	//d_temp_out[] is temporary array used to temporarily store sorted output in kernel

	cudaMalloc((void**) &d_arr, n * sizeof(unsigned int));   //Allocate d_arr[] in GPU for n elements
	cudaMemcpy((void*) d_arr, (void*) h_arr, n * sizeof(unsigned int), cudaMemcpyHostToDevice);

	cudaMalloc((void**) &d_compact_0, n * sizeof(unsigned int));   //Allocate compact_0 array [] in GPU for n elements
	cudaMalloc((void**) &d_compact_1, n * sizeof(unsigned int));
	cudaMalloc((void**) &d_compact_2, n * sizeof(unsigned int));
	cudaMalloc((void**) &d_compact_3, n * sizeof(unsigned int));
	cudaMalloc((void**) &d_temp_out, n * sizeof(unsigned int));   //Allocate compact_1 array [] in GPU for n elements



	//-------------------------------------------------Create Scan Arrays in Host--------------------------------------------------
		//n+1 because we need 1 extra element to hold scan of arr[i+1]
	unsigned int* h_scan_0 = new unsigned int [n+1];
	unsigned int* h_scan_1 = new unsigned int [n+1];
	unsigned int* h_scan_2 = new unsigned int [n+1];
	unsigned int* h_scan_3 = new unsigned int [n+1];
	

	//-------------------------------------------------Create Scan Arrays in Device--------------------------------------------------
	unsigned int* d_scan_0;   //For bit-00
	cudaMalloc((void**) &d_scan_0, (n+1) * sizeof(unsigned int));   //Allocate compact_0 array [] in GPU for n+1 elements

	unsigned int* d_scan_1;   //For bit-01
	cudaMalloc((void**) &d_scan_1, (n+1) * sizeof(unsigned int));

	unsigned int* d_scan_2;   //For bit-10
	cudaMalloc((void**)&d_scan_2, (n + 1) * sizeof(unsigned int));

	unsigned int* d_scan_3;   //For bit-11
	cudaMalloc((void**)&d_scan_3, (n + 1) * sizeof(unsigned int));

	GpuTimer timer;
	timer.Start();

	// ---------------------------Iterate 32 times (1 time for each bit of integer) and Make Kernel Call-------------------------
	int bit_pair;   //First index of bit-pair in arr[]
	for (bit_pair = 0; bit_pair <= 30; bit_pair = bit_pair+2)
	{
		//int bit_pair = 0;
		compacter <<< 1, n >>> (d_arr, d_compact_0, d_compact_1, d_compact_2, d_compact_3, bit_pair);     //Launch kernel : 1 block of n threads

		//Produce scan arrays
		sum_scan_blelloch(d_scan_0, d_compact_0, n + 1);
		sum_scan_blelloch(d_scan_1, d_compact_1, n + 1);
		sum_scan_blelloch(d_scan_2, d_compact_2, n + 1);
		sum_scan_blelloch(d_scan_3, d_compact_3, n + 1);

		scatter_sort <<<1, n>>>(d_arr, d_scan_0, d_scan_1, d_scan_2, d_scan_3, d_compact_0, d_compact_1, d_compact_2, d_compact_3, n, d_temp_out);

	}
	
	timer.Stop();
	double time_elapsed = timer.Elapsed();

	/*
	//--------------------------------Copy output to CPU h_out[] and print sorted array-----------------------------------------------------------
	unsigned int* h_out = new unsigned int[n];
	cudaMemcpy((void*) h_out, (void*) d_arr, (n) * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < n; i++)
	{
		printf("%d ", h_out[i]);
	}
	*/


	
	cudaMemcpy((void*)h_scan_0, (void*)d_arr, (n) * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	/*
	cudaMemcpy((void*)h_scan_1, (void*)d_scan_1, (n+1) * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)h_scan_2, (void*)d_scan_2, (n+1) * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)h_scan_3, (void*)d_scan_3, (n+1) * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	*/


	for (int i = 0; i < n; i++)
	{
		//printf("%d		%d		%d		%d \n", h_scan_0[i], h_scan_1[i], h_scan_2[i], h_scan_3[i]);
		printf("%d ", h_scan_0[i]);
	}

	printf("\n Time Elapsed : %g ms", time_elapsed);
	/*
	//Copy output
	cudaMemcpy((void*)h_scan_0, (void*)d_scan_0, (n+1) * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)h_scan_1, (void*)d_scan_1, (n+1) * sizeof(unsigned int), cudaMemcpyDeviceToHost);


	for (int i = 0; i < 9; i++)
	{
		printf("%d ", h_scan_0[i]);
	}
	printf("\n");

	for (int i = 0; i < 9; i++)
	{
		printf("%d ", h_scan_1[i]);
	}
	*/
}


void main()
{
	

	radix_sort();

	/*
	struct timeval timediff;

	
	gettimeofday(&timediff, NULL);
	double t1 = timediff.tv_sec + (timediff.tv_usec / 1000000.0);

	unsigned int h_arr[] = { 3871, 3742, 1863, 2402, 904, 3951, 2762, 1795, 3884, 2894, 3505, 2386, 1871, 2802, 3890, 1051, 2738, 975, 1875, 1372, 107, 2, 1800, 1453, 742, 2564, 2012, 1332, 3359, 327, 291, 2288, 137, 3779, 1268, 2358, 3608, 134, 2031, 3770, 2817, 2307, 1463, 383, 1796, 3704, 2222, 2432, 2338, 416, 921, 3702, 114, 3375, 3191, 1854, 2810, 3331, 2087, 1763, 1842, 957, 3514, 2675, 1513, 202, 2436, 677, 2873, 2504, 2744, 103, 2964, 2997, 141, 285, 3333, 3497, 3512, 2861, 3278, 1384, 3009, 1874, 1536, 1062, 280, 365, 2119, 1809, 3669, 3963, 2274, 2334, 2773, 457, 1528, 1045, 3825, 1548, 223, 3063, 2724, 323, 1722, 1018, 1412, 3920, 3665, 2048, 616, 1245, 406, 124, 2231, 666, 840, 1519, 2572, 279, 2884, 316, 209, 1621, 3412, 963, 1315, 177, 980, 3475, 43, 1630, 1725, 680, 1627, 3497, 3552, 1401, 190, 102, 3254, 463, 2854, 663, 3984, 2604, 511, 2353, 65, 3900, 2476, 3144, 3997, 621, 633, 2867, 2250, 2958, 3031, 3063, 3605, 863, 752, 1443, 459, 2689, 3500, 278, 643, 56, 1934, 1140, 3868, 2692, 3287, 3134, 3575, 1961, 2400, 1960, 3054, 3911, 3811, 1242, 3866, 3677, 501, 725, 2042, 1306, 957, 1312, 2180, 3708, 1932, 2619, 3720, 3681, 2887, 153, 513, 360, 2032, 2050, 1086, 3553, 3594, 213, 950, 21, 1435, 610, 1369, 877, 367, 2557, 3444, 98, 3635, 1839, 2828, 2167, 3989, 3850, 3549, 1882, 2498, 2263, 2718, 3522, 1180, 3080, 3004, 3645, 1582, 2193, 3087, 3977, 2647, 10, 2440, 1054, 1785, 1441, 1383, 2020, 616, 3122, 2255, 1362, 3668, 3369, 1419, 69, 1843, 2458, 125, 2226, 1856, 2916, 428, 1703, 1279, 2236, 2016, 2963, 791, 165, 2535, 2819, 3806, 2728, 2820, 2929, 3665, 1422, 2733, 1430, 3988, 1571, 2716, 1236, 1689, 2262, 2746, 2162, 2322, 951, 2029, 2890, 3607, 2771, 2867, 2015, 3205, 1011, 3197, 3367, 1089, 1776, 1537, 1611, 298, 1067, 2537, 86, 549, 2708, 3010, 407, 3120, 2722, 1028, 239, 1881, 805, 1612, 2568, 789, 3755, 1045, 3483, 1495, 927, 660, 262, 28, 163, 2483, 2423, 3596, 2391, 3082, 220, 3378, 2752, 3209, 1616, 3405, 2639, 653, 3237, 3015, 3417, 3875, 961, 3932, 759, 118, 1441, 1001, 760, 3980, 1114, 981, 2681, 1182, 3707, 2761, 881, 2693, 323, 936, 2473, 1046, 1889, 3058, 2468, 2381, 2609, 910, 1847, 318, 2155, 3776, 1705, 3334, 2009, 1538, 2966, 2929, 435, 197, 21, 3489, 3751, 2209, 2426, 2329, 835, 3059, 412, 92, 383, 1317, 795, 3456, 2615, 2760, 3586, 3129, 1183, 3369, 1940, 316, 3004, 748, 3793, 3917, 1241, 3145, 1540, 1591, 3403, 3771, 627, 3213, 2179, 2199, 957, 745, 364, 3649, 1294, 101, 1027, 2682, 3508, 3262, 2357, 2062, 1704, 370, 841, 3553, 1029, 2438, 620, 320, 3368, 3810, 3373, 2617, 3072, 1057, 2023, 2138, 2712, 3769, 456, 3717, 321, 2757, 1618, 387, 1083, 2582, 1073, 2509, 2856, 3041, 2374, 2319, 1225, 125, 1783, 195, 1203, 3195, 2142, 3441, 2096, 113, 2113, 3151, 815, 3224, 2175, 2590, 2986, 3608, 486, 720, 2276, 553, 1279, 650, 825, 988, 2737, 310, 1944, 1225, 1205, 681, 3886, 3097, 3350, 79, 1900, 1863, 2072, 1851, 2926, 2507, 3396, 2226, 1035, 377, 3929, 3050, 3643, 51, 827, 1652, 3409, 1942, 2810, 2639, 1955, 2317, 1068, 1847, 1240, 1275, 2173, 2844, 1174, 3445, 3502, 482, 2693, 353, 2840, 3291, 548, 3480, 1767, 648, 1745, 2854, 8, 3780, 2827, 1243, 1917, 1184, 1559, 1524, 673, 1723, 3387, 1583, 2526, 2677, 2364, 1933, 3310, 2922, 3932, 2571, 98, 1228, 2162, 108, 974, 3168, 3369, 2674, 1049, 3714, 2725, 2340, 2224, 3641, 240, 1073, 469, 1580, 1385, 931, 963, 3474, 183, 2598, 1804, 373, 2318, 1195, 1308, 3331, 3946, 2104, 3146, 64, 901, 3478, 1527, 3516, 1525 };
	int n = sizeof(h_arr) / sizeof(unsigned int);   //Size of input array

	//seq_bubble_sort(h_arr, n);

	
	gettimeofday(&timediff, NULL);
	double t2 = timediff.tv_sec + (timediff.tv_usec / 1000000.0);
	*/

}

