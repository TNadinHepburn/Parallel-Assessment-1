// File containing all kernel functions.
//use image input to create histogram array
//use scan on histogram to make cumulative histogram
//make LUT from cum hist
//map input image values to equalized values using LUT



kernel void histogram(global const uchar* A, global int* H) {
	int id = get_global_id(0);
	//int lid = get_local_id(0);
	int bin_index = A[id];

	//if (lid < nr_bins)
		//H[lid] = 0;

	//barrier(CLK_LOCAL_MEM_FENCE);

	atomic_inc(&H[bin_index]);
}

kernel void simple_cumul_hist(global int* H, global int* CH) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	for (int i = id + 1; i < N; i++)
		atomic_add(&CH[i], H[id]);
}


//Hillis-Steele basic inclusive scan
//requires additional buffer B to avoid data overwrite 
kernel void scan_hs(global int* A, global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	global int* C;

	for (int stride = 1; stride < N; stride *= 2) {
		B[id] = A[id];
		if (id >= stride) {
			B[id] += A[id - stride];
		}
		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step

		C = A; A = B; B = C; //swap A & B between steps
	}
}

//Blelloch basic exclusive scan
//
kernel void scan_bl(global int* A) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	int t;
	//up-sweep
	for (int stride = 1; stride < N; stride *= 2) {
		if (((id + 1) % (stride * 2)) == 0) {
			A[id] += A[id - stride];
		}

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
	}
	//down-sweep
	if (id == 0) {
		A[N - 1] = 0;//exclusive scan
	}

	barrier(CLK_GLOBAL_MEM_FENCE); //sync the step

	for (int stride = N / 2; stride > 0; stride /= 2) {
		if (((id + 1) % (stride * 2)) == 0) {
			t = A[id];
			A[id] += A[id - stride]; //reduce 
			A[id - stride] = t;		 //move
		}
		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
	}
}


kernel void freqency_normalisation(global int* A, global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);

	B[id] = A[id] * 255.0 / A[255];

}

kernel void image_equalizer(global uchar* A, global uchar* B, global int* LUT) {
	int id = get_global_id(0);

	int result_LUT = LUT[A[id]];

	B[id] = result_LUT;
}

//RGBtoYCHMK
kernel void image_equalizer(global int* R, global int* G, global int* B, global int* Y, global int* Cb, global int* Cr) {
