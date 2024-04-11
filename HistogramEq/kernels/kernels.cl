// File containing all kernel functions.
//use image input to create histogram array
//use atomic_add/scan on histogram to make cumulative histogram
//make LUT from cum hist
//map input image values to equalized values using LUT

kernel void max_val_reduce(global const uchar* A, global uchar* B, local uchar* l_mem) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	l_mem[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = N / 2; i > 0; i /= 2) { 
		if (lid < i) {
			if (l_mem[lid + i] > l_mem[lid])	
				l_mem[lid] = l_mem[lid + i];	
		}
		barrier(CLK_LOCAL_MEM_FENCE);  
	}
	if (lid == 0) {
		B[id] = l_mem[0];
	}
}

kernel void histogram(global const uchar* A, global int* H) {
	int id = get_global_id(0);
	int bin_index = A[id];
	//int bin_index = A[id] / (256 / A[255]);
	atomic_inc(&H[bin_index]);
}


// scan using atomic
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

kernel void freqency_normalisation(global int* A, global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);

	B[id] = (A[id] * 255.0 )/ A[255];
}

kernel void image_equalizer(global uchar* A, global uchar* B, global int* LUT) {
	int id = get_global_id(0);

	int result_LUT = LUT[A[id]];

	B[id] = result_LUT;
}

//RGBtoYCBCR
kernel void rgb2ycbrb(global uchar* A, global uchar* B) {
	int id = get_global_id(0);
	int image_size = get_global_size(0) / 3; //each image consists of 3 colour channels
	int colour_channel = id / image_size; // 0 - red, 1 - green, 2 - blue

	//Y channel
	if (colour_channel == 0) {
		int r_val = A[id];
		int g_val = A[id] + image_size;
		int b_val = A[id] + image_size + image_size;
		B[id] = 16 + 65.738 * r_val / 256 + 129.057 * g_val / 256 + 25.064 * b_val / 256;
	}
	//Cb channel
	else if (colour_channel == 1) {
		int r_val = A[id] - image_size;
		int g_val = A[id] ;
		int b_val = A[id] + image_size;
		B[id] = 128 - 37.945 * r_val / 256 - 74.494 * g_val / 256 + 112.439 * b_val / 256;

	}
	//Cr channel
	else {
		int r_val = A[id] - image_size - image_size;
		int g_val = A[id] - image_size;
		int b_val = A[id];
		B[id] = 128 + 112.439 * r_val - 94.154 * g_val / 256 - 18.285 * b_val / 256;
	}

}

kernel void ycbrb2rgb(global uchar* A, global uchar* B) {
	int id = get_global_id(0);
	int image_size = get_global_size(0) / 3; //each image consists of 3 colour channels
	int colour_channel = id / image_size; // 0 - red, 1 - green, 2 - blue


	//red channel
	if (colour_channel == 0) {
		int Y_val = A[id];
		int Cr_val = A[id] + image_size + image_size;
		B[id] = 298.082 * Y_val / 256 + 408.583 * Cr_val / 256 - 222.921;
	}
	//green channel
	else if (colour_channel == 1) {
		int Y_val = A[id] - image_size;
		int Cb_val = A[id];
		int Cr_val = A[id] + image_size;
		B[id] = 298.082 * Y_val / 256 - 100.291 * Cb_val / 256 - 208.120 * Cr_val / 256 + 135.576;

	}
	//blue channel
	else {
		int Y_val = A[id] - image_size - image_size;
		int Cb_val = A[id] - image_size;
		B[id] = 298.082 * Y_val / 256 + 516.412 * Cb_val / 256 - 276.836;
	}
}

