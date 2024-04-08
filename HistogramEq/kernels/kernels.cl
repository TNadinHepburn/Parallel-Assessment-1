// File containing all kernel functions.
//use image input to create histogram array
//use atomic_add/scan on histogram to make cumulative histogram
//make LUT from cum hist
//map input image values to equalized values using LUT



kernel void histogram_gs(global const uchar* A, global int* H) {
	int id = get_global_id(0);
	//int lid = get_local_id(0);
	//int bin_index = A[id];
	int bin_index = A[id] / (256 / A[255]);
	//if (lid < NB)
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

