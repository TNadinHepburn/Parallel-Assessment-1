// The layout and functions in this project build upon the code from workshop Tutorial 2 & Tutorial 3

#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -f : input image file (default: test.ppm)" << std::endl;
	std::cerr << "  -b : number of bins in histogram (default: 255)" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char** argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	string image_filename = "test.pgm";
	int no_bins = 256;

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if ((strcmp(argv[i], "-b") == 0) && (i < (argc - 1))) { no_bins = std::stoi(argv[++i]); }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	cimg::exception_mode(0);

	//detect any potential exceptions
	try {
		CImg<unsigned char> image_input(image_filename.c_str());
		CImgDisplay disp_input(image_input, "input");

		//Part 2 - host operations
		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//2.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}


		typedef int mytype;
		std::vector<mytype> HISTOGRAM(no_bins);
		size_t hist_size = HISTOGRAM.size() * sizeof(mytype);//size in bytes


		//Part 4 - device operations

		//device - buffers
		cl::Buffer buf_image_input(context, CL_MEM_READ_ONLY, image_input.size());
		cl::Buffer buf_hist_output(context, CL_MEM_READ_WRITE, hist_size);
		cl::Buffer buf_cumul_hist_output(context, CL_MEM_READ_WRITE, hist_size);
		cl::Buffer buf_LUT_output(context, CL_MEM_READ_WRITE, hist_size);
		cl::Buffer buf_equal_output(context, CL_MEM_READ_WRITE, image_input.size());
		// cl::Buffer buf_bins(context, CL_MEM_READ_ONLY, sizeof(int));
		vector<unsigned char> equal_output_buffer(image_input.size());


		//4.1 Copy data to device memory
		queue.enqueueWriteBuffer(buf_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);
		queue.enqueueFillBuffer(buf_hist_output, 0, 0, hist_size);
		//queue.enqueueWriteBuffer(buf_bins, CL_TRUE, 0, sizeof(int), &nr_bins);

		//4.2 Setup and execute all kernels (i.e. device code)
		cl::Kernel kernel_histogram = cl::Kernel(program, "histogram");
		kernel_histogram.setArg(0, buf_image_input);
		kernel_histogram.setArg(1, buf_hist_output);
		//kernel.setArg(2, buf_bins);

		cl::Event prof_event_hist;

		//call all kernels in a sequence
		queue.enqueueNDRangeKernel(kernel_histogram, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &prof_event_hist);
		queue.enqueueReadBuffer(buf_hist_output, CL_TRUE, 0, hist_size, &HISTOGRAM[0]);

		std::cout << "Histogram = " << HISTOGRAM << std::endl;

		std::vector<mytype> CUMUL_HISTOGRAM(256);
		size_t cumul_hist_size = CUMUL_HISTOGRAM.size() * sizeof(mytype);//size in bytes

		//4.1 Copy data to device memory
		queue.enqueueFillBuffer(buf_cumul_hist_output, 0, 0, cumul_hist_size);

		//4.2 Setup and execute all kernels (i.e. device code)
		cl::Kernel kernel_cumul_histogram = cl::Kernel(program, "simple_cumul_hist");
		kernel_cumul_histogram.setArg(0, buf_hist_output);
		kernel_cumul_histogram.setArg(1, buf_cumul_hist_output);


		cl::Event prof_event_cumul;
		//call all kernels in a sequence
		queue.enqueueNDRangeKernel(kernel_cumul_histogram, cl::NullRange, cl::NDRange(hist_size), cl::NullRange, NULL, &prof_event_cumul);
		queue.enqueueReadBuffer(buf_cumul_hist_output, CL_TRUE, 0, cumul_hist_size, &CUMUL_HISTOGRAM[0]);

		std::cout << "Cumulative Histogram = " << CUMUL_HISTOGRAM << std::endl;

		std::vector<mytype> LUT(256);
		size_t LUT_size = LUT.size() * sizeof(mytype);//size in bytes

		queue.enqueueFillBuffer(buf_LUT_output, 0, 0, LUT_size);
		cl::Kernel kernel_LUT = cl::Kernel(program, "freqency_normalisation");
		kernel_LUT.setArg(0, buf_cumul_hist_output);
		kernel_LUT.setArg(1, buf_LUT_output);


		cl::Event prof_event_norm;

		queue.enqueueNDRangeKernel(kernel_LUT, cl::NullRange, cl::NDRange(LUT_size), cl::NullRange, NULL, &prof_event_norm);
		queue.enqueueReadBuffer(buf_LUT_output, CL_TRUE, 0, LUT_size, &LUT[0]);

		std::cout << "LUT = " << LUT << std::endl;

		cl::Kernel kernel_equalize = cl::Kernel(program, "image_equalizer");
		kernel_equalize.setArg(0, buf_image_input);
		kernel_equalize.setArg(1, buf_equal_output);
		kernel_equalize.setArg(2, buf_LUT_output);

		cl::Event prof_event_equal;

		queue.enqueueNDRangeKernel(kernel_equalize, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &prof_event_equal);


		queue.enqueueReadBuffer(buf_equal_output, CL_TRUE, 0, equal_output_buffer.size(), &equal_output_buffer.data()[0]);

		cout << endl;
		std::cout << "Histogram kernel execution time [ns]: " << prof_event_hist.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_hist.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(prof_event_hist, ProfilingResolution::PROF_US) << endl;
		cout << endl;

		std::cout << "Cumulative Histogram kernel execution time [ns]: " << prof_event_cumul.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_cumul.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(prof_event_cumul, ProfilingResolution::PROF_US) << endl;
		cout << endl;

		std::cout << "LUT kernel execution time [ns]: " << prof_event_norm.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_norm.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(prof_event_norm, ProfilingResolution::PROF_US) << endl;
		cout << endl;

		std::cout << "Image equalization kernel execution time [ns]: " << prof_event_equal.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_equal.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(prof_event_equal, ProfilingResolution::PROF_US) << endl;
		cout << endl;


		CImg<unsigned char> equalized_image(equal_output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		CImgDisplay disp_output(equalized_image, "output");

		while (!disp_input.is_closed() 
			&& !disp_input.is_keyESC() && !disp_output.is_closed() && !disp_output.is_keyESC()) {
			disp_input.wait(1);
			disp_output.wait(1);
		}
	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}
	return 0;
}
