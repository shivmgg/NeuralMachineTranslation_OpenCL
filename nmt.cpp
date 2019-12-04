#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>

#include <fstream>
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <string>
#include <math.h>
#include <stdio.h>

#include <time.h>
void encode_input(char c, float* vector)
{
  //Input Vocab size: 71

  char encoding[72] = {' ', '!', '\"', '$', '%', '\'', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            ':', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
            'R', 'S', 'T', 'U', 'V', 'W', 'Y', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
            'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '\0'};

  int index = -1;
  for (int i=0; i<71; i++) {
    if (encoding[i] == c) {
      index = i;
      break;
    }
  }

  for (int i=0; i<71; i++) {
    vector[i] = 0;
  }
  vector[index] = 1;

}

char decode_output(float* vector)
{
  char decoding[84] = {'\t', '\n', ' ', '!', '$', '%', '\'', ',', '-', '.',
      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', '?', 'A', 'B', 'C', 'D',
      'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
      'U', 'V', 'W', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
      'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A',
      'O', 'U', 'B', 'a', 'o', 'u', ',', '"', '„', '\u202f'};

  int index = -1;
  for (int i=0; i<84; i++) {
    if (vector[i] == 1) {
      index = i;
      break;
    }
  }

  char c = decoding[index];
  return c;
}

int main() 
{
   float *hInputImage;
   float *hOutputImage;

   int imageRows;
   int imageCols;

   /* Set the filter here */
   int filterWidth;
   float filterFactor;
   float *filter;

  
  FILE *filePtr;
  filePtr = fopen("weights/lstm_1_0_.txt", "r");
  float *weights1 = (float *) malloc(sizeof(float)*71*256);
  for (int i=0; i<71; i++) {
    for (int j=0; j<256; j++) {
      fscanf(filePtr, "%f ", &weights1[i*256 + j]);
      //std::cout<<weights1[i*256 + j]<<" ";
      //encoder_warr[i*256 + j] = weights1[i*256 + j];
    }
  }

  filePtr = fopen("weights/lstm_1_1_.txt", "r");
  float *weights2 = (float *) malloc(sizeof(float)*64*256);
  for (int i=0; i<64; i++) {
    for (int j=0; j<256; j++) {
      fscanf(filePtr, "%f ", &weights2[i*256 + j]);
      //std::cout<<weights2[i*256 + j]<<" ";
      //encoder_uarr[i*256 + j] = weights2[i*256 + j];
    }
  }

  filePtr = fopen("weights/lstm_1_2_.txt", "r");
  float *weights3 = (float *) malloc(sizeof(float)*1*256);
  for (int i=0; i<256; i++) {
    fscanf(filePtr, "%f ", &weights3[i]);
    //std::cout<<weights3[i]<<" ";
    //encoder_barr[i] = weights3[i];
  }

  filePtr = fopen("weights/lstm_2_0_.txt", "r");
  float *weights4 = (float *) malloc(sizeof(float)*84*256);
  for (int i=0; i<84; i++) {
    for (int j=0; j<256; j++) {
      fscanf(filePtr, "%f ", &weights4[i*256 + j]);
      //std::cout<<weights1[i*256 + j]<<" ";
      //decoder_warr[i*256 + j] = weights4[i*256 + j];
    }
  }

  filePtr = fopen("weights/lstm_2_1_.txt", "r");
    // std::vector<float,aligned_allocator<float>> decoder_uarr(64*256);
  float *weights5 = (float *) malloc(sizeof(float)*64*256);
  for (int i=0; i<64; i++) {
    for (int j=0; j<256; j++) {
      fscanf(filePtr, "%f ", &weights5[i*256 + j]);
      //std::cout<<weights5[i*256 + j]<<" ";
      //decoder_uarr[i*256 + j] = weights5[i*256 + j];
    }
  }

  filePtr = fopen("weights/lstm_2_2_.txt", "r");
    // std::vector<float,aligned_allocator<float>> decoder_barr(1*256);
  float *weights6 = (float *) malloc(sizeof(float)*1*256);
  for (int i=0; i<256; i++) {
    fscanf(filePtr, "%f ", &weights6[i]);
  }

  filePtr = fopen("weights/dense_1_0_.txt", "r");
    // std::vector<float,aligned_allocator<float>> decoder_barr(1*256);
  float *weights7 = (float *) malloc(sizeof(float)*64*84);
  for (int i=0; i<64; i++) {
    for (int j=0; j<84; j++) {
      fscanf(filePtr, "%f ", &weights7[i*84 + j]);
      //std::cout<<weights7[i*84 + j]<<" ";
      //decoder_uarr[i*256 + j] = weights5[i*256 + j];
    }
  }

  filePtr = fopen("weights/dense_1_1_.txt", "r");
    // std::vector<float,aligned_allocator<float>> decoder_barr(1*256);
  float *weights8 = (float *) malloc(sizeof(float)*1*84);
  for (int i=0; i<84; i++) {
    fscanf(filePtr, "%f ", &weights8[i]);
  }

   try 
   {
      /* Query for platforms */
      std::vector<cl::Platform> platforms;
      cl::Platform::get(&platforms);

      /* Get a list of devices on this platform */
      std::vector<cl::Device> devices;
      platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

      /* Create a context for the devices */
      cl::Context context(devices);
      
      /* Create a command queue for the first device */
      cl::CommandQueue queue = cl::CommandQueue(context, devices[0]);

      //Allocate Buffer in Global Memory
      cl::Buffer buffer_warr (context, CL_MEM_READ_ONLY,
              71*256*sizeof(float));
      cl::Buffer buffer_uarr (context, CL_MEM_READ_ONLY,
              64*256*sizeof(float));
      cl::Buffer buffer_barr (context, CL_MEM_READ_ONLY,
              256*sizeof(float));
      cl::Buffer buffer_xt (context, CL_MEM_READ_ONLY,
              1*71*sizeof(float));
      cl::Buffer buffer_htemp (context, CL_MEM_READ_WRITE,
              1*64*sizeof(float));
      cl::Buffer buffer_ctemp (context, CL_MEM_READ_WRITE,
              1*64*sizeof(float));
      // cl::Buffer buffer_output(context, CL_MEM_WRITE_ONLY,
      //         1*64*sizeof(float));

    queue.enqueueWriteBuffer(buffer_warr, CL_TRUE, 0, 71*256*sizeof(float), weights1, NULL, NULL);
    queue.enqueueWriteBuffer(buffer_uarr, CL_TRUE, 0, 64*256*sizeof(float), weights2, NULL, NULL);
    queue.enqueueWriteBuffer(buffer_barr, CL_TRUE, 0, 1*256*sizeof(float), weights3, NULL, NULL);

    /* Read the program source */
    std::ifstream sourceFile("kernel.cl");
    std::string sourceCode(
       std::istreambuf_iterator<char>(sourceFile),
       (std::istreambuf_iterator<char>()));
    cl::Program::Sources source(1,
       std::make_pair(sourceCode.c_str(),
       sourceCode.length() + 1));
    
    //  Make program from the source code 
    cl::Program program = cl::Program(context, source);
    
    // /* Build the program for the devices */
    program.build(devices);
    
    // /* Create the kernel */
    cl::Kernel kernel(program, "LSTM_encoder");
    cl::Kernel kernel2(program, "LSTM_decoder");

  FILE *filePtr;
  filePtr = fopen("test_samples.txt", "r");
  // float *test_data = (float *) malloc(sizeof(float)*71*256);
  
  // int i = 0;
  // char str[1000];
  // char* input_string = "Hello!         ";

  // char **test_total = (char **) malloc(sizeof(char)*16*1000);
  // for (int i=0; i<10; i++) {
  //   //for (int j=0; j<16; j++) {
  //   fgets(filePtr, "%s\n", &test_total[i]);
  //   // if (test_total[i][j] != '\n') {
  //   //   break;
  //   // }      
  //   //}
  // }

FILE * fp;
char * line = NULL;
size_t len = 0;
ssize_t read;
char * line1 = NULL;
size_t len1 = 0;
ssize_t read1;
fp = fopen("test_samples.txt", "r");
FILE *fp1 = fopen("test_actual.txt", "r");
int correct = 0;
int total = 0;
float acc = 0.0;

while ((read = getline(&line, &len, fp)) != -1 && (read1 = getline(&line1, &len1, fp1)) != -1) {

    printf("Input sentence: %s", line);
    printf("Expected sentence: %s\n", line1);

  for (int i=read-1; i<16; i++) {
    line[i] = ' ';
  }

    int max_encoder_time = 16;

    clock_t start, end;
    start = clock(); 
    for (int i=0; i<max_encoder_time; i++) {
      //char LSTM_inp = input_string[i];
      //printf("%s\n", line);
      //printf("%c\n", line[i]);
      char c = line[i];
      float* LSTM_input = (float*) malloc(sizeof(float)*71);
      float* LS_input = (float*) malloc(sizeof(float)*71);
      encode_input(c, LSTM_input);
      //encode_input(c, LS_input);

      queue.enqueueWriteBuffer(buffer_xt, CL_TRUE, 0, 1*71*sizeof(float), LSTM_input);

        if (i == 0) {
       float *weights_ht = (float *) malloc(sizeof(float)*1*64);
       float *weights_ct = (float *) malloc(sizeof(float)*1*64);
           for (int i=0; i<64; i++) {
             weights_ht[i] = 0;
             weights_ct[i] = 0;
           }
        queue.enqueueWriteBuffer(buffer_htemp, CL_TRUE, 0, 1*64*sizeof(float), weights_ht);
        queue.enqueueWriteBuffer(buffer_ctemp, CL_TRUE, 0, 1*64*sizeof(float), weights_ct);
        }

        //Call encoder function
        //Set the Kernel Arguments
        int nargs=0;

        kernel.setArg(0, buffer_warr);
        kernel.setArg(1, buffer_uarr);
        kernel.setArg(2, buffer_barr);
        kernel.setArg(3, buffer_xt);
        kernel.setArg(4, buffer_ctemp);
        kernel.setArg(5, buffer_htemp);
        kernel.setArg(6, 71);
        kernel.setArg(7, 64);

      cl::NDRange global(1, 1);
      cl::NDRange local(1, 1);
      queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);

    }

      /* Copy the output data back to the host */

      //Allocate Buffer in Global Memory
      cl::Buffer buffer_warr_dec (context, CL_MEM_READ_ONLY,
              84*256*sizeof(float));
      cl::Buffer buffer_uarr_dec (context, CL_MEM_READ_ONLY,
              64*256*sizeof(float));
      cl::Buffer buffer_barr_dec (context, CL_MEM_READ_ONLY,
              256*sizeof(float));
      cl::Buffer buffer_bias (context, CL_MEM_READ_ONLY,
              84*sizeof(float));
      cl::Buffer buffer_dense (context, CL_MEM_READ_ONLY,
              64*84*sizeof(float));

      cl::Buffer buffer_xt_dec (context, CL_MEM_READ_ONLY,
              1*84*sizeof(float));
      cl::Buffer buffer_output(context, CL_MEM_WRITE_ONLY,
              1*84*sizeof(float));


    queue.enqueueWriteBuffer(buffer_warr_dec, CL_TRUE, 0, 84*256*sizeof(float), weights4, NULL, NULL);
    queue.enqueueWriteBuffer(buffer_uarr_dec, CL_TRUE, 0, 64*256*sizeof(float), weights5, NULL, NULL);
    queue.enqueueWriteBuffer(buffer_barr_dec, CL_TRUE, 0, 1*256*sizeof(float), weights6, NULL, NULL);
    queue.enqueueWriteBuffer(buffer_dense, CL_TRUE, 0, 64*84*sizeof(float), weights7, NULL, NULL);
    queue.enqueueWriteBuffer(buffer_bias, CL_TRUE, 0, 1*84*sizeof(float), weights8, NULL, NULL);

    int max_decoder_time = 45;
    char output[45];
    int k= 0;
int u = 0;
    for (int i=0; i<max_decoder_time; i++) {
          // int u = 0;
      //char LSTM_inp = input_string[i];
      float *w2 = (float *) malloc(sizeof(float)*1*84);
      queue.enqueueReadBuffer(buffer_xt_dec, CL_TRUE, 0, 1*84*sizeof(float),
           w2);

      if (i == 0) {
      float* LSTM_input = (float*) malloc(sizeof(float)*84);
      for (int i=0; i<84; i++) {
        LSTM_input[i] = 0;
      }
      LSTM_input[0] = 1;
      queue.enqueueWriteBuffer(buffer_xt_dec, CL_TRUE, 0, 1*84*sizeof(float), LSTM_input);

      }
        int nargs=0;

        kernel2.setArg(0, buffer_warr_dec);
        kernel2.setArg(1, buffer_uarr_dec);
        kernel2.setArg(2, buffer_barr_dec);
        kernel2.setArg(3, buffer_bias);
        kernel2.setArg(4, buffer_dense);
        kernel2.setArg(5, buffer_output);
        kernel2.setArg(6, buffer_xt_dec);
        kernel2.setArg(7, buffer_ctemp);
        kernel2.setArg(8, buffer_htemp);
        kernel2.setArg(9, 84);
        kernel2.setArg(10, 64);

      cl::NDRange global(1, 1);
      cl::NDRange local(1, 1);
      queue.enqueueNDRangeKernel(kernel2, cl::NullRange, global, local);

      float *weigh = (float *) malloc(sizeof(float)*1*84);
      queue.enqueueReadBuffer(buffer_output, CL_TRUE, 0, 1*84*sizeof(float),
           weigh);

      float *weigh2 = (float *) malloc(sizeof(float)*1*64);
      queue.enqueueReadBuffer(buffer_htemp, CL_TRUE, 0, 1*64*sizeof(float),
           weigh2);

      float m = -INFINITY;
      for (size_t i = 0; i < 84; i++) {
        if (weigh[i] > m) {
          m = weigh[i];
        }
      }
    
      float ymax = m;
      for(int i = 0; i < 84; i++) {
        weigh[i] = exp(weigh[i] - ymax);
      }

      float r = 0;
      for (int i=0; i<84; i++) {
        r += weigh[i];
      }

      for (int i=0; i<84; i++) {
        weigh[i] = weigh[i]/r;
      }


      float max_val = 0.0;
      int index = 1;
      for (int i=0; i<84; i++) {
       if (weigh[i]>max_val) {
         index = i;
         max_val = weigh[i];
       }
      }

      for (int i=0; i<84; i++) {
       if (i != index) {
         weigh[i] = 0;
       }
      }
      weigh[index] = 1;
     
      char output_char = decode_output(weigh);
      //printf("%c", output_char);

      if (output_char == '\n') {
        break;
      }
      output[k++] = output_char;
      //printf("%c\n", output_char);

    queue.enqueueWriteBuffer(buffer_xt_dec, CL_TRUE, 0, 1*84*sizeof(float), weigh);
    }
    end = clock();
    output[k++] = '\0';
    printf("%s\n", output);


    while (line1[u] != '\n') {
      if (line1[u] == output[u]) {
        correct += 1;
      }
      total+=1;
      u += 1;
    }
    // Calculating total time taken by the program. 
    double time_taken = double(end - start) / double(CLOCKS_PER_SEC); 
    std::cout << "Time taken by program on device : " << std::fixed  
         << time_taken; 
    std::cout << " sec \n" << std::endl; 

   }

   acc = ((float)correct/(float)total);
   printf("Accuracy:%f\n", 1 - acc);
}
   catch(cl::Error error)
   {
      std::cout << error.what() << "(" << error.err() << ")" << std::endl;
   }

   return 0;
}
