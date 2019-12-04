//ENCODER
// #include <iostream>

// using namespace std;
#define LOCAL_MEM_SIZE  256

float sigmoid_(float x) {
  
 float t = 1.0 + exp(-1*x);
 return 1.0/t;    
}

// kernel __attribute__ ((reqd_work_group_size(1, 1, 1)))

//SIGMOID FUNCTION
float sigmoid(float x) {
	float y = 0;
	if(x > -1 && x<1) {
	y = 0.238*x + 0.5000;
	}
	else if(x >=-2 && x<= -1) {
	y = 0.0467* x*x + 0.1239*x + 0.2969;
	} 
	else if(x>= -3 && x<-2) {
	y = 0.0298*x*x + 0.2202*x + 0.4400;
    }
	else if(x>=-4 && x< -3) {
	y = 0.0135 * x*x + 0.1239*x + 0.2969;
    }
	else if(x>= -5 && x<-4) {
	y = 0.0054*x*x + 0.0597*x + 0.1703;
	}	
	else if(x>=1 && x<2) {
	y = -0.0467*x*x + 0.2896*x + 0.4882;
    }
	else if(x>=2 && x<3) {
 	y = -0.0298*x*x + 0.2202*x + 0.56;
    }
	else if(x>=3 && x<4) {
	y = -0.0135* x*x + 0.1239*x + 0.7030;
    }
	else if(x>=4 && x<5) {
	y = -0.0054*x*x +0.0597*x + 0.8297;
    }
	else if (x < -5) {
	y = 0.003;
    }
	else if (x>= 5) {
	y = 0.9990;
    }
	return y;
 }

float tanh_copy(float x) 
{
    float x_temp = 2*x;
    x_temp = sigmoid(x_temp);
	return (2*(x_temp) - 1); 
}

__kernel
void LSTM_encoder(
        __global float *warr, // Read-Only Vector 1
        __global float *uarr, // Read-Only Vector 1
        __global float *barr, // Read-Only Vector 2
        __global float *xt,
        __global float *ct,
        __global float *ht, // Output Result
        int nfeatures,                   // Size in integer
        int hunits
)
{
    float v1_local[LOCAL_MEM_SIZE];    // Local memory to store vector1
    float result_local[LOCAL_MEM_SIZE];// Local Memory to store result
    float xt_out[LOCAL_MEM_SIZE];
	float forget_gate[64], input_gate[64], out_gate[64], gate_gate[64];
    float ht_out[LOCAL_MEM_SIZE];
    float s_t[LOCAL_MEM_SIZE];

    for(int i=0;i<LOCAL_MEM_SIZE;i++) {
		float t = 0.0;
		for(int j=0;j<71;j++) {
			//printf("%s\n","Hello" );
			t += *(xt + j) * *(warr + i + (LOCAL_MEM_SIZE*j));
			
		}
		xt_out[i] = t;
		//printf("%f\n", xt_out[i]);
   }

   for(int i=0;i<LOCAL_MEM_SIZE;i++) {
   		float t = 0.0;
	   	for(int j=0;j<hunits;j++) {
	   		t += *(ht + j) * *(uarr + i + (LOCAL_MEM_SIZE*j));
	   	}
	   	ht_out[i] = t;
    }

	for(int i=0;i<LOCAL_MEM_SIZE;i++)
	{
		s_t[i] = xt_out[i] + ht_out[i] + barr[i];
	}

	int k1= 0;
	int k2 = 0;
	int k3 = 0;
	int k4 = 0;

	for (int j = 0; j<LOCAL_MEM_SIZE; j++)
	{
		if(j<64) {
		//printf("%f\n", input_gate[k1]);
		input_gate[k1] = sigmoid(s_t[j]);
		//printf("%f\n", input_gate[k1]);
		k1 = k1 + 1;
		}
		else if (j>=64 && j< 128) {
		forget_gate[k2] = sigmoid(s_t[j]);
		k2 = k2 + 1;
		}
		else if (j>=128 && j<192) {
		gate_gate[k3++] = tanh_copy(s_t[j]);
		}
		else { 
			out_gate[k4++] = sigmoid(s_t[j]);
		}
	}

	for(int i=0;i<64;i++)
	{   
		float temp = ct[i];
			// printf("%f\n", ct[i]);
		*(ct + i) = (input_gate[i]* gate_gate[i]) + (forget_gate[i]* temp);
		*(ht + i) = out_gate[i]* tanh_copy(*(ct+i));
	}
}
	
__kernel	
void LSTM_decoder(
        __global float *warr, // Read-Only Vector 1
        __global float *uarr, // Read-Only Vector 1
        __global float *barr, // Read-Only Vector 2
        __global float *bias, //dense layer bias weights
        __global float *dense_weights, // dense layer
        __global float *out, //output(dense layer)
        __global float *xt,
        __global float *ct,
        __global float *ht,
        // __global float *o_t,  //output
        int nfeatures,                   // Size in integer
        int hunits
)
{

    float v1_local[LOCAL_MEM_SIZE];    // Local memory to store vector1
    float result_local[LOCAL_MEM_SIZE];// Local Memory to store result
    float xt_out[LOCAL_MEM_SIZE];
	float forget_gate[64], input_gate[64], out_gate[64], gate_gate[64];
    float ht_out[LOCAL_MEM_SIZE];
    float s_t[LOCAL_MEM_SIZE];

    for(int i=0;i<LOCAL_MEM_SIZE;i++) {
		float t = 0.0;
		for(int j=0;j<71;j++) {
			t += *(xt + j) * *(warr + i + (LOCAL_MEM_SIZE*j));
			
		}
		xt_out[i] = t;
	}

   for(int i=0;i<LOCAL_MEM_SIZE;i++) {
   		float t = 0.0;
	   	for(int j=0;j<hunits;j++) {
	   		t += *(ht + j) * *(uarr + i + (LOCAL_MEM_SIZE*j));
	   	}
	   	ht_out[i] = t;
    }

	for(int i=0;i<LOCAL_MEM_SIZE;i++)
	{
		s_t[i] = xt_out[i] + ht_out[i] + barr[i];
		//printf("%f\n", s_t[i]);
	}

	int k1= 0;
	int k2 = 0;
	int k3 = 0;
	int k4 = 0;

	for (int j = 0; j<LOCAL_MEM_SIZE; j++)
	{
		if(j<64) {
		//printf("%f\n", input_gate[k1]);
		input_gate[k1] = sigmoid(s_t[j]);
		//printf("%f\n", input_gate[k1]);
		k1 = k1 + 1;
		}
		else if (j>=64 && j< 128) {
		forget_gate[k2] = sigmoid(s_t[j]);
		k2 = k2 + 1;
		}
		else if (j>=128 && j<192) {
		gate_gate[k3++] = tanh_copy(s_t[j]);
		}
		else { 
			out_gate[k4++] = sigmoid(s_t[j]);
		}
	}

	for(int i=0;i<64;i++)
	{   
		float temp = ct[i];
			// printf("%f\n", ct[i]);
		*(ct + i) = (input_gate[i] * gate_gate[i]) + (forget_gate[i]* temp);
		//printf("%f\n", ct[i]);
		*(ht + i) = out_gate[i]* tanh_copy(*(ct+i));
		out[i] = out_gate[i];
		//printf("%f\n", ht[i]);
	}

//Dense layer implementation
for(int j=0; j<84 ; j++ )     //since german vocab = 84
{
 float sum = 0.0;	
  for(int i=0; i<64; i++)
	{
	//float val = 0.0;
	sum += ht[i]* *(dense_weights + i*84 + j);  //out_gate(1*64) dense_weights = dense1_0(64*84)
	}
	//sum = sum + *(bias + j);        //bias = dense1_1(1*84)
   out[j] = sum + bias[j];
}

}