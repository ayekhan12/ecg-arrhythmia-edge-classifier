//==========================================================
//Definitions and Libraries
//============================================================

// Clear potential conflicts with TensorFlow and Arduino definitions
#undef abs
#undef min
#undef max

#include <cmath>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"

//include the model
#include "model_data.h"

//Code to use time short cuts, used later for setting time to look for new data values
using namespace std::chrono_literals;

// load the board's OS hardware layer
#include "mbed.h"
#include "mbed_stats.h"

//================================================================
//Variables
//============================================================


//counter for when to  print our stats
int windowcount = 0;

// declares the size of the window and shift, 360 samples per second
const int windowsize = 1080;
const int shiftsize = 360;

//global variables for model: the model, interpreter, and model input and outputs
const tflite::Model* tf_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output = nullptr;

//===================================================================
//Data Storage
//=====================================================================================

// this buffer holds the data until processed
mbed::CircularBuffer<int8_t, windowsize> fullwindowbuffer;

// this line creates the array to store the data for processing, need a different variable since ringbuffer will be changing too fast
int8_t analysiswindow[windowsize];

// Flag to tell the main loop that a fresh 1080-sample window is ready for AI inference
volatile bool windowReadyToProcess = false;


//storage created for model to stick all of the math during run time
constexpr int kTensorArenaSize = 90 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];


//=================================================================
//Timing Thread
//================================================================

// Elements to enforce 360 hz data collection
mbed::LowPowerTicker sampleTicker;
events::EventQueue sampleQueue(32 * EVENTS_EVENT_SIZE);

// Sets thread priority above main loop, how to ensure this runs on time
rtos::Thread realTimeThread(osPriorityRealtime); 


// This function is guaranteed to run precisely at 360 Hz (every 2778 us)
void readSampleTask() {
    // Safely reads the hardware serial interface, using int 8 hence 1 bytes
    if (Serial.available() >= 1) {
        int8_t sample;
        Serial.readBytes((char*)&sample, 1);
        
        if (!fullwindowbuffer.full()) {
            fullwindowbuffer.push(sample);
        }
    }

    // Check if our sliding window hit 1080 samples
    if (fullwindowbuffer.full() && !windowReadyToProcess) {
        
        // Copy the entire buffer out to our analysis array
        for (int i = 0; i < windowsize; i++) {
            fullwindowbuffer.pop(analysiswindow[i]);
        }

        // Refill the buffer with the oldest 720 samples to maintain the sliding window
        for (int i = shiftsize; i < windowsize; i++) {
            fullwindowbuffer.push(analysiswindow[i]);
        }

        // Notify the main loop that the data is locked and ready for inference
        windowReadyToProcess = true;
    }
}


//====================================================================
//Metric Evaluation
//===================================================================


//This code chunk is used to track the ram usage, called every 15 seconds
void printram() {
  mbed_stats_heap_t heap_stats;
  mbed_stats_heap_get(&heap_stats);

  mbed_stats_stack_t stack_stats;
  mbed_stats_stack_get(&stack_stats);

  size_t arena_used = interpreter->arena_used_bytes();

  
  Serial.print(F("Heap peak:   ")); Serial.print(heap_stats.max_size); Serial.println(F(" bytes"));
  Serial.print(F("Stack peak:  ")); Serial.print(stack_stats.max_size);
  Serial.print(F(" / ")); Serial.print(stack_stats.reserved_size); Serial.println(F(" bytes"));
  Serial.print(F("Arena used:  ")); Serial.print(arena_used);
  Serial.print(F(" / ")); Serial.print(kTensorArenaSize); Serial.println(F(" bytes"));

  size_t total_used = heap_stats.max_size + stack_stats.max_size + arena_used;
  Serial.print(F("Total (approx): ")); Serial.print(total_used); Serial.println(F(" bytes"));
  
}

//=======================================================================
//Model Functions and setup
//===========================================================

//creating the lookup table of math operations interpreter is allowed to run
//doing it this way allows the model code to only include what it actually needs during run time


//AddBatchMatMul is tricky, I had to duct tape a solution together to get it to work, it isn't usually part of this version of arduino software
//how we designed our code shouldn't need it at all, but we couldn't get tfmot to behave, tfmot added Batch_MatMul for an operation that shouldn't need it in our custom multi-head attention code

tflite::MicroMutableOpResolver<13> resolver;

//Setup the model
bool setupModel() {

   //get and store the model object
   tf_model = tflite::GetModel(g_model);

   //check if the model was created using the same tflite schema
   if (tf_model->version() != TFLITE_SCHEMA_VERSION) {
       Serial.println(F("Model schema mismatch"));
       return false;
   }

   //resolver.AddBuiltin(tflite::BuiltinOperator_BATCH_MATMUL,tflite::Register_BATCH_MATMUL(),1 ,2);

   //add math operations to the resolver
   resolver.AddAdd();
   resolver.AddBatchMatMul();
   resolver.AddConv2D();
   resolver.AddDepthwiseConv2D();
   resolver.AddFullyConnected();
   resolver.AddMean();
   resolver.AddMul();
   resolver.AddReshape();
   resolver.AddRsqrt();
   resolver.AddSoftmax();
   resolver.AddSquaredDifference();
   resolver.AddSub();  
   resolver.AddTranspose();
   
   
   
   
   //create the interpreter and store its pointer
   static tflite::MicroInterpreter static_interpreter( tf_model, resolver, tensor_arena, kTensorArenaSize);
   interpreter= &static_interpreter;

   //flag if the tensor_arena is too small
   if (interpreter->AllocateTensors() != kTfLiteOk) {
       Serial.println(F("Allocate Tensor Failed"));
       return false;
   }
 
   model_input = interpreter->input(0);
   model_output = interpreter->output(0);
   return true;

   }
    
//function to run the model

void runModel() {

  
  
  float scale = model_input->params.scale;
   int zero_point = model_input->params.zero_point;

   for (int i = 0; i < windowsize; i++) {
//this is where onboard quantization would occur if necessary
//The Python code feeding the Arduino is handling this
//If sensors are being used in the future, it might have to move onboard again
    
    model_input->data.int8[i] = analysiswindow[i];
    }

    unsigned long t_invoke_start = millis();
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println(F("Invoke() failed"));
        return;
    }
    unsigned long t_invoke_end = millis();
    
    // Output is [1, 2] -- dequantize all class scores
    float out_scale = model_output->params.scale;
    int out_zero_point = model_output->params.zero_point;

    Serial.print(F("Result: "));
    for (int i = 0; i < 2; i++) {
        float result = (model_output->data.int8[i] - out_zero_point) * out_scale;
        Serial.print(result, 4);
        Serial.print(i < 1 ? F(", ") : F("\n"));
    }

    //on device classification, int16_t avoids potential overflow
    //generally this is where a threshold code would go, but it was being finicky
    //instead the board is classifying based on which class has a higher probability
    int16_t delta = (int16_t)model_output->data.int8[1] - (int16_t)model_output->data.int8[0]* out_scale; 
    bool is_abnormal = delta > 0;
    
    Serial.print(F("Classification: "));
    Serial.println(is_abnormal ? F("Abnormal") : F("Normal"));
    
    Serial.print(F("Invoke time (ms): "));
    Serial.println(t_invoke_end - t_invoke_start);
}



//========================================================================
//Setup Code
//==================================================

void setup() {
    Serial.begin(115200);
    
    // Halt code execution until the Serial Monitor is open
    while (!Serial); 
    delay(2000); 
    while (Serial.available() > 0) { Serial.read(); }

    //call the model setup and verify it loaded correctly
   bool model_loaded = setupModel();
   if (model_loaded == false) {
      Serial.println(F("Model setup failed"));
      while (1);
   }
   Serial.println(F("Model loaded"));

   Serial.print(F("Arena used: "));
   Serial.print(interpreter->arena_used_bytes());
   Serial.print(F(" / "));
   Serial.print(kTensorArenaSize);
   Serial.println(F(" bytes"));

    // 1. Start our dedicated high-priority real-time thread
    realTimeThread.start(callback(&sampleQueue, &events::EventQueue::dispatch_forever));

    // 2. Attach the 360 Hz event to the hardware Ticker via our real-time queue
    sampleTicker.attach(sampleQueue.event(&readSampleTask), 2778us);

    //sending an output to the computer, this will tell the computer it is safe to start sending data
    Serial.println('k');
    
    }

//===================================================================
//Main Loop
//=============================================================================

void loop() {
    
    // The realTimeThread will seamlessly interrupt this loop at 360 Hz to pull data.
    
    if (windowReadyToProcess) {
      
        // Code to prevent the 360 hz loop from interupting changing the windowready variable
        core_util_critical_section_enter();
        windowReadyToProcess = false;
        core_util_critical_section_exit();


        // Signal Python that we are ready to receive the next 360 samples, this signaling ensures the two don't get too far out of synch
        Serial.println('k');
        
        //Code to output Metrics
        windowcount++;
        if (windowcount >= 15) {
            printram();
            windowcount = 0;
        }
        
        // Code to run model when window ready
        runModel();





    }
}
