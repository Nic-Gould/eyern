// TODO

The model is composed of files containing a variety of lists/arrays/tensors such as

	- the module/layer names (i.e. what functions and in what order)
	- the parameters for these modules
	- The size and shape of the output tensors required for each module (so a statically allocated array can be created)
	- The weights tensors required for each module

 all of the following are passed as params anyway. No need to include a tensor size array
Kernal_size = tensor_size[1]
Kernal_size = tensor_size[2]
Input_layers=tensor_size[3]
Output_layers=tensor_size[4]


	
On initialisation the model
	Acquires the input data
	Loads the weights for the module
	and creates a blank output tensor of the right dimensions
	Reads the first module name, the parameters, and creates a blank output tensor of the right dimensions
	
	
To proceed the model
	Perfoms the required function on the data as defined by the module
	Loads and initialises the next module
	Passes the data to that module as an input.
    For inference on a known system, the size of all data structures is known in advance, as are all paramaters and their sizes. 

    A struct is created to contain the weights for each model layer. On systems with more memory these could all be created at once. For embedded systems (ref design esp32s3) the unzipped tensor data exceeds the total device storage let alone the ram.
    
    A struct is created to contain the output data for each layer. 
    For the first layer, the input data is read from the frame buffer.
    For all subsequent layers, the input data is the output data from the previous model layer (with some skips etc which require an extra data struct).
    
    Initialisation params for the structs such as the tensor dimensions can be read from file, input as a param, or calculated from .length type methods.
    At the moment I'm leaning towards reading everything from file.
    
    As such all of the above listed sizes and parameters will have to be determined.
    
    
Levels of difficulty in this project

-Trying to figure out how to use existing crates and modules for this project [failed]
-Trying to initialise an array with data [failed]
-Reading abstracted python functions and pytorch documentation to figure out what the hell is happening in the Yolov5 model [success]
-Exporting (python) and importing (rust) the data [success]
-The  (general) ML principles   [success]
-The Matrix math and functions [success]