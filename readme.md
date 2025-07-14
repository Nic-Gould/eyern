// TODO
* deciding on how to pass params to inner funtions from outerfunctions in main (e.g. C3 calling conv)
* new setters and getters if merging data classes. (maybe I just won't and I'll have 3 similar structs...)
* type checking
* finish detection function impementation

GOALS
A fun project to help me learn rust. The idea is to build YoloV5 (detection only) from scratch in rust. From scratch to me means standard library only.
I've used an image library to load the images as I don't consider loading images part of the scope anymore than I consider openCV part of YOLO.
The program will load and process a single image, outputting the results to a text file. 

NEXT STEPS
Once operational I hope to add support for reading from a MJPEG stream, and hopefully outpuuting bounding boxes onto the image. I also imagine I'll use external crates for these functions.     
  
   
    
