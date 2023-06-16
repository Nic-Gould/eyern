
mod functions;
mod weights;




// The data pub struct is an output from each layer, and is used as an input in the subsequent layer.
// On a few occasions the data is used in later layers and needs to be kept.
// mutable access to a data pub struct is only required during creation
// all other accesses can use read only values.

fn main(){

let num_classes = 80;
let max_nms = 30000;
let conf_thres = 0.25; 
let iou_thres = 0.45; 
let max_det = 300;
  //weights= "yolov5s.pt",              // model.pt path(s)
let source = "data/images/cat.jpg";     // file/dir/URL/glob, 0 for webcam
let imgsz = 640;                         // inference size (pixels)
let out_path= "detections.txt";        

    //Layer Zero
    let img= functions::read_image(source);
    let weights = weights::layer_0::conv_weight;
    let output = functions::conv(3, 32, 6, 2,2,&weights, &img);
    let input = output; 

    //First Layer
    let weights = weights::layer_1::conv_weight;
    let output = functions::conv(3, 32, 6, 2,2,&weights, &img);
    let input = output;       
  
    //Second Layer
    let weights = weights::layer_2::conv_weight;
    let input = output;    
    let output = functions::C3(64, 64, 1,1,0, &weights, &input);
          
  
    //Third Layer
    let weights = weights::layer_3::conv_weight;
    let input = output;    
    let output = functions::conv(64, 128, 3,2,1,&weights, &input);
          
      
    //Forth Layer
    let weights = weights::layer_4::conv_weight;
    let input = output;    
    let output = functions::C3(128,128,1,1,0,&weights, &input);
          
  
    //Fifth Layer
    let weights = weights::layer_5::conv_weight;
    let input = output;    
    let output = functions::conv(128, 256, 3,2,1,&weights, &input);
          
        
    //Layer(6)
    let weights = weights::layer_6::conv_weight;
    let input = output;    
    let output = functions::C3(256,256,1,1,0,&weights, &input);


    //Layer(7)
    let weights = weights::layer_7::conv_weight;
    let input = output;    
    let output =  functions::conv(256, 512, 1,1,0,&weights, &input);
          
      
    //Layer(8)
    let weights = weights::layer_8::conv_weight;
    let input = output;    
    let output = functions::C3(512,512,1,1,0,&weights, &input);
    

    //Layer(9)
    let output = functions::sppf(512,512,1,1,0,&weights, &input);
         //  functions::maxpool(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
    

    //Layer(10)
    let weights = weights::layer_10::conv_weight;
    let input = output;    
    let output = functions::conv(512, 256, 1,1,0,&weights, &input);
    

    //Layer(11)
    functions::upsample(scale_factor=2.0, mode=nearest, &output, theotherone);


    //Layer(12)
    functions::concat(&output, theotherone)


    //Layer(13)
    let weights = weights::layer_13::conv_weight;
    let input = output;    
    let output = functions::C3(512,256,1,1,0,&weights, &input);
       

    //Layer(14)
    let weights = weights::layer_14::conv_weight;
    let input = output;    
    let output = functions::conv(256, 128, 1,1,0,&weights, &input);
    
    
    //Layer(15)
    functions::upsample(scale_factor=2.0, mode=nearest, &output, theotherone);


    //Layer(16)
    functions::concat(&output, theotherone);
  

    //Layer(17)
    let weights = weights::layer_17::conv_weight;
    let input = output;    
    let output = functions::C3(256,128, 1,1,0,&weights, &input)
         
      
    //Layer(18)
    let weights = weights::layer_18::conv_weight;
    let input = output;    
    let output = functions::conv(128, 128, 3,2,1,&weights, &input);
         
   
    //Layer(19)
    functions::concat(&output,theotherone);
  

    //Layer(20)
    let weights = weights::layer_20::conv_weight;
    let input = output;    
    let output = functions::C3(256,256,1,1,0,&weights, &input);
    

    //Layer(21)
    let weights = weights::layer_21::conv_weight;
    let input = output;    
    let output = functions::conv(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          
          
    //Layer(22)
    functions::concat(&output. theotherone);


    //Layer(23)
    let weights = weights::layer_23::conv_weight;
    let input = output;    
    let output = functions::C3(512,512,1,1,0,&weights, &input);
  

    //Layer(24)
    detect(
    
    let weights = weights::layer_24::conv_weight;
    let input = output;    
    let output = functions::conv(128, 255, 1,1,0,&weights, &input)
    
    let weights = weights::layer_24::conv_weight;
    let input = output;    
    let output = functions::conv(256, 255, 1,1,0,&weights, &input)
    
    let weights = weights::layer_24::conv_weight;
    let input = output;    
    let output = functions::conv(512, 255, 1,1,0,&weights, &input)
            
    )
        
      
    
  
  }
  
  