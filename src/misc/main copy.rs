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

//First Layer
    let img= functions::read_image(source);
    let weights = weights::layer_0::conv_weight;
    let output = functions::conv(3, 32, 6, 2,2,&weights, &img);
    let input = output;       
  
  //Second Layer
    let weights = weights::layer_1::conv_weight;
    let input = output;    
    let out = C3(64, 64, 1,1,0, &weights, &input)
          (cv1):
          (conv): functions::conv(64, 32, kernel_size=(1, 1), stride=(1, 1))
          
          )
          (cv2):
          (conv): functions::conv(64, 32, kernel_size=(1, 1), stride=(1, 1))
              )(cv3):
          (conv): functions::conv(64, 64, kernel_size=(1, 1), stride=(1, 1))
          )
          
          
          )
          (m): Sequential(
          (0): Bottleneck(
              (cv1):
              (conv): functions::conv(32, 32, kernel_size=(1, 1), stride=(1, 1))
              
              )
              (cv2):
              (conv): functions::conv(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              
              )
          )
      
      )
  
      //Third Layer
  
      (3):
          (conv): functions::conv(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          
      )
  
      //Forth Layer
  
      (4): C3(
          (cv1):
          (conv): functions::conv(128, 64, kernel_size=(1, 1), stride=(1, 1))
          
          )
          (cv2):
          (conv): functions::conv(128, 64, kernel_size=(1, 1), stride=(1, 1))
          
          )
          (cv3):
          (conv): functions::conv(128, 128, kernel_size=(1, 1), stride=(1, 1))
          
          )
          (m): Sequential(
          (0): Bottleneck(
              (cv1):
              (conv): functions::conv(64, 64, kernel_size=(1, 1), stride=(1, 1))
              
              )
              (cv2):
              (conv): functions::conv(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              
              )
          )
          (1): Bottleneck(
              (cv1):
              (conv): functions::conv(64, 64, kernel_size=(1, 1), stride=(1, 1))
              
              )
              (cv2):
              (conv): functions::conv(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              
              )
          )
          )
      )
  
      //Fifth Layer
  
      (5):
          (conv): functions::conv(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          
      )
      
      //Layer(6)
      C3(
          (cv1):
          (conv): functions::conv(256, 128, kernel_size=(1, 1), stride=(1, 1))
          
          )
          (cv2):
          (conv): functions::conv(256, 128, kernel_size=(1, 1), stride=(1, 1))
          
          )
          (cv3):
          (conv): functions::conv(256, 256, kernel_size=(1, 1), stride=(1, 1))
          
          )
          (m): Sequential(
          (0): Bottleneck(
              (cv1):
              (conv): functions::conv(128, 128, kernel_size=(1, 1), stride=(1, 1))
              
              )
              (cv2):
              (conv): functions::conv(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              
              )
          )
          (1): Bottleneck(
              (cv1):
              (conv): functions::conv(128, 128, kernel_size=(1, 1), stride=(1, 1))
              
              )
              (cv2):
              (conv): functions::conv(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              
              )
          )
          (2): Bottleneck(
              (cv1):
              (conv): functions::conv(128, 128, kernel_size=(1, 1), stride=(1, 1))
              
              )
              (cv2):
              (conv): functions::conv(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              
              )
          )
          )
      )
      //Layer(7)
      (7):
          (conv): functions::conv(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          
      )
      
      //Layer(8)
      (8): C3(
          (cv1):
          (conv): functions::conv(512, 256, kernel_size=(1, 1), stride=(1, 1))
          
          )
          (cv2):
          (conv): functions::conv(512, 256, kernel_size=(1, 1), stride=(1, 1))
          
          )
          (cv3):
          (conv): functions::conv(512, 512, kernel_size=(1, 1), stride=(1, 1))
          
          )
          (m): Sequential(
          (0): Bottleneck(
              (cv1):
              (conv): functions::conv(256, 256, kernel_size=(1, 1), stride=(1, 1))
              
              )
              (cv2):
              (conv): functions::conv(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              
              )
          )
          )
      )
      
      //Layer(9)
      (9): SPPF(
          (cv1):
          (conv): functions::conv(512, 256, kernel_size=(1, 1), stride=(1, 1))
          
          )
          (cv2):
          (conv): functions::conv(1024, 512, kernel_size=(1, 1), stride=(1, 1))
          
          )
          (m): MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
      )
      //Layer(10)
      (10):
          (conv): functions::conv(512, 256, kernel_size=(1, 1), stride=(1, 1))
          
      )
      
      //Layer(11)
      (11): Upsample(scale_factor=2.0, mode=nearest)
      
      //Layer(12)
      (12): Concat()
      
      //Layer(13)
      (13): C3(
          (cv1):
          (conv): functions::conv(512, 128, kernel_size=(1, 1), stride=(1, 1))
          
          )
          (cv2):
          (conv): functions::conv(512, 128, kernel_size=(1, 1), stride=(1, 1))
          
          )
          (cv3):
          (conv): functions::conv(256, 256, kernel_size=(1, 1), stride=(1, 1))
          
          )
          (m): Sequential(
          (0): Bottleneck(
              (cv1):
              (conv): functions::conv(128, 128, kernel_size=(1, 1), stride=(1, 1))
              
              )
              (cv2):
              (conv): functions::conv(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              
              )
          )
          )
      )
      
      //Layer(14)
      (14):
          (conv): functions::conv(256, 128, kernel_size=(1, 1), stride=(1, 1))
          
      )
      //Layer(15)
      (15): Upsample(scale_factor=2.0, mode=nearest)
      
      //Layer(16)
      (16): Concat()
  
      //Layer(17)
      (17): C3(
          (cv1):
          (conv): functions::conv(256, 64, kernel_size=(1, 1), stride=(1, 1))
          
          )
          (cv2):
          (conv): functions::conv(256, 64, kernel_size=(1, 1), stride=(1, 1))
          
          )
          (cv3):
          (conv): functions::conv(128, 128, kernel_size=(1, 1), stride=(1, 1))
          
          )
          (m): Sequential(
          (0): Bottleneck(
              (cv1):
              (conv): functions::conv(64, 64, kernel_size=(1, 1), stride=(1, 1))
              
              )
              (cv2):
              (conv): functions::conv(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              
              )
          )
          )
      )
      
      //Layer(18)
      (18):
          (conv): functions::conv(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          
      )
      //Layer(19)
      (19): Concat()
  
      //Layer(20)
      (20): C3(
          (cv1):
          (conv): functions::conv(256, 128, kernel_size=(1, 1), stride=(1, 1))
          
          )
          (cv2):
          (conv): functions::conv(256, 128, kernel_size=(1, 1), stride=(1, 1))
          
          )
          (cv3):
          (conv): functions::conv(256, 256, kernel_size=(1, 1), stride=(1, 1))
          
          )
          (m): Sequential(
          (0): Bottleneck(
              (cv1):
              (conv): functions::conv(128, 128, kernel_size=(1, 1), stride=(1, 1))
              
              )
              (cv2):
              (conv): functions::conv(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              
              )
          )
          )
      )
  
      //Layer(21)
      (21):
          (conv): functions::conv(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          
      )
  
      //Layer(22)
      (22): Concat()
      
      //Layer(23)
      (23): C3(
          (cv1):
          (conv): functions::conv(512, 256, kernel_size=(1, 1), stride=(1, 1))
          
          )
          (cv2):
          (conv): functions::conv(512, 256, kernel_size=(1, 1), stride=(1, 1))
          
          )
          (cv3):
          (conv): functions::conv(512, 512, kernel_size=(1, 1), stride=(1, 1))
          
          )
          (m): Sequential(
          (0): Bottleneck(
              (cv1):
              (conv): functions::conv(256, 256, kernel_size=(1, 1), stride=(1, 1))
              
              )
              (cv2):
              (conv): functions::conv(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              
                )
              )
            )
          )
  
  //Layer(24)
          (24): Detect(
            (m): ModuleList(
              (0): functions::conv(128, 255, kernel_size=(1, 1), stride=(1, 1))
              (1): functions::conv(256, 255, kernel_size=(1, 1), stride=(1, 1))
              (2): functions::conv(512, 255, kernel_size=(1, 1), stride=(1, 1))
            )
          )
        
      
    
  
  }
  
  