use std::{vec, collections::hash_map};
use image::GenericImageView;
use std::fs::File;
use std::collections::HashMap;
use std::io::Write;

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


// The data pub struct is an output from each layer, and is used as an input in the subsequent layer.
// On a few occasions the data is used in later layers and needs to be kept.
// mutable access to a data pub struct is only required during creation
// all other accesses can use read only values.


#[derive(Debug)]
pub struct Data{                                     
    height:u16,
    width:u16,
    depth:u16,
    count:u32,
    data:Vec<f32>,
} 
impl Data {         //probs make it a trait and hit it with that detection pub struct
    pub fn new (size:[u16;3])->Data{
        Data {
            width:size[0],   //row length
            height:size[1],  // num rows
            depth:size[2],  // num layers
            count:(size[0]*size[1]*size[2]).into(),
            data: vec![0.0; (size[0]*size[1]*size[2]).into()],
        }    
    }
    pub fn get (&self, x:u16, y:u16, z:u16)-> f32{             
        let index:usize = (z * (self.height*self.width) + y*self.width +x).into();
        self.data[index]
    }
    pub fn set (&mut self, x:u16, y:u16, z:u16, exp:f32) {             
        let index:usize = (z * (self.height*self.width) + y*self.width +x).into();
        self.data[index] = exp;
    }  
    pub fn add (&mut self, data:&Data) {
    
        for i in 0..data.height {
            for j in 0..data.width{
                for k in 0..data.depth{           
                    self.set(i,j,k, self.get(i,j,k)+data.get(i,j,k)); // This is real ugly. 
                }
            }
        }
    }



    }

    #[derive(Debug)] 
    pub struct Detections{
        dims: Vec<u16>, //Width x Height x Depth x Batches
        data: Vec<f32>,
    }
    impl Detections{
        pub fn view(&self, new_dims:Vec<u16>)
        {
            if new_dims.into_iter().product()==self.data.len(){ //consume
                self.dims = new_dims;
            }
                else
            {
                panic!("Bro you can't make a view like that, the dimensions are all munted.")
            }
        }
        pub fn new (dims: Vec<u16>)->Detections{
            let length = dims.iter().product();
            Detections { 
                dims,
                data: vec![0.0; length],
            }    
        }
        pub fn get (&self, x:u16, y:u16, z:u16, a:u16 )-> f32{             
            let index:usize = (a* self.dims[0]*self.dims[1]*self.dims[2] 
                            + z * (self.dims[0]*self.dims[1]) 
                            + y*self.dims[0] 
                            + x).into();
            self.data[index]
        }
        pub fn set (&mut self, x:u16, y:u16, z:u16, a:u16, exp:f32) {             
            let index:usize = (a* self.dims[0]*self.dims[1]*self.dims[2] 
                + z * (self.dims[0]*self.dims[1]) 
                + y*self.dims[0] 
                + x).into();
            self.data[index] = exp;
        }  
        pub fn add (&mut self, data:&Detections) {
        
            for i in 0..self.dims[0] {
                for j in 0..self.dims[1]{
                    for k in 0..self.dims[2]{
                        for l in 0..self.dims[3]{           
                        self.set(i,j,k, l, self.get(i,j,k, l)+data.get(i,j,k,l)); // This is real ugly. 
                    }
                    }    
                }
            }
        }
        pub fn permute(&self, new_dims:Vec<u16>){
            if new_dims.len()==self.dims.len(){ 
            let temp = Detections::new(self.dims);
            
                for i in 0..self.dims[0]{
                    for j in 0..self.dims[1]{
                        for k in 0..self.dims[2]{
                            for l in 0..self.dims[3]{
                                temp.set(i,k,l,j, self.get(i,j,k,l));
                            }
                        }
                    }
                }
            }
                else
            {
                panic!("Bro you can't permute like that, the dimensions are all munted.")
            }
        }
    }

    #[derive(Debug)]
    pub struct Weights{
        ksize_h:u16,
        ksize_w:u16,
        ch_in:u16,
        ch_out:u16,
        count:usize,
        data:Vec<f32>,
    }

    impl Weights {
        pub fn new (&self, size:[u16;4], path:&str)->Weights{
            Weights {
                ksize_h:size[0],
                ksize_w:size[1],
                ch_in:size[2],
                ch_out:size[3],
                count:(self.ksize_h * self.ksize_w * self.ch_in * self.ch_out).into(),
                data: vec![0.0; self.count]
            }            
        } 
        pub fn get (&self, x:u16, y:u16, z:u16, a:u16)-> &f32{        
            let index:usize = (a*(self.ksize_h*self.ksize_w*self.ch_in) + z*(self.ksize_h * self.ksize_w) + y*(self.ksize_w) +x).into();
            &self.data[index]
        }          
    }
    pub fn read_image(){
        let path = settings::model_io::source;    
        let im0 = image::open(source).unwrap();  
        //img.dimensions();
        //img.color();
    
    
        // Padded resize
        //img = letterbox(img0, img_size, stride=stride)[0]
    
        // Convert
        //img = img[:, :, ::-1].transpose(2, 0, 1)  // BGR to RGB, to DxWxH
        /*
        img /= 255.0  // 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
    */
    
        let img:Data = Data::new([640,640,3]);      //A data is already filled with zeros
            let vert_padding = (640-480)/2;
            for i in 0..img.height{
                for j in 0..img.width{
                    for k in 0..img.depth{
                        for pixel in im0.pixels(){
                            let colour_value = pixel.2[k.into()] as f32/255.0; // RGBA value is third in struct, k iterates over the RGB values.
                            img.set(i+vert_padding,j,k, colour_value) ;
                        }    
                    }
                }
            }
            
    }

    pub fn conv(in_channels:u16, out_channels:u16, ksize:u16, s:u16, p:u16, d:u16, weights:&Weights, data:&Data) -> Data{
        
        let out_height = ((data.height+2*p-d*(ksize-1) )-1/s)+1;
        let out_width = ((data.width+2*p-d*(ksize-1) )-1/s)+1;
        let out_size:[u16;3] = [out_channels,  out_height,  out_width];
        let num_strides_high = data.height/s;
        let num_strides_wide = data.width/s;
        let mut multiply_buffer= Data::new(out_size);
        
        for a in 0..out_channels{
        //let mut total = 0;
        //let mut count = 0;
            for b in 0..in_channels{
                for i in 0..num_strides_high{
                    for j in 0..num_strides_wide{
                        for l in 0..ksize{
                            for k in 0..ksize{
                                multiply_buffer.set(a, i, j, data.get(b, (i*s) + (k*d)-1, (j*s)+(l*d)-1) * weights.get(a, b, l, k));
                            }
                        }
                    }
                }
            }
        }
        return multiply_buffer;
    }

    pub fn maxpool(data:&mut Data, ksize:u16, s:u16 , p:u16)->Data{           //Needs to return Data because of how its used in the SPPF function.
        let d = 0; // no dilation in maxpool function.
        let num_strides_high = data.height/s;
        let num_strides_wide = data.width/s;   
        let in_channels = data.depth;
        
        let out = Data::new([data.width, data.height, data.depth]);

        for b in 0..in_channels {
            for i in 0..num_strides_high {
                for j in 0..num_strides_wide {
                    let mut pool_max=0.0;
                    for l in 0..ksize {
                        for k in 0..ksize {
                            if data.get(b,(i*s) + (k*d), (j*s)+(l*d) )> pool_max {
                                pool_max = data.get(b,(i*s) + (k*d), (j*s)+(l*d));
                            }
                        }
                    }
                    for l in 0..ksize {
                        for k in 0..ksize {
                            out.set(b,(i*s) + (k*d), (j*s)+(l*d),pool_max); 
                        }
                    }    
                }
            }
        }out
    }
 
    pub fn concat(data1: &Data, data2: &Data)->Data{                     
        //concats along X dimension. Yolo concats along [1] dimension. Is same?.
        //c2 = sum([ch[x] for x in f]) seems to indicate adding channels(depth)

        let width = data1.width + data2.width;   //row length
        let height = data1.height + data2.height;  // num rows
        let depth = data1.depth + data2.depth;  // num layers
        let out_size = [width,height,depth];
        let mut concat_buffer= Data::new(out_size);

        for a in 0..data1.width{
            for b in 0..data1.height{
                for c in 0..depth{
                    concat_buffer.set(a, b, c, data1.get(a, b, c,));
                }
            }
        }
        for a in 0..data2.width{
            for b in 0..data2.height{
                for c in 0..depth{
                    concat_buffer.set(a+data1.width, b, c, data1.get(a, b, c,));
                }
            }
        }
        concat_buffer
    }
     
      
     
    pub fn upsample(data:&Data, out_size:[u16;3], scaling_factor:u16)->Data{
        let mut out_buffer= Data::new(out_size);
        for i in 0..data.height {
            for j in 0..data.width{
                for k in 0..data.depth{
                    for a in 0..scaling_factor{
                        out_buffer.set(i+a,j+a,k, data.get(i,j,k)); 
                    }
                }
            }            
        } out_buffer
    }        

pub fn sigmoid(x:f32)->f32{
    let euler = std::f32::consts::E;
    1/(1+euler.pow(-x))
    
}

    pub fn bottleneck(data:&Data,c1:u16, c2:u16, shortcut:bool, g:u16, e:f32, weights:&Weights,)->Data{	// ch_in, ch_out, shortcut, groups, expansion
        let cbomb:u16 =(c2 as f32*e) as u16;  // hidden channels
        let temp = conv(c1, cbomb, 1, 1,0,0,weights, data);
        let mut out = conv(cbomb, c2, 3, 1, 0, 0,weights, &temp);
        if c1==c2 && shortcut==true {
        out.add(data);	
        } out
    }

  // CSP Bottleneck with 3 convolutions    
    pub fn C3(c1:u16, c2:u16, shortcut:bool, g:u16, e:f32, weights:&Weights, data:&Data )->Data{	// ch_in, ch_out, shortcut, groups, expansion
      // n = max(round(n * gd), 1) //Divide N by 3 lol. Number of iterations in YAML file /3 this has been adjusted in the program flow. 
        let cbomb:u16 = (c2 as f32 * e) as u16;  //hidden channels

        let d1:Data = conv(c1, cbomb, 1, 1,0,0,weights,data);
        let b1 = bottleneck(&d1,cbomb, cbomb, shortcut, g, 1.0,weights);     //for _ in range(n)
        let d2:Data = conv(c1, cbomb, 1, 1,0,0,weights,&b1);
        let cdata:Data =concat(&b1, &d2);
        let out = conv(2 * cbomb, c2, 1, 1,0,0, weights, &cdata);  //optional act=FReLU(c2)
        out
    }

/*     pub fn spp(data:&Data, c1:u16, c2:u16, )->Data{
        let k=[5, 9, 13];
        let cbomb = c1/ 2;  // hidden channels should be floor division
        let tmp1 = conv(c1, cbomb, 1, 1, 0, 0, weights, data,);
        let tmp2 = Vec::new();
            for i in 0..k.len() {
                tmp2[i] = maxpool(&mut tmp1, k[i], 1, k[i]/2)
            }
        let tmp3 = concat(&tmp1, &tmp2[0]);
        let tmp4= concat(&tmp2[1], &tmp2[2]);
        let cat = concat(&tmp3, &tmp4);   

        let out = conv(cbomb * (k.len() as u16 + 1), c2, 1, 1, 0,0, weights, &cat,);         

            out

    } */

    pub fn sppf(c1:u16, c2:u16, data:&Data, weights:&Weights)->Data{
        let k=5;
        let cbomb = c1/ 2;  // hidden channels should be floor division
        let tmp1 = conv(c1, cbomb, 1, 1, 0, 0, weights, data,);
        let tmp2 = maxpool(&mut tmp1, k, 1, k/2);
        let tmp3 = maxpool(&mut tmp2, k, 1, k/2);
        let cat1 = concat(&tmp1, &tmp2);
        drop(tmp1);
        drop(tmp2);
        let tmp4 = maxpool(&mut tmp3, k, 1, k/2);
        let cat2 = concat(&tmp3, &tmp4);
        drop(tmp3);
        drop(tmp4);
        let cat3 = concat(&cat1, &cat2);
        drop(cat1);
        drop(cat2);
        let out = conv(cbomb * 4, c2,k, 1, k/2, 0, weights, data);
        out
    }

 
    pub fn xyxy2xywh(xyxy:[u32;4]){
        // Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
        let x1 = xyxy[0];
        let x2 =xyxy[1];
        let y1 = xyxy[2];
        let y2 = xyxy[3];

        let x_center   =   (x2-x1)/ 2       ;        
        let y_center   =   (y2-y1) / 2      ;
        let width      =   x2-x1            ;
        let height       =   y2-y1            ;
        return [x_center,y_center,width,height];
    }
    
pub fn output(predictions:Vec<Pred>, out_path:&str){ 

    for prediction in predictions{
        
        let class = prediction.pred_label;
        let conf = prediction.pred_score.to_string();
        let bbox = prediction.pred_box.to_string();
        let out_string = format!("{},{},{}", class, conf, bbox);
        
        let mut output = File::create(out_path)?;
        write!(output, "{}", out_string)?;
    }

}

                                  //pub fn count_instances(){
          /*   for c in pred[:, -1].unique():
                n = (pred[:, -1] == c).sum()  // detections per class
                s += f"{n} {names[int(c)]}{"s" * (n > 1)}, "  // add to string
            s = s.rstrip(", ")
        } */


        
struct Pred{
    pred_box: [f32;4],
    pred_score: f32,
    pred_classes: [f32;80],
    pred_final: u16,	
    pred_label: & 'static str,
}
impl Pred{
    fn new (prediction: Vec<f32>)->Pred{
        Pred{
            pred_box : prediction[0..4],
            pred_score : prediction[4],
            pred_classes : prediction[5..],
            pred_final : 255,
            pred_label : "Detection Failed",
        }
    }
}

/* So the detection head will receive input at 3 different points of the model, and it will have 3 different sizes, 

(0): Conv2d(128, 255, kernel_size=(1, 1), stride=(1, 1))
(1): Conv2d(256, 255, kernel_size=(1, 1), stride=(1, 1))
(2): Conv2d(512, 255, kernel_size=(1, 1), stride=(1, 1))

however, after conv, each will be 255 layers, corresponding to 3 predictions of 85 length - one for each anchor box.

 */

//20x20x85x3

//So for each cell in the output grid
//xyxy,c,c1..c80 x3

//for each sized box
//xyxy,c,c1..c80

//convert boxes to final size
//xyxy,c,c1..c80

//run nms 
//xyxy,C,cls

//convert to xywh
//xywh, C, cls

//match on label
//xywh,C,CLS
     

fn detect(data: &Data, weights: &Weights, connection:i32){
        let num_classes = 80;  // number of classes
        let num_outputs= num_classes + 5;  // number of outputs per anchor
        let anchors =
                [[10,13, 16,30, 33,23],         // 3 small
                [30,61, 62,45, 59,119],         // 3 Medium
                [116,90, 156,198, 373,326]];    // 3 Large
        let num_sizes = anchors.len() as i32;  // number of detection layers
        let num_anchors = anchors[0].len() as i32/ 2;  // number of anchors
        
        
        let z = Vec::new();  // inference output
        for i in 0..(num_sizes){
            let data:Data = conv(data, weights, num_outputs * num_anchors, 1); // need to pass the right weights aray depending on where the detection head attaches to the model. 
            /* (0): Conv2d(128, 255, kernel_size=(1, 1), stride=(1, 1))
            (1): Conv2d(256, 255, kernel_size=(1, 1), stride=(1, 1))
            (2): Conv2d(512, 255, kernel_size=(1, 1), stride=(1, 1)) */


            let nx = data.dims[0];
            let ny = data.dims[1];                         // x(255,20,20) 
            data.view(vec![num_anchors, num_outputs, ny, nx]);  //3x85x20x20
            data.permute(vec![0,2,3,1]);                        //x(3,20,20,85)


            
            let anchor_grid:[i32;6] = anchors[connection]; 

            let pred = Vec::new();
            for i in 0..num_anchors{
                for j in 0..nx{
                    for k in 0..ny{
            
                    // prediction xywh
                    let pred_x = data.get(i,j,k,0);
                    let pred_y = data.get(i,j,k,1);
                    let pred_height = data.get(i,j,k,2);
                    let pred_width = data.get(i,j,k,3);

                    //grids and anchors
                    let grid_x = j as f32;
                    let grid_y = k as f32;
                    let anchor_width:f32 = anchor_grid[i*2];
                    let anchor_height:f32 = anchor_grid[i*2+1];
                    
                    //prediction boxes xywh
                    let box_x = 2.0* sigmoid(pred_x)-0.5 + grid_x;
                    let box_y = 2.0* sigmoid(pred_y)-0.5 + grid_y;
                    let box_width = anchor_width *(2.0 * sigmoid(pred_width)).pow(2) ;
                    let box_height = anchor_height *(2.0 * sigmoid(pred_height)).pow(2) ;
                    
                    //Update tensors 
                    data.set(i,j,k,box_x);
                    data.set(i,j,k,box_y);
                    data.set(i,j,k,box_width);
                    data.set(i,j,k,box_height);
                    
                    //Create Predictions struct
                    let temp = Vec::new();
                    for l in 0..num_outputs{
                        temp.push(data.get(i,j,k,l))
                    }
                    pred.push(Pred::new(temp));
                }    
            }
        }
    }


                //this needs to pass in a vec of 3*20*20*85 <f32> similar to [[f32;85];3*20*20]
                pred = non_max_suppression(pred); // returns list of tensors pred[0] = (xyxy, conf, class)
                output(predictions, out_path);
}

fn non_max_suppression(predictions:Vec<Pred>){
    /*Runs Non-Maximum Suppression (NMS) on inference results
    Returns:
    list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """*/
    
        filter_sort(predictions);
    
      
        for class_max in predictions{
            for (i,this_prediction) in predictions.into_iter().enumerate(){
                
                if this_prediction.pred_final == class_max.pred_final {
                    let iou = box_iou(this_prediction.pred_box, class_max.pred_box);
                    if iou > iou_thres{
                        predictions.remove(i);
                    } 
                }
        }	
    }
}

fn filter_sort(list_of_predictions:Vec<Pred>){
    list_of_predictions.sort_by_key(|k|k.pred_score);
    let conf = crate::settings::non_max_suppression::conf_thres;
    
    for prediction in  list_of_predictions{		
        
        let hotmax = 0.0;
        
        for j in 0..80 {	//num_classes
            if prediction.pred_classes[j] > hotmax{		
                hotmax=prediction.pred_classes[j];
                prediction.pred_final = j;
            }
        }
        prediction.pred_score *= prediction.pred_classes[prediction.pred_final];
        prediction.pred_label = labels[prediction.pred_final];

    }
    
    let labels =    // this is a real shitty solution, but I've just realised that I'd need to pass this as an argment to 3 different functions just to bring it into scope here.
    ["person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush"];
}	
    
fn box_iou(box1:&[f32], box2: &[f32])->f32{
    let  eps:f32	=	1e-7;		
    // convenience labels below, makes the functions easer to follow. Should be optimised out at complile time. 
    let box1_x1 = box1[0];
    let box1_y1 = box1[1];
    let box1_x2 = box1[2];
    let box1_y2 = box1[3];

    let box2_x1 = box2[0];
    let box2_y1 = box2[1];
    let box2_x2 = box2[2];
    let box2_y2 = box2[3];

    let box1_width = box1_x2 - box1_x1;
    let box1_height = box1_y2 - box1_y1;
    let box2_width = box2_x2 - box2_x1;
    let box2_height = box2_y2 - box2_y1;

    let inter:f32 = min(box1_width, box2_width) * min(box1_height, box2_height);
    let union:f32 = box1_width * box1_height + box2_width * box2_height - inter + eps;

    return inter / union
}
    
    
fn min(dim1:f32,dim2:f32)->f32{
    if dim1 > dim2 {
        dim2
    }else{
        dim1
    }
}

   
} 
