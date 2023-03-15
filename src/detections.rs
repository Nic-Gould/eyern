/* So the detection head will receive input at 3 different points of the model, and it will have 3 different sizes, 

(0): Conv2d(128, 255, kernel_size=(1, 1), stride=(1, 1))
(1): Conv2d(256, 255, kernel_size=(1, 1), stride=(1, 1))
(2): Conv2d(512, 255, kernel_size=(1, 1), stride=(1, 1))

however, after conv, each will be 255 layers, corresponding to 3 predictions of 85 length - one for each anchor box.

 */





struct Detections{
    dims: Vec<u16>, //WidthxHeightxDepth
    data: Vec<f32>,
}
impl Detections{
    fn view(&self, new_dims:Vec<u16>)
    {
        if new_dims.iter().product()=self.data.len(){ 
            self.dims = dims;
        }
            else
        {
            panic!("Bro you can't make a view like that, the dimensions are all munted.")
        }
    }
    fn new (dims: Vec<u16>)->Detections{
        let length = dims.iter().product();
        Detections { 
            dims,
            data: vec![0.0; length],
        }    
    }
    fn get (&self, x:u16, y:u16, z:u16, a:u16 )-> f32{             
        let index:usize = (a* self.dims[0]*self.dims[1]*self.dims[2] 
                        + z * (self.dims[0]*self.dims[1]) 
                        + y*self.dims[0] 
                        + x).into();
        self.data[index]
    }
    fn set (&mut self, x:u16, y:u16, z:u16, a:u16, exp:f32) {             
        let index:usize = (a* self.dims[0]*self.dims[1]*self.dims[2] 
            + z * (self.dims[0]*self.dims[1]) 
            + y*self.dims[0] 
            + x).into();
        self.data[index] = exp;
    }  
    fn add (&mut self, data:&Detections) {
    
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
    fn permute(&self, new_dims:Vec<u16>){
        if new_dims.len()=self.dims.len(){ 
        let temp = Detections::new(self.dims);
        
            for i in 0..self.dims[0]{
                for j in 0..self.dims[1]{
                    for k in 0..self.dims[2]{
                        for l in 0..self.dims[3]{
                            temp.set(i,*k,*l,*j, self.get(i,j,k,l));
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


struct Pred{
    pred_box: [u16;4],
    pred_score: f32,
    pred_classes: [f32;80],
    pred_final: [u8],	//the score and index / class number
    pred_label: &str,
}
impl Pred{
    fn new (prediction: [f32;85])->Pred{
        Pred{
            pred_box : predictions[0..4],
            pred_score : predictions[4],
            pred_classes : predictions[5..],
            pred_final : 255,
            pred_label : "Detection Failed".to_string,
        }
    }
}


fn non_max_suppression(predictions:Data , settings:Settings::non_max_suppression){
/*Runs Non-Maximum Suppression (NMS) on inference results
Returns:
list of detections, on (n,6) tensor per image [xyxy, conf, cls]
"""*/

    wrangle(predictions);

    let results = Vec::new();
    for i in 0..prediction.dims[0] {  // image index, image inference
        results.push(predictions[0]);
        let this_prediction = predictions[i];
        let to_compare = results[-1];
        if this_prediction.pred_final = to_compare.pred_final {
            let iou = box_iou(pred.pred_box, to_compare.pred_box);
            if iou > iou_thres{
                predictions.remove(i);
            } 
        }
    }	
}

fn wrangle(list_of_predictions:Vec){
    list_of_predictions.sort_by_key(|k|k[4])
    .filter(|x[4]|x[4]<conf_thres);
    
    for prediciton in  list_of_predictions{		
       
        let hotmax = 0.0;
        
        for j in 0..80 {	//num_classes
            if prediction.pred_classes[j] > hotmax[0]{		
                hotmax=prediction[i].pred_classes[j];
                prediction.pred_final = j;
            }
        }
        prediction.pred_score *= prediction.pred_classes[pred_final];
        prediction.pred_label = labels[predictions.pred_final];
    }

}	

fn box_iou(box1, box2,)->f32{


let  eps	=	1e-7;		
// convenience labels below, makes the functions easer to follow. Should be optimised out at complile time. 
let box1_x1 = &box1[0];
let box1_y1 = &box1[1];
let box1_x2 = &box1[2];
let box1_y2 = &box1[3];

let box2_x1 = &box2[0];
let box2_y1 = &box2[1];
let box2_x2 = &box2[2];
let box2_y2 = &box2[3];

let box1_width = box1_x2 - box1_x1;
let box1_height = box1_y2 - box1_y1;
let box2_width = box2_x2 - box2_x1;
let box2_height = box2_y2 - box2_y1;

let inter = min(box1_width, box2_width) * min(box1_height, box2_height);
let union = box1_width * box1_height + box2_width * box2_height - inter + eps;

return inter / union
}

fn min(dim1:u16,dim2:u16)->{
    if dim1 > dim2 {
        dim2
    }else{
        dim1
    }
}

fn xyxy2xywh(pred: &[f32]){

let pred_x1 = pred[0];
let pred_y1 = pred[1];
let pred_x2 = pred[2];
let pred_y2 = pred[3];

let box_x = 2* sigmoid(pred_x1)-0.5 + grid_x;
let box_y = 2* sigmoid(pred_y1)-0.5 + grid_y;
let box_width = anchor_width (2 * sigmoid(pred_x2))^2 ;
let box_height = anchor_height (2 * sigmoid(pred_y2))^2 ;

pred[0] = box_x;
pred[1] = box_y;
pred[2] = box_width;
pred[3] = box_height;
}

let labels = ["person",
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