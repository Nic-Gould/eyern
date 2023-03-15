use std::vec;
use image::GenericImageView;


fn main(){
       
        
        
        fn detect(data: &Data, weights: &Weights, connection:u8){
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
        
        
                   
                    let anchor_grid = anchors[connection]; 
        
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
                            let grid_x = j;
                            let grid_y = k;
                            let anchor_width = anchor_grid[i*2];
                            let anchor_height = anchor_grid[i*2+1];
                            
                            //prediction boxes xywh
                            let box_x = 2* sigmoid(pred_x)-0.5 + grid_x;
                            let box_y = 2* sigmoid(pred_y)-0.5 + grid_y;
                            let box_width = anchor_width *(2 * sigmoid(pred_width)).pow(2) ;
                            let box_height = anchor_height *(2 * sigmoid(pred_height)).pow(2) ;
                            
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
        

                        //this needs to pass in a vec of 3*20*20*85 <f32> similar to [[f32;85];3*20*20]
                        pred = non_max_suppression(pred, Settings::non_max_suppression); // returns list of tensors pred[0] = (xyxy, conf, class)
        
        }
        
       
    }
            
