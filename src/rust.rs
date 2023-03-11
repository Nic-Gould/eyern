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



fn detect(data: &Data, weights: &Weights){
        let num_classes = 80;  // number of classes
        let num_outputs= num_classes + 5;  // number of outputs per anchor
        let anchors =
                [[10,13, 16,30, 33,23],         // 3 small
                [30,61, 62,45, 59,119],         // 3 Medium
                [116,90, 156,198, 373,326]];    // 3 Large
        let num_sizes = anchors.len() as i32;  // number of detection layers
        let num_anchors = anchors[0].len() as i32/ 2;  // number of anchors
        
        //register_buffer('anchors', torch.tensor(anchors).float().view(num_sizes, -1, 2))  // shape(num_sizes,num_anchors,2)
        
        
       // m = nn.ModuleList(nn.Conv2d(x, num_outputs * num_anchors, 1) for x in ch)  // output conv
                /*(0): Conv2d(128, 255, kernel_size=(1, 1), stride=(1, 1))
                (1): Conv2d(256, 255, kernel_size=(1, 1), stride=(1, 1))
                (2): Conv2d(512, 255, kernel_size=(1, 1), stride=(1, 1)) """ */
        
            // parsing each output seperately should allow for better detection of differenly sized objects with the same centre?
    
        let z = Vec::new();  // inference output
        for i in 0..(num_sizes){
            let data:Detections = conv(data, weights, num_outputs * num_anchors, 1); // need to pass the right weights aray depending on where the detection head attaches to the model. 
            
            let nx = data.dims[0];
            let ny = data.dims[1];  // x(255,20,20) to x(3,20,20,85) the 
            data.view(vec![num_anchors, num_outputs, ny, nx]);  //3x85x20x20
            data.permute(vec![0,2,3,1]);


            let grids = make_grid(nx, ny, i);
            let grid = grids[0];
            let anchor_grid = grids[1]; 

            // Detect (boxes only)
            
            let pred = data.sigmoid();//.split((2, 2, num_classes + 1), 4);
            pred[0..2] = (pred[0..2] * 2 + grid) * stride;  // xy
            pred[2..4] = (pred[2..4] * 2) ** 2 * anchor_grid;  // wh
            z.push(pred.view(num_anchors * nx * ny, num_outputs));      //fucking why though.
        }
                let pred = (concat (z, 1), x); 
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det); // returns list of tensors pred[0] = (xyxy, conf, class)

}

    fn make_grid(nx:u8, ny:u8, i:u8, num_anchors:u8){ 
        let shape = [1, num_anchors, ny, nx, 2];  // grid shape 1x3x20x20x2
        let mc_vec = Vec::new();
        for i in 0..ny{
            for j in 0..nx{
                mc_vec.push([i,j])
            }
        }
        let grid = Vec::new();
        for i in mc_vec.iter(){
            grid.push([[[[i; num_anchors]; nx]; ny]; 2] -0.5) //lllooooooolll   -0.5, which part? , the i,j component?
        }
             
        //.expand(shape) - 0.5  // add grid offset, i.e. y = 2.0 * x - 0.5
       
        anchor_grid = (anchors * stride).view([1, num_anchors, 1, 1, 2])
        .expand(shape);
        
        let out = [grid, anchor_grid];
        out
    }