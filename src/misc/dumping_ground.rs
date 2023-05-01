for i in [1, 0, 1, 0]{
    let shape = [im.shape[i],1,1];  // [width,1,1] or [height,1,1]
    let gn = Vec::new();
    gn[i] = Data::new(shape);

}  // normalizations

xyxy2xywh(
    torch.tensor(xyxy)
    .view(1, 4)) / gn)  //gn must have the appropriate shape here.
    .view(-1).tolist()

for x in pred
    xyxy = x[0]  // xyxy pixels
    xywh = [xyxy2xywh(xyxy)]  // xywh pixels
    xyxyn = [x / g for x, g in zip(xyxy, gn)]  // xyxy normalized
    xywhn = [x / g for x, g in zip(xywh, gn)]  // xywh normalized



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
        