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
