struct Non_Max_Suppression{
    num_classes:u32 = 80,  //number of classes
    max_nms:u32 = 30000,
    conf_thres:f32=0.25, 
    iou_thres:f32=0.45, 
    max_det:u32=300,
}

  //weights= "yolov5s.pt",  // model.pt path(s)
  source="data/images",  // file/dir/URL/glob, 0 for webcam
  imgsz=640,  // inference size (pixels)
