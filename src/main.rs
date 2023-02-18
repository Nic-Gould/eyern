use std::vec;
use image::GenericImageView;


fn main(){
//the detect function runs the model and passes back a predictions struct

fn detect(weights= "yolov5s.pt",  // model.pt path(s)
       source="data/images",  // file/dir/URL/glob, 0 for webcam
       imgsz=640,  // inference size (pixels)
       conf_thres=0.25,  // confidence threshold
       iou_thres=0.45,  // NMS IOU threshold
       max_det=1000,  // maximum detections per image
       device="",  // cuda device, i.e. 0 or 0,1,2,3 or cpu
       view_img=False,  // show results
       save_txt=False,  // save results to *.txt
       save_conf=False,  // save confidences in --save-txt labels
       save_crop=False,  // save cropped prediction boxes
       nosave=False,  // do not save images/videos
       classes=None,  // filter by class: --class 0, or --class 0 2 3
       agnostic_nms=False,  // class-agnostic NMS
       augment=False,  // augmented inference
       update=False,  // update all models
       project="runs/detect",  // save results to project/name
       name="exp",  // save results to project/name
       exist_ok=False,  // existing project/name ok, do not increment
       line_thickness=3,  // bounding box thickness (pixels)
       hide_labels=False,  // hide labels
       hide_conf=False,  // hide confidences
       half=False,  // use FP16 half-precision inference
       ){
/*  save_img = not nosave and not source.endswith(".txt")  // save inference images
webcam = source.isnumeric() or source.endswith(".txt") or source.lower().startswith(
    ("rtsp://", "rtmp://", "http://", "https://"))

// Directories
save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  // increment run
(save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  // make dir

// Initialize
set_logging()
device = select_device(device)
half &= device.type != "cpu"  // half precision only supported on CUDA

// Load model
model = attempt_load(weights, map_location=device)  // load FP32 model
stride = int(model.stride.max())  // model stride
imgsz = check_img_size(imgsz, s=stride)  // check image size
names = model.module.names if hasattr(model, "module") else model.names  // get class names
if half:
    model.half()  // to FP16 */


// Load image into a data struct and convert to f32 # BGR to RGB, to DxWxH


       

fn read_image(){
            
    let im0 = image::open("cat.jpg").unwrap();  
    //img.dimensions();
    //img.color();


    // Padded resize
    //img = letterbox(img0, img_size, stride=stride)[0]

    // Convert
    //img = img[:, :, ::-1].transpose(2, 0, 1)  // BGR to RGB, to DxWxH
    

    let img:Data = Data::new([640,640,3]);      //A data is already padded with zeros
        let vert_padding = (640-480)/2;
        for i in img.height{
            for j in img.width{
                for k in img.depth{
                    for pixel in im0.pixels(){
                    img.set(i+vert_padding,j,k, pixel[k] as f32);
                }
            }
        }
        }


}

/*    // Set Dataloader
vid_path, vid_writer = None, None
if webcam:
    view_img = check_imshow()
    cudnn.benchmark = True  // set True to speed up constant image size inference
    dataset = LoadStreams(source, img_size=imgsz, stride=stride)
else:
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

// Run inference
/*    if device.type != "cpu":
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  // run once */
t0 = time.time()
for path, img, im0s, vid_cap in dataset:
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  // uint8 to fp16/32
    img /= 255.0  // 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
*/

fn detect(){
    // Inference
    let t1 = time_synchronized();
    let pred = model(img);

    // Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det);
    t2 = time_synchronized();



Need to examine the actual output of pred at this point;

It should be something like [c,x,y,w,y,c1..c80] for each grid cell?


   /*  // Apply Classifier		//not using second stage classifier
    if classify:
        pred = apply_classifier(pred, modelc, img, im0s)
*/
    // Process detections
    for det in pred {  // detections per image
        // p = path
        // s = ""
        // im0 = im0s.copy
        //frame = getattr(dataset,"frame", 0)
       
        let out_path = "labels";
        s += "%gx%g " % img.shape[2:]  // print string
        
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  // normalization gain whwh

        
        

            // Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            // Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  // detections per class
                s += f"{n} {names[int(c)]}{"s" * (n > 1)}, "  // add to string

            // Write results
            for *xyxy, conf, class in reversed(det):
                if save_txt:  // Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  // normalized xywh
                    line = (class, *xywh, conf) if save_conf else (class, *xywh)  // label format
                    with open(txt_path + ".txt", "a") as f:
                        f.write(("%g " * len(line)).rstrip() % line + "\n")

      /*     // Add bbox to image
                    c = int(class)  // integer class
                    label = f"{names[c]} {conf:.2f}")
                    plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness) */
           
    }
       


/*    print(f"Done. ({time.time() - t0:.3f}s)")
       } */

/* if __name__ == "__main__":
parser = argparse.ArgumentParser()
parser.add_argument("--weights", nargs="+", type=str, default="yolov5s.pt", help="model.pt path(s)")
parser.add_argument("--source", type=str, default="data/images", help="file/dir/URL/glob, 0 for webcam")
parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="inference size (pixels)")
parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
parser.add_argument("--view-img", action="store_true", help="show results")
parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --class 0, or --class 0 2 3")
parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
parser.add_argument("--augment", action="store_true", help="augmented inference")
parser.add_argument("--update", action="store_true", help="update all models")
parser.add_argument("--project", default="runs/detect", help="save results to project/name")
parser.add_argument("--name", default="exp", help="save results to project/name")
parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
opt = parser.parse_args()
print(opt)
check_requirements(exclude=("tensorboard", "thop"))

detect(**vars(opt)) */


/* Detect:

            args.append([ch[x] for x in f])

            if isinstance(args[1], int):  # number of anchors

                args[1] = [list(range(args[1] * 2))] * len(f) */

                fn detect(ch=(),) {
                    let num_classes = 80; // number of classes
                    let anchors =
                        [[10,13, 16,30, 33,23],
                        [30,61, 62,45, 59,119],
                        [116,90, 156,198, 373,326]];
                  
                /*   // YOLOv5 Detect head for detection models
                    stride = None  // strides computed during build
                    dynamic = False  // force grid reconstruction
                    export = False  // export mode */
                
                        let num_outputs = num_classes + 5;  // number of outputs per anchor
                        let num_layers = anchors.len() ;   // number of detection layers
                        let num_anchors = anchors[0].len()as u32 / 2;    //This is just 3 right?           //should be floor division
                        
                        let grid = Vec::new();  // init grid
                        let anchor_grid = Vec::new();  // init anchor grid
                        // shape(nl,na,2)
                        
                
                
                       // m = nn.ModuleList(nn.Conv2d(x, no * na, 1) for x in ch)  // output conv = do a convulution on each input Channel?
                        let z = vec::new(); 
                        for i in 0..num_layers{
                            x[i] = conv(data.depth(), num_outputs * num_anchors, 1); //x[i] = m[i](x[i])  // conv
                            let batch_size = x[i].depth;
                            let ny = x[i].height;
                            let nx = x[i].width;
                                                        // x(batch_size,255,20,20) to x(batch_size,3,20,20,85)
                            x[i] = x[i].view(batch_size, num_anchors, num_outputs, ny, nx).permute(0, 1, 3, 4, 2).contiguous();
                
                
                            if grid[i].shape[2:4] != x[i].shape[2:4]{
                                grid[i]=make_grid(nx, ny, i);
                                anchor_grid[i] = make_grid(nx, ny, i);
                            }    
                            xy, wh, conf = x[i].sigmoid().split((2, 2, num_classes + 1), 4);
                            xy = (xy * 2 + grid[i]) * stride[i] ;
                            wh = (wh * 2) ** 2 * anchor_grid[i]  ;
                            y = torch.cat((xy, wh, conf), 4);
                            z.push(y.view(batch_size, num_anchors * nx * ny, num_outputs));
                        }
                        return (cat(z, 1), x);
                }
                    fn make_grid(nx:u16, ny:u16, i:u16,)->vec<f64> {
                      
                        shape = [1, num_anchors, ny, nx, 2]  // grid shape
                        
                         
                        let grid = Vec::from([[0.0;nx];ny]);
                        torch.stack((xv, yv), 2).expand(shape) - 0.5  //add grid offset, i.e. y = 2.0 * x - 0.5
                        
                        anchor_grid = (anchors[i] * stride[i]).view((1, num_anchors , 1, 1, 2)).expand(shape);
                        return grid, anchor_grid
                torch    }
                 }

fn setup(){
(self, ims, pred, files, times=(0, 0, 0), names=None, shape=None)   
    // list of images as numpy arrays,
    // list of tensors pred[0] = (xyxy, conf, class)
    // class names
    // image filenames
    // profiling times

     
        gn = [torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d) for im in ims]  // normalizations
        

        
        xyxy = pred  // xyxy pixels
        xywh = [xyxy2xywh(x) for x in pred]  // xywh pixels
        xyxyn = [x / g for x, g in zip(xyxy, gn)]  // xyxy normalized
        xywhn = [x / g for x, g in zip(xywh, gn)]  // xywh normalized
        
        n = pred.len() // number of images (batch size)
        t = tuple(x.t / n * 1E3 for x in times)  // timestamps (ms)
        s = tuple(shape)  // inference BCHW shape

}
fn output (self, pprint=False, show=False, save=False, crop=False, render=False, labels=True, save_dir=Path("")){
        
        s, crops = "", []
        for i, (im, pred) in enumerate(zip(ims, pred)):
            s += f"\nimage {i + 1}/{len(pred)}: {im.shape[0]}x{im.shape[1]} "  // string
            if pred.shape[0]:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  // detections per class
                    s += f"{n} {names[int(c)]}{"s" * (n > 1)}, "  // add to string
                s = s.rstrip(", ")
                
                

                if show or save or render or crop:
                    annotator = Annotator(im, example=str(names))
                    for box, conf, class in reversed(pred):  // xyxy, confidence, class
                        label = f"{names[int(class)]} {conf:.2f}"
                        if crop:
                            file = save_dir / "crops" / names[int(class)] / files[i] if save else None
                            crops.append({
                                "box": box,
                                "conf": conf,
                                "class": class,
                                "label": label,
                                "im": save_one_box(box, im, file=file, save=save)})
                        else:  // all others
                            annotator.box_label(box, label if labels else "", color=colors(class))
                    im = annotator.im
/*             else:
                s += "(no detections)"

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if show:
                display(im) if is_notebook() else im.show(files[i])
            if save:
                f = files[i]
                im.save(save_dir / f)  # save
                if i == n - 1:
                    LOGGER.info(f"Saved {n} image{"s" * (n > 1)} to {colorstr("bold", save_dir)}")
            if render:
                ims[i] = np.asarray(im)
        if pprint:
            s = s.lstrip("\n")
            return f"{s}\nSpeed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {s}" % t
        if crop:
            if save:
                LOGGER.info(f"Saved results to {save_dir}\n")
            return crops */

/*     @TryExcept("Showing images is not supported in this environment")
    def show(self, labels=True):
        _run(show=True, labels=labels)  # show results

    def save(self, labels=True, save_dir="runs/detect/exp", exist_ok=False):
        save_dir = increment_path(save_dir, exist_ok, mkdir=True)  # increment save_dir
        _run(save=True, labels=labels, save_dir=save_dir)  # save results

    def crop(self, save=True, save_dir="runs/detect/exp", exist_ok=False):
        save_dir = increment_path(save_dir, exist_ok, mkdir=True) if save else None
        return _run(crop=True, save=save, save_dir=save_dir)  # crop results

    def render(self, labels=True):
        _run(render=True, labels=labels)  # render results
        return ims

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = "xmin", "ymin", "xmax", "ymax", "confidence", "class", "name"  # xyxy columns
        cb = "xcenter", "ycenter", "width", "height", "confidence", "class", "name"  # xywh columns
        for k, c in zip(["xyxy", "xyxyn", "xywh", "xywhn"], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. "for result in results.tolist():"
        r = range(n)  # iterable
        x = [Detections([ims[i]], [pred[i]], [files[i]], times, names, s) for i in r]
        # for d in x:
        #    for k in ["ims", "pred", "xyxy", "xyxyn", "xywh", "xywhn"]:
        #        setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def print(self):
        LOGGER.info(__str__())

    def __len__(self):  # override len(results)
        return n

    def __str__(self):  # override print(results)
        return _run(pprint=True)  # print results

    def __repr__(self):
        return f"YOLOv5 {__class__} instance\n" + __str__() */
                            }

                        }