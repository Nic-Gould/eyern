/* Detect:

            args.append([ch[x] for x in f])

            if isinstance(args[1], int):  # number of anchors

                args[1] = [list(range(args[1] * 2))] * len(f) */

                fn detect(self, ch=(),) {
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
                        
                
                
                       // self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  // output conv = do a convulution on each input Channel?
                        let x = vec::new(); 
                        for i in 0..num_layers{
                            x[i] = conv(data.depth(), num_outputs * num_anchors, 1); //x[i] = self.m[i](x[i])  // conv
                            let bs = x[i].depth;
                            let ny = x[i].height;
                            let nx = x[i].width;
                                                        // x(bs,255,20,20) to x(bs,3,20,20,85)
                            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous();
                
                
                            if self.grid[i].shape[2:4] != x[i].shape[2:4]{
                                self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                            }    
                            xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4);
                            xy = (xy * 2 + self.grid[i]) * self.stride[i] ;
                            wh = (wh * 2) ** 2 * self.anchor_grid[i]  ;
                            y = torch.cat((xy, wh, conf), 4);
                            z.append(y.view(bs, self.na * nx * ny, self.no));
                        }
                        return (cat(z, 1), x);
                }
                    fn make_grid(self, nx:u16, ny:u16, i:u16,) {
                      
                        shape = 1, self.na, ny, nx, 2  // grid shape
                        
                         
                        let grid = Vec::from([[0.0;nx];ny]);
                        torch.stack((xv, yv), 2).expand(shape) - 0.5  //add grid offset, i.e. y = 2.0 * x - 0.5
                        
                        anchor_grid = (anchors[i] * stride[i]).view((1, self.na, 1, 1, 2)).expand(shape);
                        return grid, anchor_grid
                torch    }
                 }

fn setup(){
(self, ims, pred, files, times=(0, 0, 0), names=None, shape=None)

        d = pred[0].device  # device
        gn = [torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d) for im in ims]  // normalizations
        
   // list of images as numpy arrays
       // list of tensors pred[0] = (xyxy, conf, cls)
        // class names
        // image filenames
       // profiling times
        
        self.xyxy = pred  // xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  // xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  // xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  // xywh normalized
        
        self.n = pred.len() // number of images (batch size)
        self.t = tuple(x.t / self.n * 1E3 for x in times)  // timestamps (ms)
        self.s = tuple(shape)  # inference BCHW shape

}
fn run (self, pprint=False, show=False, save=False, crop=False, render=False, labels=True, save_dir=Path('')){
        
        s, crops = '', []
        for i, (im, pred) in enumerate(zip(self.ims, self.pred)):
            s += f'\nimage {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '  // string
            if pred.shape[0]:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  // detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  // add to string
                s = s.rstrip(', ')
                if show or save or render or crop:
                    annotator = Annotator(im, example=str(self.names))
                    for *box, conf, cls in reversed(pred):  // xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            file = save_dir / 'crops' / self.names[int(cls)] / self.files[i] if save else None
                            crops.append({
                                'box': box,
                                'conf': conf,
                                'cls': cls,
                                'label': label,
                                'im': save_one_box(box, im, file=file, save=save)})
                        else:  // all others
                            annotator.box_label(box, label if labels else '', color=colors(cls))
                    im = annotator.im
            else:
                s += '(no detections)'

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if show:
                display(im) if is_notebook() else im.show(self.files[i])
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                if i == self.n - 1:
                    LOGGER.info(f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")
            if render:
                self.ims[i] = np.asarray(im)
        if pprint:
            s = s.lstrip('\n')
            return f'{s}\nSpeed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {self.s}' % self.t
        if crop:
            if save:
                LOGGER.info(f'Saved results to {save_dir}\n)
            return crops

    @TryExcept('Showing images is not supported in this environment')
    def show(self, labels=True):
        self._run(show=True, labels=labels)  # show results

    def save(self, labels=True, save_dir='runs/detect/exp', exist_ok=False):
        save_dir = increment_path(save_dir, exist_ok, mkdir=True)  # increment save_dir
        self._run(save=True, labels=labels, save_dir=save_dir)  # save results

    def crop(self, save=True, save_dir='runs/detect/exp', exist_ok=False):
        save_dir = increment_path(save_dir, exist_ok, mkdir=True) if save else None
        return self._run(crop=True, save=save, save_dir=save_dir)  # crop results

    def render(self, labels=True):
        self._run(render=True, labels=labels)  # render results
        return self.ims

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        r = range(self.n)  # iterable
        x = [Detections([self.ims[i]], [self.pred[i]], [self.files[i]], self.times, self.names, self.s) for i in r]
        # for d in x:
        #    for k in ['ims', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
        #        setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def print(self):
        LOGGER.info(self.__str__())

    def __len__(self):  # override len(results)
        return self.n

    def __str__(self):  # override print(results)
        return self._run(pprint=True)  # print results

    def __repr__(self):
        return f'YOLOv5 {self.__class__} instance\n' + self.__str__()
                            }