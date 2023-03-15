// The data struct is an output from each layer, and is used as an input in the subsequent layer.
// On a few occasions the data is used in later layers and needs to be kept.
// mutable access to a data struct is only required during creation
// all other accesses can use read only values.

use std::vec;

 fn main(){
    
    #[derive(Debug)]
    struct Data{                                     
        height:u16,
        width:u16,
        depth:u16,
        count:u32,
        data:Vec<f32>,
    } 
    impl Data {         //probs make it a trait and hit it with that detection struct
        fn new (size:[u16;3])->Data{
            Data {
                width:size[0],   //row length
                height:size[1],  // num rows
                depth:size[2],  // num layers
                count:(size[0]*size[1]*size[2]).into(),
                data: vec![0.0; (size[0]*size[1]*size[2]).into()],
            }    
        }
        fn get (&self, x:u16, y:u16, z:u16)-> f32{             
            let index:usize = (z * (self.height*self.width) + y*self.width +x).into();
            self.data[index]
        }
        fn set (&mut self, x:u16, y:u16, z:u16, exp:f32) {             
            let index:usize = (z * (self.height*self.width) + y*self.width +x).into();
            self.data[index] = exp;
        }  
        fn add (&mut self, data:&Data) {
        
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
    struct Weights{
        ksize_h:u16,
        ksize_w:u16,
        ch_in:u16,
        ch_out:u16,
        count:usize,
        data:Vec<f32>,
    }

    impl Weights {
        fn new (&self, size:[u16;4], path:&str)->Weights{
            Weights {
                ksize_h:size[0],
                ksize_w:size[1],
                ch_in:size[2],
                ch_out:size[3],
                count:(self.ksize_h * self.ksize_w * self.ch_in * self.ch_out).into(),
                data: vec![0.0; self.count]
            }            
        } 
        fn get (&self, x:u16, y:u16, z:u16, a:u16)-> &f32{        
            let index:usize = (a*(self.ksize_h*self.ksize_w*self.ch_in) + z*(self.ksize_h * self.ksize_w) + y*(self.ksize_w) +x).into();
            &self.data[index]
        }          
    }
    fn read_image(){
            
        let im0 = image::open("cat.jpg").unwrap();  
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
    
        let img:Data = Data::new([640,640,3]);      //A data is already padded with zeros
            let vert_padding = (640-480)/2;
            for i in img.height{
                for j in img.width{
                    for k in img.depth{
                        for pixel in im0.pixels(){
                        img.set(i+vert_padding,j,k, pixel[k] as f32/255.0);
                    }
                }
            }
            }
    }
    fn conv(in_channels:u16, out_channels:u16, ksize:u16, s:u16, p:u16, d:u16, weights:&Weights, data:&Data) -> Data{
        
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

    fn maxpool(data:&mut Data, ksize:u16, s:u16 , p:u16)->Data{           //Needs to return Data because of how its used in the SPPF function.
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
 
    fn concat(data1: &Data, data2: &Data)->Data{                     
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
     
      
     
    fn upsample(data:&Data, out_size:[u16;3], scaling_factor:u16)->Data{
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



    fn bottleneck(data:&Data,c1:u16, c2:u16, shortcut:bool, g:u16, e:f32, weights:&Weights,)->Data{	// ch_in, ch_out, shortcut, groups, expansion
        let cbomb:u16 =(c2 as f32*e) as u16;  // hidden channels
        let temp = conv(c1, cbomb, 1, 1,0,0,weights, data);
        let mut out = conv(cbomb, c2, 3, 1, 0, 0,weights, &temp);
        if c1==c2 && shortcut==true {
        out.add(data);	
        } out
    }

  // CSP Bottleneck with 3 convolutions    
    fn C3(c1:u16, c2:u16, shortcut:bool, g:u16, e:f32, weights:&Weights, data:&Data )->Data{	// ch_in, ch_out, shortcut, groups, expansion
      // n = max(round(n * gd), 1) //Divide N by 3 lol. Number of iterations in YAML file /3 this has been adjusted in the program flow. 
        let cbomb:u16 = (c2 as f32 * e) as u16;  //hidden channels

        let d1:Data = conv(c1, cbomb, 1, 1,0,0,weights,data);
        let b1 = bottleneck(&d1,cbomb, cbomb, shortcut, g, 1.0,weights);     //for _ in range(n)
        let d2:Data = conv(c1, cbomb, 1, 1,0,0,weights,&b1);
        let cdata:Data =concat(&b1, &d2);
        let out = conv(2 * cbomb, c2, 1, 1,0,0, weights, &cdata);  //optional act=FReLU(c2)
        out
    }

/*     fn spp(data:&Data, c1:u16, c2:u16, )->Data{
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

    fn sppf(c1:u16, c2:u16, data:&Data, weights:&Weights)->Data{
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

 
    fn xyxy2xywh(xyxy:[u32;4]){
        // Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
        let x1 = xyxy[0][1];
        let x2 =xyxy[1][1];
        let y1 = xyxy[0][0];
        let y2 = xyxy[1][0];

        let x_center    =   (x2-x1)/ 2       ;        
        let y_center    =   (y2-y1) / 2      ;
        let width       =   x2-x1            ;
        let height:u8   =   y2-y1            ;
        return [x_center,y_center,width,height];
    }
    
    fn output(predictions:Vec, out_path:str){ // or will it be a custom struct?
    
                for prediction in predictions{
                    
                    let class = prediction[0];
                    let conf = prediction[1];
                    let bbox = prediction[2];
                    let out_string = format!("{},{},{}", class, conf, bbox);
                    
                    let mut output = File::create(out_path)?;
                    write!(output, out_string)?;
                }


                        }

                                  //fn count_instances(){
          /*   for c in pred[:, -1].unique():
                n = (pred[:, -1] == c).sum()  // detections per class
                s += f"{n} {names[int(c)]}{"s" * (n > 1)}, "  // add to string
            s = s.rstrip(", ")
        } */